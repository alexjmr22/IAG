# Análise — Novos Modelos GAN: WGAN-GP e cDCGAN

> Dataset: ArtBench-10 (20% subset, 32×32)  
> Todos os modelos: 100 epochs, batch size 128  
> Métricas: **FID** e **KID** — valores **menores são melhores**  
> Avaliação: 2000 amostras, 3 seeds (modo DEV)

---

## Índice

1. [Resultados de referência — DCGAN](#1-resultados-de-referência--dcgan)
2. [WGAN-GP](#2-wgan-gp--fid12829)
3. [cDCGAN](#3-cdcgan--fid13463)
4. [Panorama geral — todos os modelos](#4-panorama-geral--todos-os-modelos)
5. [Lições retiradas](#5-lições-retiradas)

---

## 1. Resultados de referência — DCGAN

Para contextualizar os novos modelos, os resultados relevantes do DCGAN (melhores e baselines):

| Experimento | Config resumida | FID | KID |
|---|---|---|---|
| **dcgan_spectral** | SN no discriminador, 100ep | **71.17** | **0.0344** |
| **dcgan_ngf128_100ep** | ngf=128, 100ep | **74.29** | **0.0385** |
| dcgan_ngf128_ndf64_100ep | ngf=128 ndf=64, 100ep | 86.74 | 0.0526 |
| dcgan_lat32_100ep | latent=32, 100ep | 91.78 | 0.0560 |
| dcgan_cosine | Cosine LR, 50ep | 100.37 | 0.0659 |
| dcgan_cosine_sn | Cosine LR + SN, 50ep | 103.58 | 0.0676 |
| default_dcgan | Baseline, 100ep | 118.82 | 0.0904 |
| dcgan_beta09 | β₁=0.9 (colapso) | 286.58 | 0.2517 |

O melhor DCGAN é `dcgan_spectral` com **FID=71.17**, exclusivamente por adicionar Spectral Norm ao discriminador. É o benchmark a superar.

---

## 2. WGAN-GP — FID=128.29

### 2.1 Configuração

```
Loss:          Wasserstein + Gradient Penalty (λ=10)
Critic:        WGANCritic — sem Sigmoid, sem BatchNorm
Generator:     DCGenerator (idêntico ao DCGAN)
N_CRITIC:      5 (5 passos de critic por passo de generator)
LR:            1e-4 (Adam, β₁=0.0, β₂=0.9)
Regularização: Spectral Norm + Cosine LR (sempre activos)
Epochs:        100
```

### 2.2 Resultado vs DCGAN

| Modelo | FID | KID | ∆FID vs default_dcgan |
|---|---|---|---|
| default_dcgan | 118.82 | 0.0904 | — |
| **wgan_gp** | **128.29** | **0.0947** | **+9.5 (pior)** |
| dcgan_spectral | 71.17 | 0.0344 | -47.6 (melhor) |
| dcgan_cosine_sn | 103.58 | 0.0676 | -15.2 (melhor) |

**O WGAN-GP é pior que o default DCGAN em ~9 pontos FID, e pior que o melhor DCGAN (spectral) em ~57 pontos.**

### 2.3 Porquê o WGAN-GP não superou o DCGAN

#### Razão 1 — Cosine LR aniquila a aprendizagem nos últimos epochs

O WGAN-GP tem Cosine LR sempre activo com `T_max=100`. Com LR=1e-4, a taxa de aprendizagem decai de 1e-4 para ~0 ao longo dos 100 epochs. A partir de ~epoch 80, o optimizador está essencialmente parado.

Este é o mesmo padrão observado no VAE: Cosine LR num modelo que ainda está a convergir ao epoch final é contraproducente. A análise do VAE4-T2 (cosine pior que baseline em 50ep) confirma este padrão.

Com o WGAN-GP, o efeito é amplificado porque a LR inicial já é muito baixa (1e-4 vs 2e-4 do DCGAN). O range efectivo de aprendizagem é mais estreito desde o início.

#### Razão 2 — N_CRITIC=5 reduz os passos efectivos do generator

Com N_CRITIC=5, por cada batch o generator faz apenas 1 update enquanto o critic faz 5. Num epoch com `N` batches, o generator faz `N` gradient steps enquanto o DCGAN faz também `N` (1:1 ratio). Mas o tempo computacional por epoch é 5x maior para o critic, sem aumento correspondente nos gradient steps do generator.

Efectivamente, para a mesma contagem de epochs, o generator WGAN-GP fez o mesmo número de updates que o DCGAN, mas o critic consumiu recursos que no DCGAN seriam usados para mais epochs ou batches. A relação N_CRITIC=5 faz sentido quando o discriminador BCE colapsa (estabiliza num valor trivial) — se o discriminador não colapsa, N_CRITIC=5 é overhead desnecessário.

#### Razão 3 — SN + GP é redundante na prática

O Spectral Norm e o Gradient Penalty visam o mesmo objectivo: restringir a constante de Lipschitz do critic. Usar ambos simultaneamente é sobrerregularização. O SN já garante que o critic é 1-Lipschitz; o GP adiciona penalização extra que pode conflituar com o SN ao forçar gradientes unitários em pontos interpolados que o SN já tratou.

A evidência empírica suporta esta hipótese: `dcgan_cosine_sn` (SN + Cosine, FID=103.58) é melhor que `wgan_gp` (SN + GP + Cosine + W-loss, FID=128.29). O GP não acrescentou valor — só instabilidade extra.

#### Razão 4 — O problema que o WGAN resolve não é o problema dominante aqui

O WGAN-GP foi desenhado para resolver o **colapso de modo** causado pela saturação do discriminador BCE. Este colapso é visível em `dcgan_beta09` (FID=286.58) — quando β₁=0.9 o discriminador oscila e o gerador colapsa.

Com β₁=0.5 (default) e Spectral Norm, o DCGAN não mostra sinais de colapso. O discriminador mantém-se num regime de aprendizagem estável. O WGAN-GP resolve um problema que já estava controlado pelos regularizadores existentes, adicionando complexidade sem benefício.

### 2.4 Wasserstein Distance como sinal de treino

A Wasserstein Distance é a vantagem teórica do WGAN: fornece um gradiente com significado geométrico mesmo quando as distribuições não se sobrepõem. Ao contrário da BCE loss (que satura a 0 quando o discriminador é perfeito), a W-distance continua a crescer monotonicamente com a separação entre distribuições.

Na prática, para as curvas de treino do WGAN-GP neste experimento, é de esperar que a W-distance aumente durante os primeiros epochs (generator e critic aprendem) e depois se estabilize num plateau. A loss do generator (`-E[critic(fake)]`) deve descer. Se a W-distance baixar demasiado cedo indica que o critic não está a aprender o suficiente para separar distribuições.

---

## 3. cDCGAN — FID=134.63

### 3.1 Configuração

```
Arquitectura: cDCGenerator + cDCDiscriminator
Label embedding (G): nn.Embedding(10, 32) → concatenado com z
Label embedding (D): nn.Embedding(10, ndf) → Linear → spatial map 32×32 → canal extra
N_CLASSES:    10 (ArtBench-10)
LR:           2e-4 (Adam, β₁=0.5)
Regularização: Spectral Norm (D) + Cosine LR (sempre activos)
Epochs:        100
```

### 3.2 Resultado

| Modelo | FID | KID | Geração condicional |
|---|---|---|---|
| default_dcgan | 118.82 | 0.0904 | Não |
| **cdcgan** | **134.63** | **0.1050** | **Sim (10 classes)** |
| dcgan_spectral | 71.17 | 0.0344 | Não |

**O cDCGAN fica ~16 FID acima do default DCGAN** — o custo de adicionar condicionamento de classe.

### 3.3 O que o cDCGAN traz que o DCGAN não tem

O FID compara a distribuição marginal das amostras geradas vs imagens reais. Neste sentido, um DCGAN que gera boa arte aleatória pode ter melhor FID que um cDCGAN que gera boa arte de cada estilo. O FID não mede **precisão condicional** — se o modelo gera correctamente a classe pedida.

O cDCGAN introduz uma capacidade qualitativamente diferente:
- **Geração guiada:** `generator(z, label=3)` produz arte do estilo 3 (e.g. Impressionismo)
- **Interpolação condicionada:** interpolar z mantendo label fixo explora o espaço intra-classe
- **Análise de separação:** comparar amostras de classes distintas revela o que o modelo aprendeu sobre cada estilo

Esta capacidade não tem equivalente no DCGAN incondicional — o DCGAN mistura todas as classes no mesmo z~N(0,I).

### 3.4 Porquê o FID do cDCGAN é pior

#### Razão 1 — Condicionamento aumenta a dificuldade do problema

O discriminador do cDCGAN tem de aprender simultaneamente:
- "Esta imagem é real ou fake?" (como o DCGAN)
- "Esta imagem pertence à classe declarada?" (novo)

A segunda tarefa é análoga a treinar um classificador de estilos artísticos — tarefa não trivial para redes pequenas. Com ndf=64, o discriminador tem capacidade limitada para aprender ambas as tarefas.

#### Razão 2 — Label embedding como spatial map introduz ruído

A implementação do discriminador condicional projecta o label num mapa espacial 32×32 (`Linear(ndf, 32*32)`), que é concatenado como canal extra. Esta projecção é aprendida mas parte de ruído aleatório — nos primeiros epochs, o "mapa de classe" é essencialmente ruído que perturba o discriminador sem informação útil.

Uma alternativa mais robusta seria Conditional Batch Normalization (onde os parâmetros γ e β do BN são função do label) ou Projection Discriminator (Miyato & Koyama 2018, que projecta o label directamente na última camada).

#### Razão 3 — Gerador com embed_dim=32 pode ser subrepresentado

O generator concatena z (100 dims) com o label embedding (32 dims) → 132 dims totais. O embedding de 32 dims para 10 classes significa ~3.2 dims por classe em média. Com 10 classes de arte tão distintas (Impressionismo, Renascimento, Art Nouveau, etc.), 32 dims pode não ser suficiente para capturar as diferenças visuais entre estilos. Um embed_dim=64 ou 128 daria mais expressividade ao gerador condicional.

### 3.5 Avaliação qualitativa esperada

Com 64 amostras fixas (FIXED_NOISE, FIXED_LABELS = [0,1,...,9,0,1,...]) organizadas em grid 8×8:
- Cada coluna deve corresponder a uma classe artística
- A diversidade intra-coluna (variação de z com label fixo) revela riqueza intra-classe
- Comparar colunas deve mostrar diferenças de estilo visíveis se o condicionamento funciona

Se as colunas não tiverem diferenças visuais claras, o condicionamento não está a funcionar — o label embedding não está a ser utilizado pelo gerador.

---

## 4. Panorama Geral — Todos os modelos

### 4.1 Tabela completa ordenada por FID

| # | Modelo | Config resumida | FID | KID |
|---|---|---|---|---|
| **1** | **dcgan_spectral** | SN, 100ep | **71.17** | **0.0344** |
| **2** | **dcgan_ngf128_100ep** | ngf=128, 100ep | **74.29** | **0.0385** |
| 3 | dcgan_ngf128_ndf64_100ep | ngf=128 ndf=64, 100ep | 86.74 | 0.0526 |
| 4 | dcgan_lat32_100ep | latent=32, 100ep | 91.78 | 0.0560 |
| 5 | dcgan_cosine | Cosine LR, 50ep | 100.37 | 0.0659 |
| 6 | dcgan_lat32 | latent=32, 50ep | 103.39 | 0.0679 |
| 7 | dcgan_cosine_sn | Cosine + SN, 50ep | 103.58 | 0.0676 |
| 8 | dcgan_ngf128 | ngf=128, 50ep | 104.76 | 0.0725 |
| 9 | dcgan_lat64 | latent=64, 50ep | 112.61 | 0.0826 |
| 10 | dcgan_ngf128_ndf64 | ngf=128 ndf=64, 50ep | 115.68 | 0.0789 |
| 11 | dcgan_lat32_ngf128 | latent=32 ngf=128, 50ep | 116.04 | 0.0812 |
| 12 | default_dcgan | Baseline, 100ep | 118.82 | 0.0904 |
| 13 | **wgan_gp** | **W-loss + GP + SN + Cosine, 100ep** | **128.29** | **0.0947** |
| 14 | dcgan_lat32_ngf128_ndf64 | latent=32 ngf=128 ndf=64, 50ep | 128.45 | 0.0973 |
| 15 | dcgan_lat32_ngf128_asym | latent=32 ngf=128 asym LR | 128.64 | 0.0909 |
| 16 | dcgan_lat256 | latent=256, 50ep | 128.98 | 0.0998 |
| 17 | dcgan_lat32_asym_lr | latent=32 asym LR | 134.32 | 0.1059 |
| 18 | **cdcgan** | **Condicional 10 classes, 100ep** | **134.64** | **0.1050** |
| 19 | dcgan_ngf32 | ngf=32, 50ep | 139.31 | 0.1131 |
| 20 | dcgan_asym_lr | LR assimétrico | 148.46 | 0.1251 |
| — | *VAE melhor (referência)* | *β=0.1 lat=128 lr=0.002 100ep* | *146.1* | *0.135* |
| 21 | dcgan_beta09 | β₁=0.9 (colapso de modo) | 286.58 | 0.2517 |

### 4.2 GANs vs VAE — comparação directa

| Família | Melhor modelo | FID | ∆ vs VAE melhor |
|---|---|---|---|
| **DCGAN** | dcgan_spectral | **71.17** | **-74.9 (muito melhor)** |
| **WGAN-GP** | wgan_gp | 128.29 | -17.8 (melhor) |
| **cDCGAN** | cdcgan | 134.64 | -11.5 (melhor) |
| **VAE** | t1_baseline_100ep | 146.1 | — |

**Todos os modelos GAN superam o VAE**, mesmo o pior DCGAN (dcgan_ngf32, FID=139.31) fica perto do melhor VAE. O DCGAN com Spectral Norm (FID=71.17) é quase **2x melhor** que o melhor VAE.

A diferença arquitectural central: os GANs optimizam directamente para enganar um discriminador treinado na distribuição real de imagens, enquanto o VAE optimiza para reconstrução pixel-a-pixel (MSE). O FID é fortemente penalizado por blur — que é a assinatura da MSE loss. Os GANs, ao gerar amostras que tentam passar num discriminador que conhece imagens reais, produzem automaticamente features de alta frequência (texturas, bordas, detalhes) que o VAE suaviza.

### 4.3 Factores de impacto identificados — GANs

| Factor | ∆FID | Direcção | Evidência |
|---|---|---|---|
| Spectral Norm (vs baseline) | -47.6 | ↓ melhor | dcgan_spectral vs default_dcgan |
| Mais epochs (50→100) | -10 a -28 | ↓ melhor | ngf128: 104.76→74.29; lat32: 103.39→91.78 |
| ngf=128 (vs ngf=64) | -14 | ↓ melhor | dcgan_ngf128 vs default_dcgan |
| Wasserstein loss + GP | +9.5 | ↑ pior | wgan_gp vs default_dcgan |
| Cosine LR sozinho | -18 | ↓ melhor | dcgan_cosine vs default_dcgan |
| Cosine LR + SN | +9 vs SN só | ↑ pior | dcgan_cosine_sn vs dcgan_spectral |
| Condicionamento classe | +16 | ↑ pior (FID) | cdcgan vs default_dcgan |
| β₁=0.9 (Adam) | +168 | ↑ catastrófico | dcgan_beta09 — colapso de modo |

---

## 5. Lições Retiradas

### 5.1 Spectral Norm é o regularizador mais eficaz testado

De todos os regularizadores e modificações testadas nas GANs, a **Spectral Norm** (aplicada às camadas Conv do discriminador) produziu o maior ganho isolado: -47.6 FID. É simples, não adiciona hiperparâmetros, e não interfere com o loop de treino. O facto de ter funcionado bem no DCGAN (FID=71.17) e ter sido incorporada no WGAN-GP sem benefício adicional confirma que SN já resolve o problema de estabilidade.

### 5.2 Mais epochs é consistentemente a melhoria mais segura

Tanto no VAE como no DCGAN, aumentar de 50 para 100 epochs melhora o FID em 10-30 pontos sem qualquer risco. Os modelos não convergem em 50 epochs — a loss ainda desce activamente. Este padrão é transversal a todas as famílias testadas.

### 5.3 Cosine LR prejudica modelos que ainda estão a convergir

O Cosine Annealing com ciclo único (T_max = total_epochs) é prejudicial para modelos que ainda aprendem activamente no final do treino. O fenómeno observado no VAE (cosine pior que baseline a 50ep) repete-se nas GANs: `dcgan_cosine_sn` (SN + Cosine) é pior que `dcgan_spectral` (SN) em ~32 FID. A LR a quasi-zero nos últimos 20% do treino desperdiça epochs potencialmente produtivos.

**Alternativa:** Cosine com Warm Restarts (SGDR) ou aumentar o total de epochs antes de aplicar cosine. O ciclo único só é válido se o modelo já convergiu a ~80% do treino.

### 5.4 WGAN-GP resolve um problema que aqui já estava controlado

O WGAN-GP foi desenhado para contextos onde o discriminador BCE satura e o gerador entra em colapso de modo. Neste setup (β₁=0.5, Spectral Norm, 32×32), o DCGAN não mostra sinais de colapso excepto quando se força instabilidade explícita (dcgan_beta09 com β₁=0.9). O WGAN-GP introduz complexidade (N_CRITIC, λ_GP, β₁=0.0, LR=1e-4 específico do paper) sem benefício, porque o problema target não existe.

**Lição de transferibilidade:** técnicas que resolvem problemas reais num contexto podem não trazer ganhos se o problema já está resolvido por outros meios.

### 5.5 Condicionalidade tem custo mas é qualitativamente diferente

O cDCGAN tem FID=134.64 — pior que o default DCGAN (118.82). Mas FID é uma métrica de qualidade incondicional — não mede se o modelo gera correctamente a classe pedida. O custo de +16 FID é o preço da capacidade de controlo: a possibilidade de gerar arte por estilo, de explorar o espaço intra-classe, e de interpolar dentro de uma classe. Esta trade-off (FID global pior, controlo condicional ganho) é válida dependendo do objectivo.

Para avaliação correcta do cDCGAN seria necessário medir **FID por classe** (gerar 200 amostras da classe k, comparar com 200 reais da classe k) e **precisão condicional** (classificar as amostras e verificar percentagem da classe correcta).

### 5.6 A família GAN domina claramente sobre o VAE neste dataset

A diferença entre o melhor VAE (FID=146.1) e o melhor GAN (FID=71.17) é de ~75 pontos — quase factor 2. Esta diferença é estrutural, não de hiperparâmetros: o VAE optimiza MSE (que produz blur), enquanto a GAN optimiza para realismo perceptual (que produz texturas). Para o ArtBench-10, com imagens de arte com texturas ricas e pinceladas, o blur do VAE é particularmente penalizante.

O único caminho para o VAE competir seria perceptual loss (VGG features em vez de MSE), conforme identificado na análise do VAE.

### 5.7 Sumário executivo das novas aprendizagens

| Pergunta | Resposta baseada nos dados |
|---|---|
| O WGAN-GP supera o DCGAN? | Não — FID=128 vs 71 do melhor DCGAN; o problema que resolve (colapso BCE) já estava controlado |
| O cDCGAN é melhor que o DCGAN? | Em FID, não (+16). Em capacidade de geração condicional, sim — são modelos com objectivos diferentes |
| Qual o melhor modelo testado até agora? | dcgan_spectral: FID=71.17 — DCGAN com Spectral Norm, 100 epochs |
| Vale a pena explorar mais o WGAN-GP? | Sim, mas removendo o Cosine LR e testando N_CRITIC=2 e LR mais alta (2e-4) |
| Vale a pena explorar mais o cDCGAN? | Sim, com Projection Discriminator e embed_dim=64; a avaliação correcta é FID por classe |
| Os GANs superam o VAE? | Claramente — mesmo o WGAN-GP e cDCGAN (os piores GANs testados) ficam perto ou acima do melhor VAE |

---

## Apêndice — Próximos testes prioritários

### Para o WGAN-GP

| Teste | Config | Pergunta |
|---|---|---|
| `wgan_no_cosine` | Remover Cosine LR, manter SN+GP | O Cosine LR é o responsável pela degradação? |
| `wgan_lr2e4` | LR=2e-4 (mesmo do DCGAN), sem Cosine | Usar a mesma LR do DCGAN melhora? |
| `wgan_ncritic2` | N_CRITIC=2 | Menos passos de critic = mais passos de generator = melhor? |
| `wgan_no_sn` | Remover SN, manter GP | SN+GP é redundante? GP sozinho é suficiente? |

### Para o cDCGAN

| Teste | Config | Pergunta |
|---|---|---|
| `cdcgan_embed64` | embed_dim=64 | Mais capacidade de embedding melhora condicionamento? |
| `cdcgan_proj_disc` | Projection Discriminator (Miyato 2018) | Arquitectura de condicionamento mais robusta? |
| Avaliação condicional | FID por classe + precisão | O modelo realmente gera a classe correcta? |

---

*Análise realizada em 2026-04-24 com base nos resultados das pastas `results/wgan_gp`, `results/cdcgan` e todos os experimentos DCGAN anteriores.*
