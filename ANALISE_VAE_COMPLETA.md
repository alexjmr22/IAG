# Análise Completa de Experiências VAE — Rondas 1 a 4

> Dataset: WikiArt (20% subset, 32×32)  
> Batch size: 128 em todas as experiências  
> Métricas: **FID** e **KID** — valores **menores são melhores**

---

## Índice

1. [Ronda 1 — VAE: Exploração de hiperparâmetros individuais](#1-ronda-1--vae-exploração-de-hiperparâmetros-individuais)
2. [Ronda 2 — vae2: Refinamento e combinações](#2-ronda-2--vae2-refinamento-e-combinações)
3. [Ronda 3 — VAE3: Melhor config + mais epochs](#3-ronda-3--vae3-melhor-config--mais-epochs)
4. [Ronda 4 — VAE4: Técnicas avançadas e VQ-VAE](#4-ronda-4--vae4-técnicas-avançadas-e-vq-vae)
   - [T1 — Baseline 100 epochs](#t1--baseline-100-epochs-fid1461)
   - [T2 — Cosine LR Scheduling](#t2--cosine-lr-scheduling-fid1643)
   - [T3 — KL Annealing](#t3--kl-annealing-fid1603--bug-crítico)
   - [T4 — Cosine + KL Annealing](#t4--cosine--kl-annealing-fid1760)
   - [VQ-VAE](#vq-vae-fid2834--problema-arquitectural)
5. [Panorama geral — evolução entre rondas](#5-panorama-geral--evolução-entre-rondas)
6. [Testes sugeridos e comparação de features](#6-testes-sugeridos-e-comparação-de-features)

---

## 1. Ronda 1 — VAE: Exploração de hiperparâmetros individuais

**Configuração base (default_vae):** lat=128, beta=0.7, lr=0.001, 30 epochs

### 1.1 Beta sweep — lat=128, lr=0.001, 30 epochs

| Experimento | Beta | FID | KID | Observação |
|---|---|---|---|---|
| vae_beta0 | 0.0 | 336.8 | 0.360 | Sem regularização KL |
| vae_beta01 | **0.1** | **185.3** | **0.172** | **Melhor da ronda** |
| vae_beta05 | 0.5 | 223.2 | 0.206 | |
| default_vae | 0.7 | 241.6 | 0.231 | Config base |
| vae_beta2 | 2.0 | 316.3 | 0.331 | Sobre-regularização |

**Análise:**

- **Beta=0** é o pior resultado: sem penalização KL, o modelo degenera num autoencoder determinístico clássico. O espaço latente não tem estrutura gaussiana e a amostragem de z~N(0,I) produz imagens fora da distribuição aprendida. O FID elevado (336) reflecte directamente a ausência de prior — o decoder não foi treinado para decodificar pontos do prior.
- **Beta=0.1** vence claramente. O peso KL baixo permite ao encoder aprender representações informativas sem ser forçado a colapsar distribuições muito distintas no mesmo ponto. A tensão entre reconstruction e regularização está equilibrada.
- A degradação acima de beta=0.5 é progressiva e esperada: quanto maior o beta, mais o encoder é penalizado por usar o espaço latente, resultando em representações menos ricas. Beta=2.0 é próximo de beta=0 em termos de FID (316 vs 337), mas por razão oposta — sobre-regulação que empurra todas as representações para a gaussiana unitária, destruindo discriminação entre imagens.

### 1.2 Latent dim sweep — beta=0.7, lr=0.001, 30 epochs

| Experimento | Lat | FID | KID |
|---|---|---|---|
| vae_lat16 | 16 | 356.3 | 0.373 |
| vae_lat32 | 32 | 311.1 | 0.326 |
| vae_lat64 | 64 | 234.1 | 0.223 |
| default_vae / vae_lat128 | 128 | 241.6 | 0.231 |
| vae_lat256 | 256 | 240.9 | 0.233 |

**Análise:**

- Lat=16 e 32 são claramente insuficientes para CIFAR/WikiArt 32×32. Uma imagem 32×32×3 tem 3072 dimensões; comprimir para 16 dimensões com beta=0.7 impõe uma taxa de compressão de ~192:1, que a arquitectura não consegue suportar com qualidade.
- A partir de lat=64 há saturação — lat=128 e lat=256 têm FID e KID praticamente iguais (diferença de 0.7 e 0.001 respectivamente). Isto indica que o bottleneck informacional do dataset com beta=0.7 está satisfeito a ~64 dimensões. Aumentar para 256 não traz mais informação; com beta=0.7 alto, o KL penaliza igualmente todas as dimensões, neutralizando o benefício de um espaço maior.
- Nota: lat=64 é ligeiramente melhor que lat=128 neste beta. Com beta elevado, um espaço maior é mais difícil de regularizar uniformemente — há mais dimensões para "desperdiçar" sob pressão KL alta.

### 1.3 Learning rate sweep — lat=128, beta=0.7, 30 epochs

| Experimento | LR | FID | KID |
|---|---|---|---|
| vae_lr1e4 | 0.0001 | 443.7 | 0.527 |
| vae_lr5e4 | 0.0005 | 296.5 | 0.311 |
| default_vae | 0.001 | 241.6 | 0.231 |
| vae_lr5e3 | 0.005 | **210.7** | **0.189** |

> **Nota:** vae_lr1e3 tem experiment_params.md mas sem results.csv, sendo equivalente ao default_vae (lr=0.001).

**Análise:**

- LR=0.0001 é o pior: 30 epochs com lr muito baixo equivale a ~100 epochs com lr normal em termos de gradient steps efectivos. O modelo simplesmente não converge a tempo.
- LR=0.005 supera o default — sugere que 0.001 era ainda conservador para 30 epochs. A taxa de aprendizagem mais alta permite ao modelo percorrer mais rapidamente o landscape de loss em epochs limitados.
- LR=0.01 não foi testado nesta ronda, mas vae2 mostrou ser instável.

**Conclusões Ronda 1:** sweet spot identificado — beta=0.1, LR entre 0.001 e 0.005, lat=64–128.

---

## 2. Ronda 2 — vae2: Refinamento e combinações

### 2.1 Beta fine-sweep — lat=128, lr=0.001, 30 epochs

| Experimento | Beta | FID | KID |
|---|---|---|---|
| vae_beta005 | 0.05 | 205.1 | 0.200 |
| vae_beta01 (R1) | **0.1** | **185.3** | **0.172** |
| vae_beta02 | 0.2 | 187.9 | 0.167 |
| vae_beta1 | 1.0 | 262.3 | 0.261 |

**Análise:**

- Beta=0.1 e 0.2 estão quase empatados no FID (185 vs 188), mas beta=0.2 tem ligeiramente melhor KID (0.167 vs 0.172). Ambos definem a **zona óptima: 0.1–0.2**.
- Beta=0.05 já começa a perder alguma regularização estrutural do espaço latente — o encoder tem pouquíssima pressão para organizar as representações perto do prior, o que começa a aproximar-se do comportamento de beta=0.
- Beta=1.0 confirma a degradação esperada, próximo do default_vae com beta=0.7.

### 2.2 LR fine-sweep — lat=128, beta=0.7, 30 epochs

| Experimento | LR | FID | KID |
|---|---|---|---|
| default (R1) | 0.001 | 241.6 | 0.231 |
| vae_lr2e3 | 0.002 | 207.4 | 0.185 |
| vae_lr1e2 | 0.01 | 456.6 | 0.441 |

**Análise:**

- LR=0.002 melhora claramente sobre 0.001 (207 vs 241). Confirma a tendência da R1: LR ligeiramente mais alto converge melhor em 30 epochs.
- LR=0.01 é instável — o loss oscila e o modelo não converge. Para esta arquitectura e batch size=128, o limite superior da LR está entre 0.005 e 0.01.

### 2.3 Latent 96 — beta=0.7, lr=0.001, 30 epochs

| Experimento | Lat | FID | KID |
|---|---|---|---|
| vae_lat96 | 96 | 232.9 | 0.220 |

Valor intermédio entre lat=64 (234) e lat=128 (241), confirmando saturação na zona 64–128 para beta=0.7.

### 2.4 Combinações — lat=64, 30 epochs

| Experimento | Beta | LR | FID | KID |
|---|---|---|---|---|
| vae_best_combo | 0.1 | 0.001 | 245.2 | 0.236 |
| vae_combo_bold | 0.05 | 0.005 | 231.9 | 0.223 |
| vae_combo_full | 0.1 | 0.005 | 225.0 | 0.215 |

**Análise — resultado inesperado:**

Os combos com lat=64 ficam **abaixo do esperado** e abaixo do melhor da R1 (FID=185 com lat=128, beta=0.1). Porquê?

O erro foi assumir que lat=64 era óptimo para beta=0.1. Mas os dados da R1 mostravam lat=64 como melhor apenas para **beta=0.7**. Com beta baixo (0.1), o encoder tem pouca pressão para comprimir as representações — precisa de mais dimensões para organizar adequadamente o espaço. lat=128 + beta=0.1 (FID=185) supera lat=64 + beta=0.1 (FID=245) em 60 pontos. A relação óptima beta↔lat não é independente.

**Conclusões Ronda 2:** vae2 não superou R1 devido à escolha incorrecta de lat=64 nos combos. O melhor individual mantém-se vae_beta01 da R1 (FID=185). LR=0.002 identificado como novo óptimo.

---

## 3. Ronda 3 — VAE3: Melhor config + mais epochs

**Configuração:** lat=128, lr=0.002 (melhor da R2), refinamento de beta, epochs variáveis

| Experimento | Beta | LR | Epochs | FID | KID |
|---|---|---|---|---|---|
| vae_r3_beta01_lat128_lr2e3_e30 | 0.1 | 0.002 | 30 | 169.6 | 0.155 |
| vae_r3_beta015_lat128_lr2e3_e30 | 0.15 | 0.002 | 30 | 170.7 | 0.154 |
| vae_r3_beta01_lat128_lr2e3_e50 | **0.1** | **0.002** | **50** | **157.7** | **0.144** |

**Análise:**

- Salto significativo face à R1/R2: lr=0.002 + beta=0.1 a 30 epochs já dá FID=169 vs 185 da R1. A melhoria vem inteiramente do aumento de LR — a combinação LR×beta encontrou uma trajectória de treino mais eficiente.
- Beta=0.15 é marginalmente equivalente a 0.1 (170 vs 169), confirmando a zona óptima 0.1–0.2.
- **50 epochs dão um ganho sólido adicional:** FID cai de 169 para 157, uma melhoria de 7%. A trajectória de treino ainda não saturou — a loss continua a descer.

**Conclusão Ronda 3:** mais epochs compensam directamente. O modelo beneficia de treino mais longo com lr=0.002. A configuração óptima até este ponto é **beta=0.1, lat=128, lr=0.002, 50+ epochs**.

---

## 4. Ronda 4 — VAE4: Técnicas avançadas e VQ-VAE

**Configuração base de todas as experiências:** lat=128, beta=0.1, lr=0.002

### Tabela de resultados

| Experimento | Técnica extra | Epochs | FID | KID |
|---|---|---|---|---|
| t1_baseline_100ep | nenhuma | 100 | **146.1** | **0.1355** |
| t2_cosine | Cosine LR scheduling | 50 | 164.3 | 0.1467 |
| t3_kl_annealing | KL warm-up (10ep) | 50 | 160.3 | 0.1419 |
| t4_both | Cosine + KL annealing | 50 | 176.0 | 0.1546 |
| vq_vae_t6_quantized | VQ-VAE (codebook=256) | 50 | 283.4 | 0.2931 |

---

### T1 — Baseline 100 epochs (FID=146.1)

#### Curvas de treino

| Métrica | Epoch 0 | Epoch 100 | Comportamento |
|---|---|---|---|
| Loss total | ~370 | ~75 | Descida suave, **ainda a descer ao epoch 100** |
| Reconstruction | ~340 | ~52 | Descida exponencial + plateau suave |
| KL | ~295 (spike) → ~170 | ~200 | Spike inicial, drop rápido, sobe gradualmente até plateau |

O spike inicial no KL é normal: nas primeiras iterações o encoder ainda não foi treinado para controlar a variância, produzindo distribuições q(z|x) com alta divergência face ao prior. Após estabilização em ~200, o modelo encontrou um equilíbrio saudável: 128 dimensões × ~1.56 nats/dim = encoder activo e informativo sem colapso.

**Sinal crítico: a loss total ainda decresce ao epoch 100.** O modelo não convergiu. Existe margem para melhorar apenas com mais epochs.

#### Análise visual

**Progressão das amostras:**
- *Epoch 10:* ruído estruturado — cores desorganizadas, formas pouco distinguíveis. O decoder ainda não aprendeu a distribuição visual do dataset.
- *Epoch 50:* formas reconhecíveis emergem — arquitectura, figuras, paisagens. Paletas de cor coerentes por imagem.
- *Epoch 100:* melhor qualidade — estruturas mais definidas, diversidade razoável. Blur omnipresente (limitação intrínseca da MSE loss no pixel-space).

**Amostras geradas (generated_samples.png):** visível diversidade de conteúdo e paleta, mas todas as amostras partilham o efeito "aquarela desfocada". Identificam-se figuras, cenas de grupo, paisagens. O efeito é típico de VAE com MSE: o decoder aprende a média das imagens consistentes com cada z, suavizando os detalhes de alta frequência.

**Reconstruções:** preservam composição e paleta. O essencial da imagem sobrevive ao bottleneck. Detalhe fino perdido — brushstrokes, texturas de tinta, detalhes de fundo.

**Interpolação latente:** transição suave entre dois pontos do espaço latente. As frames intermédias são coerentes visualmente mas semanticamente próximas dos extremos — os dois pontos de âncora são cenas de tons semelhantes. Para demonstrar interpolação entre domínios diferentes seria necessário seleccionar dois pontos de classes distintas.

#### Porquê FID=146 e não melhor

1. **Loss MSE no pixel-space** — a função objectivo optimiza diretamente para a média pixel-a-pixel, que é matematicamente equivalente a assumir que a distribuição condicional do decoder é Gaussiana. O resultado é blur inevitável que o FID penaliza fortemente (o Inception network reconhece a distribuição de frequências espaciais, e blur = ausência de high frequencies).
2. **Modelo não convergiu** — como a loss ainda desce ao epoch 100, é de esperar ganhos adicionais com mais treino.
3. **Arquitectura CNN básica** — sem skip connections no decoder, sem normalização por grupos, sem residual blocks.
4. **Dataset apenas 20%** — o modelo vê menos variedade, podendo subajustar regiões menos representadas do espaço de imagens.

---

### T2 — Cosine LR Scheduling (FID=164.3)

#### Curvas de treino

As curvas de Loss, Recon e KL são **visualmente indistinguíveis de T1 até epoch 50**. O KL também estabiliza em ~200. O Cosine Annealing com 50 epochs e T_max=50 reduz a LR de 0.002 para 0 ao longo de todo o treino.

#### Porquê FID=164 e não melhor

A comparação correcta é T2 (cosine, 50ep) vs T1 a 50 epochs. T1 a 50 epochs corresponde a VAE3 (FID=157.7). **T2 com cosine é pior que T1 sem cosine à mesma epoch count (164 vs 158).**

O problema: `CosineAnnealingLR(T_max=50)` faz `lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π·t/T_max))`. Nos últimos ~10 epochs, a LR é quase 0, essencialmente parando o treino prematuramente enquanto o modelo ainda está a convergir. O cosine decay de ciclo único "desperdiça" os últimos epochs num estado de LR quase nula.

#### O que falhou na hipótese

A hipótese era que o cosine LR ajudaria a "assentar" o modelo num mínimo melhor nos últimos epochs. Mas para um modelo que ainda está em aprendizagem activa a 50 epochs, reduzir a LR para quasi-zero limita precisamente o que falta aprender.

---

### T3 — KL Annealing (FID=160.3) — ⚠️ Bug crítico

#### Curvas de treino — diagnóstico do bug

| Métrica | Epoch 0 | Comportamento | Esperado (correcto) |
|---|---|---|---|
| Loss | ~540 | Descida normal | ~340 (apenas recon se beta=0) |
| Recon | ~540 | Descida normal | ~340 |
| **KL** | **~1.6 × 10¹³** | **Colapsa para ~0 no epoch 2–3** | Inicialmente ~0, cresce gradualmente até ~200 |

**O gráfico do KL tem escala 1e13 — não é 200, é 1.6 × 10¹³.** Este é o sinal de diagnóstico central.

#### Mecanismo do bug

A implementação correcta de KL annealing seria:

```python
beta_current = min(1.0, epoch / warmup_epochs) * beta_target
# epoch=0: beta=0.0  → loss = recon_loss + 0 * KL  → Loss ≈ 340
# epoch=5: beta=0.05 → loss = recon_loss + 0.05 * KL
# epoch=10+: beta=0.1 → loss = recon_loss + 0.1 * KL
```

Com esta implementação, **Loss no epoch 0 = apenas recon ≈ 340**, e o KL começa pequeno e cresce.

Mas o que se observa: **Loss no epoch 0 ≈ 540** e **KL ≈ 1.6×10¹³**. Isto indica que no primeiro epoch o beta efectivo é muito superior a 1.0, provavelmente porque:

- O annealing foi implementado como `beta = 1.0` no epoch 0 (sem warm-up correcto), ou
- O KL não está normalizado por batch_size e/ou dimensões latentes — um KL raw não normalizado pode ser `batch_size × lat_dim × KL_per_dim = 128 × 128 × algo_grande`, ou
- A escala de annealing vai em sentido inverso (começa em beta grande e desce).

O valor 10¹³ sugere especificamente ausência de normalização: KL sumado sobre batch e dimensões sem dividir dá `128 × 128 × KL_per_sample_per_dim`. Se o encoder gera distribuições inicialmente desreguladas, KL_per_sample_per_dim pode ser ~600, resultando em 128 × 128 × 600 ≈ 10⁷. Ainda não chega a 10¹³, mas combinado com outros factores de escala arquitectural (convoluções no espaço de features) pode agravar.

#### Consequência: Posterior Collapse

Após a explosão inicial, o KL colapsa para ~0 e permanece lá. Este é o fenómeno de **posterior collapse** (Lucas et al. 2019, Bowman et al. 2016):

O optimizador aprende que, para minimizar a loss com KL ~10¹³ no epoch 0, a solução mais rápida é forçar `q(z|x) → N(0,I)` para todo o x. Quando q(z|x) = p(z) = N(0,I), o KL = 0. O decoder passa então a receber z que não contém informação sobre x e aprende a reconstruir sem usar z, funcionando como um autoencoder determinístico condicionado apenas nos features do decoder.

**Estado pós-collapse:** z~N(0,I) não codifica nada sobre o input. O decoder, para minimizar a reconstruction loss, aprende a produzir a "melhor estimativa média" para cada região do espaço z, sem correspondência com imagens reais específicas.

#### Porquê FID=160 apesar do colapso

O FID não é catastrófico (160 vs 146 do baseline) porque:
1. O decoder ainda foi treinado com real images como alvo durante 50 epochs — aprendeu a produzir imagens pictóricas plausíveis.
2. A reconstrução mede o erro face ao dado real (não face à sample aleatória), por isso a qualidade de reconstrução mantém-se razoável.
3. Ao amostrar z~N(0,I) e decodificar, o resultado não é ruído puro — é uma mistura suave das imagens de treino mais associadas a cada região do espaço z mesmo num estado colapsado parcial.

No entanto, a **diversidade das amostras** é claramente reduzida: amostras tendem a paletas e composições repetitivas, o que o FID detecta como menor qualidade.

#### Reconstruções T3

Visualmente as reconstruções de T3 são funcionais mas com mais blur e menos detalhe que T1. A figura humana, a mão, a figura feminina com vestido — as formas grossas estão lá, mas o detalhe pictórico está mais deteriorado. Consistente com um encoder menos informativo.

---

### T4 — Cosine + KL Annealing (FID=176.0)

O mesmo bug de T3 está presente: KL em 1e13 no epoch 0, colapsa para ~0. A adição de Cosine LR **agrava o problema** de forma compreensível:

- Com T3, a LR mantém-se em 0.002 durante todo o treino. O decoder tem capacidade de se adaptar ao estado colapsado e aprender representações razoáveis mesmo sem z informativo.
- Com T4, a LR decresce para ~0 nos últimos epochs. O modelo fica "frozen" num estado colapsado com pouca LR para adaptação.
- A interacção é: `(KL colapsado) × (LR a zero) = decoder paralisado num estado subóptimo`.

**FID=176 > FID=160 (T3):** a cosine annealing piora o colapso em ~10 pontos FID.

---

### VQ-VAE (FID=283.4) — ⚠️ Problema arquitectural e de geração

#### Parâmetros

```
Arquitectura: Vector Quantized VAE
Latent Dim: 128 (vector único)
Num Embeddings (Codebook): 256
Commitment Loss Weight: 0.25
Learning Rate: 0.002
Epochs: 50
Cosine Annealing LR: True
```

#### Curvas de treino

| Métrica | Epoch 0 | Epoch 50 | Comportamento |
|---|---|---|---|
| Loss | 3.5 | 0.45 | Convergência limpa |
| Recon | 0.18 | 0.045 | Convergência limpa |
| VQ (commitment) | 3.2 | 0.35 | Convergência limpa |

O treino do VQ-VAE **em si é saudável** — não há explosão, não há colapso de loss. O problema não está no treino — está na **geração**.

#### Problema fundamental 1 — Quantização 1D

O VQ-VAE original (van den Oord et al. 2017) quantiza **feature maps espaciais**:

```
Encoder: imagem → feature map [H×W×D]  (e.g. 8×8×64)
Quantização: cada posição (i,j) → índice no codebook
Decoder: feature map quantizado → imagem
```

A capacidade informacional é: `H × W × log₂(K)` bits

Na implementação actual:
```
Encoder: imagem → vector [128]
Quantização: vector inteiro → 1 índice no codebook
Decoder: embedding do índice → imagem
```

Capacidade: `1 × log₂(256) = 8 bits`

Uma imagem 32×32×3 comprimida para **8 bits** é uma taxa de compressão de 384:1. É impossível reconstruir com qualidade. Esta é a causa primária do FID elevado.

#### Problema fundamental 2 — Ausência de prior aprendido

O VQ-VAE **não tem prior gaussiano** como o VAE contínuo. Quando um VAE contínuo gera amostras, usa `z ~ N(0,I)` que é o mesmo prior usado na regularização KL — a distribuição z no treino ≈ a distribuição z na geração.

Para o VQ-VAE, a distribuição dos índices usados no treino (quais os 256 embeddings mais activados e em que contextos) não é uniforme. Se na geração os índices são amostrados aleatoriamente de forma uniforme, a distribuição de inputs do decoder é completamente diferente da distribuição de treino → má qualidade.

**A geração correcta de VQ-VAE exige um prior separado** treinado sobre os índices discretos:
- PixelCNN autoregressivo (van den Oord et al. 2017)
- Transformer autoregressivo (VQ-VAE-2, Razavi et al. 2019)
- Gumbel-softmax com learned prior (dVAE, Ramesh et al. 2021)

#### Reconstruções VQ-VAE

Revelam **codebook collapse parcial**: algumas imagens reconstruem surpreendentemente bem (retrato com chapéu vermelho — imagem de alta frequência preservada) enquanto outras colapsam completamente (piano → blob castanho; paisagem → mancha monocromática). A inconsistência indica que parte dos 256 embeddings está activa e especializada, mas outra parte é pouco usada ou inactiva, não conseguindo capturar certos conteúdos visuais.

Paradoxo aparente: reconstruções boas + geração má. Confirma o diagnóstico — reconstrução usa o encoder para encontrar o índice certo; geração usa índices aleatórios que podem não corresponder a embeddings bem treinados.

---

## 5. Panorama Geral — Evolução entre rondas

### Melhor resultado de cada ronda

| Ronda | Experimento | Config resumida | FID | KID | ∆FID |
|---|---|---|---|---|---|
| VAE (R1) | vae_beta01 | β=0.1, lat=128, lr=0.001, 30ep | 185.3 | 0.172 | baseline |
| vae2 (R2) | vae_beta02 | β=0.2, lat=128, lr=0.001, 30ep | 187.9 | 0.167 | +2.6 |
| VAE3 (R3) | r3_beta01_e50 | β=0.1, lat=128, lr=0.002, 50ep | 157.7 | 0.144 | **-27.6** |
| VAE4 (R4) | t1_baseline_100ep | β=0.1, lat=128, lr=0.002, 100ep | **146.1** | **0.135** | **-39.2** |

### Todos os experimentos ordenados por FID

| # | Experimento | Ronda | FID | KID |
|---|---|---|---|---|
| 1 | vae_r5_t1_baseline_100ep | VAE4 | 146.1 | 0.1355 |
| 2 | vae_r5_t3_kl_annealing | VAE4 | 160.3 | 0.1419 |
| 3 | vae_r3_beta01_lat128_lr2e3_e50 | VAE3 | 157.7 | 0.1444 |
| 4 | vae_r5_t2_cosine | VAE4 | 164.3 | 0.1467 |
| 5 | vae_r5_t4_both | VAE4 | 176.0 | 0.1546 |
| 6 | vae_r3_beta015_lat128_lr2e3_e30 | VAE3 | 170.7 | 0.1541 |
| 7 | vae_r3_beta01_lat128_lr2e3_e30 | VAE3 | 169.6 | 0.1548 |
| 8 | vae_beta01 | VAE | 185.3 | 0.1721 |
| 9 | vae_beta02 | vae2 | 187.9 | 0.1668 |
| 10 | vae_beta005 | vae2 | 205.1 | 0.2000 |
| 11 | vae_lr2e3 | vae2 | 207.4 | 0.1850 |
| 12 | vae_lr5e3 | VAE | 210.7 | 0.1891 |
| 13 | vae_combo_full | vae2 | 225.0 | 0.2149 |
| 14 | vae_beta05 | VAE | 223.2 | 0.2059 |
| 15 | vae_combo_bold | vae2 | 231.9 | 0.2227 |
| 16 | vae_lat96 | vae2 | 232.9 | 0.2201 |
| 17 | vae_lat64 | VAE | 234.1 | 0.2226 |
| 18 | default_vae / vae_lat128 | VAE | 241.6 | 0.2310 |
| 19 | vae_best_combo | vae2 | 245.2 | 0.2361 |
| 20 | vae_beta1 | vae2 | 262.3 | 0.2612 |
| 21 | vq_vae_r5_t6_quantized | VAE4 | 283.4 | 0.2931 |
| 22 | vae_lr5e4 | VAE | 296.5 | 0.3110 |
| 23 | vae_lat32 | VAE | 311.1 | 0.3261 |
| 24 | vae_beta2 | VAE | 316.3 | 0.3311 |
| 25 | vae_beta0 | VAE | 336.8 | 0.3596 |
| 26 | vae_lat16 | VAE | 356.3 | 0.3730 |
| 27 | vae_lr1e4 | VAE | 443.7 | 0.5267 |
| 28 | vae_lr1e2 | vae2 | 456.6 | 0.4414 |

### Factores de impacto identificados

| Factor | Impacto FID | Direcção | Nota |
|---|---|---|---|
| Beta: 0.7 → 0.1 | -56 | ↓ melhor | Maior ganho individual da R1 |
| Epochs: 30 → 100 | -39 | ↓ melhor | Maior ganho transversal |
| LR: 0.001 → 0.002 | -34 | ↓ melhor | Identificado na R2 |
| KL annealing (com bug) | +14 vs baseline | ↑ pior | Bug causa collapse |
| Cosine LR | +18 vs baseline (50ep) | ↑ pior | Pior que baseline sem tricks |
| VQ-VAE vs VAE | +137 | ↑ muito pior | Sem prior; quantização 1D |

---

## 6. Testes Sugeridos e Comparação de Features

### 6.1 Para o VAE contínuo — testes de melhoria

#### Testes de alta prioridade

| # | Teste | Config sugerida | Paper de referência | Ganho esperado | Justificação |
|---|---|---|---|---|---|
| **A1** | **Perceptual loss** | `loss = MSE + λ·||φ(x̂) - φ(x)||₂` com φ = VGG-16 relu3_3, λ=0.1 | Johnson et al. 2016, "Perceptual Losses for Real-Time Style Transfer" | -30 a -50 FID | Elimina blur ao optimizar no feature space em vez do pixel space; o Inception network do FID usa features similares |
| **A2** | **Baseline 150–200 epochs** | β=0.1, lat=128, lr=0.002 | — | -10 a -20 FID | A loss ainda desce ao epoch 100; análise de convergência mais completa |
| **A3** | **Free bits** | `KL_loss = Σ max(KL_dim, λ)` com λ=1.0 | Kingma et al. 2016, "Improving VAEs with IAF" | -5 a -10 FID + previne collapse | Garante que cada dimensão latente usa pelo menos λ nats; imune ao posterior collapse independentemente do beta |
| **A4** | **KL annealing corrigido** | Warmup 20 epochs, beta 0→0.1, lr=0.002, 100ep | Bowman et al. 2016, "Generating Sentences from a Continuous Space" | -5 a -15 FID vs T3/T4 actuais | Corrigir o bug + more epochs permitirá avaliar o benefício real do annealing |

#### Testes de investigação

| # | Teste | Config sugerida | Paper | Objectivo |
|---|---|---|---|---|
| **A5** | **β-VAE com β=4** | β=4, lat=64, lr=0.002, 100ep | Higgins et al. 2017 (ICLR), "β-VAE: Learning Basic Visual Concepts" | Avaliar disentanglement com sacrifice de FID — β alto força representações factorizadas |
| **A6** | **β-TCVAE** | decomposição MI + TC + DW-KL | Chen et al. 2018 (NeurIPS), "Isolating Sources of Disentanglement in VAEs" | Medir separadamente: Mutual Information (encoder informativo?), Total Correlation (dimensões independentes?), Dimension-wise KL (uso uniforme?) |
| **A7** | **Cyclical KL annealing** | 4 ciclos de 25ep cada, β: 0→0.1 por ciclo | Fu et al. 2019 (NAACL), "Cyclical Annealing Schedule" | Alternativa robusta ao annealing monotónico — previne collapse mais eficazmente ao re-expor o modelo a β=0 ciclicamente |
| **A8** | **ResNet decoder** | Substituir upconv layers por residual blocks | Vahdat & Kautz 2020 (NeurIPS), "NVAE" | Avaliar se o bottleneck é a loss ou a capacidade arquitectural |

### 6.2 Para o VQ-VAE — testes obrigatórios

Os testes abaixo são necessários para uma comparação justa com o VAE contínuo, dado que a implementação actual não corresponde ao VQ-VAE canónico.

| # | Teste | O que mudar | Paper | Justificação |
|---|---|---|---|---|
| **V1** | **Spatial latent maps** | Encoder: img → 4×4×64, quantizar cada posição independentemente | van den Oord et al. 2017 (NeurIPS), "Neural Discrete Representation Learning" | Aumenta capacidade de 8 bits para `4×4×log₂(256) = 128 bits`; arquitectura canónica do VQ-VAE |
| **V2** | **EMA codebook updates** | Substituir straight-through gradient por Exponential Moving Average | van den Oord 2017, Appendix A | EMA elimina praticamente o codebook collapse; mais estável que gradientes straight-through |
| **V3** | **Codebook reset** | Reiniciar periodicamente embeddings com usage < threshold | Dhariwal et al. 2020 | Previne embeddings "mortos" (nunca activados) que desperdiçam capacidade do codebook |
| **V4** | **PixelCNN prior** | Treinar PixelCNN sobre índices latentes após treino do VQ-VAE | van den Oord et al. 2017 | Sem este prior, a geração é aleatória e incoerente. Com prior, FID esperado <200 |
| **V5** | **VQ-VAE-2 hierarchical** | Top level: 4×4×D, Bottom level: 8×8×D, prior separado por nível | Razavi et al. 2019 (NeurIPS), "Generating Diverse High-Fidelity Images with VQ-VAE-2" | Estado da arte em geração discreta; captura estrutura global (top) e detalhe local (bottom) |
| **V6** | **VQGAN** | VQ-VAE + perceptual loss + discriminador adversarial | Esser et al. 2021 (CVPR), "Taming Transformers for High-Resolution Image Synthesis" | O discriminador força realismo local que nem MSE nem perceptual loss garantem; base do Stable Diffusion |

### 6.3 Métricas de avaliação adicionais

Além de FID e KID, as seguintes métricas são standard nos papers e permitiriam comparações mais informativas:

| Métrica | O que mede | Relevante para | Como computar |
|---|---|---|---|
| **LPIPS** (Zhang et al. 2018) | Similaridade perceptual nas reconstruções | VAE contínuo | `lpips.LPIPS(net='vgg')(recon, original)` — correlaciona melhor com percepção humana que MSE |
| **SSIM** | Estrutura, luminância, contraste | VAE contínuo | `skimage.metrics.structural_similarity` — métrica standard em compression |
| **Active Units (AU)** | Quantas dimensões latentes têm variância > threshold (e.g. 0.01) | VAE + VQ-VAE | `Var_x[E[z_i|x]] > threshold` — diagnóstico directo de posterior collapse; T3/T4 terão AU << 128 |
| **Mutual Information Gap (MIG)** | Disentanglement — cada factor de variação mapeado a dimensão latente separada | β-VAE, β-TCVAE | Chen et al. 2018; requer ground truth factors (WikiArt: estilo, artista, época) |
| **Codebook Utilization %** | Percentagem de embeddings activos no VQ-VAE | VQ-VAE | `len(unique_indices) / num_embeddings × 100` — valor <50% indica collapse |
| **Reconstruction FID** | FID calculado nas reconstruções (não amostras) | Todos | Separa capacidade de reconstrução da qualidade do prior/espaço latente |
| **Precision & Recall** (Kynkäänniemi 2019) | Precision = fidelidade das amostras; Recall = cobertura do dataset | Todos | Distingue modelos que são fiéis mas pouco diversos (alto P, baixo R) dos que são diversos mas pouco fiéis (baixo P, alto R) |

### 6.4 Comparação de features — experimentos de análise do espaço latente

Estes testes não melhoram o FID directamente mas são essenciais para compreender o que o modelo aprendeu:

| Teste | Procedimento | O que revela |
|---|---|---|
| **Latent traversal** | Fixar z em z_mean, variar cada dimensão de -3σ a +3σ, decodificar | Quais dimensões capturam que atributos visuais (cor, estilo, composição) |
| **Interpolação entre classes** | Seleccionar z de imagens de classes opostas (retrato vs paisagem), interpolação linear | Suavidade da transição, se há descontinuidades no espaço latente |
| **Clustering latente** | t-SNE/UMAP das representações z em imagens do dataset | Se o espaço latente organiza semanticamente imagens por estilo/época/artista sem supervisão |
| **Direcções semânticas** | Treinar classificador de atributos no espaço latente (e.g. "retrato" vs "não-retrato") | Se a estrutura latente é linearmente separável por conceitos visuais (análogo ao espaço GAN) |
| **Posterior sharpness** | Histograma de log-var por dimensão | Dimensões com log-var ≈ 0 (prior) não contribuem para encoding; indicador de uso eficiente |

---

## Sumário executivo

| Pergunta | Resposta baseada nos dados |
|---|---|
| **Qual o melhor modelo?** | T1 baseline 100ep: β=0.1, lat=128, lr=0.002, sem tricks — FID=146.1 |
| **Qual o factor mais impactante?** | Beta=0.1 (-56 FID vs default β=0.7) + mais epochs (-39 FID de R1 para R4) |
| **O que falhou?** | KL annealing tem bug (colapso KL 1e13); VQ-VAE sem prior e com quantização 1D |
| **Próximo passo de maior impacto?** | Perceptual loss (VGG) — pode reduzir FID 30-50 pontos eliminando o blur |
| **Próximo passo para VQ-VAE?** | Spatial latent + EMA codebook + PixelCNN prior — sem estes, o VQ-VAE não é comparável |
| **O modelo converge?** | Não — a loss ainda desce ao epoch 100; mais epochs esperados trazer FID <130 |

---

*Análise realizada em 2026-04-23 com base nos resultados das pastas `Results_vae/VAE`, `Results_vae/vae2`, `Results_vae/VAE3` e `Results_vae/VAE4`.*
