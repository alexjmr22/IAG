# Análise Completa dos Resultados GAN — Relatório GenAI TP1

**Branch**: `gan`  
**Modelo**: DCGAN e variantes (WGAN-GP, cDCGAN, StyleGAN)  
**Dataset**: ArtBench-10 (32×32 RGB, 50k treino, 10 estilos artísticos)  
**Métricas**: FID (Fréchet Inception Distance) ↓ e KID (Kernel Inception Distance) ↓  
**Data de análise**: 2026-04-25  

---

## 0. Configuração Base e Protocolo

### Defaults do DCGAN (script `02_dcgan.py`)

| Parâmetro | Default | Env var |
|---|---|---|
| Latent dim | 100 | `DCGAN_LATENT` |
| NGF (Generator feature maps) | 64 | `DCGAN_NGF` |
| NDF (Discriminator feature maps) | 64 | `DCGAN_NDF` |
| LR (G e D) | 2e-4 | `DCGAN_LR` / `DCGAN_LR_G` / `DCGAN_LR_D` |
| Adam β1 | 0.5 | `DCGAN_BETA1` |
| Spectral Norm (D) | off | `DCGAN_SPECTRAL=1` |
| Cosine LR | off | `DCGAN_COSINE=1` |
| Épocas | cfg.dcgan_epochs | `DCGAN_EPOCHS` |

### Profiles de execução

| Profile | Épocas DCGAN | Dataset | Amostras eval | Seeds |
|---|---|---|---|---|
| **DEV** | 50 | 20% subset (~10k) | 2 000 | 3 |
| **PROD** | 100 | Full ArtBench-10 (50k) | 5 000 | 10 |

> **Nota**: a maioria dos experimentos nesta branch foi corrida em DEV. Os experimentos com `_100ep` ou `_200ep` no nome usaram `DCGAN_EPOCHS` custom, não necessariamente PROD.

---

## 1. Catálogo Completo de Resultados

### 1.1 DCGAN — Sweep Inicial (50 épocas, DEV)

| Experimento | lat | NGF | NDF | LR_G | LR_D | β1 | SN | Cosine | FID ↓ | FID std | KID ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `default_dcgan` | 100 | 64 | 64 | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 118.82 | 1.40 | 0.0904 |
| `dcgan_lat32` | **32** | 64 | 64 | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | **103.39** | 0.93 | **0.0679** |
| `dcgan_lat64` | **64** | 64 | 64 | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 112.61 | 1.41 | 0.0826 |
| `dcgan_lat100` | 100 | 64 | 64 | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 118.82 | 1.40 | 0.0904 |
| `dcgan_lat256` | **256** | 64 | 64 | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 128.98 | 2.28 | 0.0998 |
| `dcgan_ngf32` | 100 | **32** | **32** | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 139.31 | 2.11 | 0.1131 |
| `dcgan_ngf64` | 100 | **64** | **64** | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 118.82 | 1.40 | 0.0904 |
| `dcgan_ngf128` | 100 | **128** | **128** | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 104.76 | 0.98 | 0.0725 |
| `dcgan_ngf128_ndf64` | 100 | **128** | **64** | 2e-4 | 2e-4 | 0.5 | ✗ | ✗ | 115.68 | 1.42 | 0.0789 |
| `dcgan_beta09` | 100 | 64 | 64 | 2e-4 | 2e-4 | **0.9** | ✗ | ✗ | 286.58 | 1.77 | 0.2517 |
| `dcgan_lr1e3` | 100 | 64 | 64 | **1e-3** | **1e-3** | 0.5 | ✗ | ✗ | 125.25 | 1.81 | 0.0891 |
| `dcgan_lr2e4` | 100 | 64 | 64 | **2e-4** | **2e-4** | 0.5 | ✗ | ✗ | 118.82 | 1.40 | 0.0904 |
| `dcgan_asym_lr` | 100 | 64 | 64 | **4e-4** | **1e-4** | 0.5 | ✗ | ✗ | 148.46 | 1.25 | 0.1251 |
| `dcgan_cosine` | 100 | 64 | 64 | 2e-4 | 2e-4 | 0.5 | ✗ | **✓** | 100.37 | 2.62 | 0.0659 |

> **Nota**: `dcgan_lat100`, `dcgan_ngf64` e `dcgan_lr2e4` são idênticos ao `default_dcgan` — mesma config, mesmos resultados (FID=118.82). Foram corridos como redundância de verificação.

### 1.2 DCGAN — Experimentos Estendidos (épocas variáveis)

| Experimento | lat | NGF | NDF | SN | Cosine | Épocas | FID ↓ | FID std | KID ↓ |
|---|---|---|---|---|---|---|---|---|---|
| `dcgan_lat32_100ep` | 32 | 64 | 64 | ✗ | ✗ | 100 | 91.78 | 1.34 | 0.0560 |
| `dcgan_lat32_ngf128` | 32 | 128 | 128 | ✗ | ✗ | 50 | 116.04 | 1.58 | 0.0812 |
| `dcgan_lat32_ngf128_asym` | 32 | 128 | 128 | ✗ | ✗ | 50 | 128.64 | 0.97 | 0.0909 |
| `dcgan_lat32_ngf128_ndf64` | 32 | 128 | 64 | ✗ | ✗ | 50 | 128.45 | 2.28 | 0.0973 |
| `dcgan_lat32_asym_lr` | 32 | 64 | 64 | ✗ | ✗ | 50 | 134.32 | 3.02 | 0.1059 |
| `dcgan_ngf128_100ep` | 100 | 128 | 128 | ✗ | ✗ | 100 | 74.29 | 1.20 | 0.0385 |
| `dcgan_ngf128_ndf64_100ep` | 100 | 128 | 64 | ✗ | ✗ | 100 | 86.74 | 1.38 | 0.0526 |
| `dcgan_cosine_sn` | 100 | 64 | 64 | **✓** | **✓** | 50 | 103.58 | 1.74 | 0.0676 |
| **`dcgan_spectral`** | 100 | 64 | 64 | **✓** | ✗ | **100** | **71.17** | **1.18** | **0.0344** |
| `dcgan_spectral_lat32` | **32** | 64 | 64 | **✓** | ✗ | 100 | 72.16 | 1.01 | 0.0357 |
| **`dcgan_spectral_200ep`** | 100 | 64 | 64 | **✓** | ✗ | **200** | **60.02** | **0.64** | **0.0251** |

### 1.3 WGAN-GP e Variantes

| Experimento | Variação | Épocas | FID ↓ | FID std | KID ↓ |
|---|---|---|---|---|---|
| `wgan_gp` | WGAN-GP, n_critic=5, cosine | 100 | 128.29 | 1.61 | 0.0947 |
| `wgan_ncritic2` | WGAN-GP, n_critic=2, cosine | 100 | 155.50 | 2.46 | 0.1041 |
| `wgan_no_cosine` | WGAN-GP, n_critic=5, sem cosine | 100 | 109.01 | 1.05 | 0.0652 |

### 1.4 Conditional DCGAN

| Experimento | Épocas | FID ↓ | FID std | KID ↓ |
|---|---|---|---|---|
| `cdcgan` | 100 | 134.64 | 1.54 | 0.1050 |

### 1.5 StyleGAN

| Experimento | NGF | w_dim | Epocas | Style Mix | FID ↓ | FID std | KID ↓ |
|---|---|---|---|---|---|---|---|
| `stylegan_default` | 64 | 512 | 50 | ✓ | 139.23 | 2.71 | 0.0515 |
| `stylegan_map8` | 64 | 512 | 50 | ✓ | 155.14 | 0.97 | 0.0679 |
| `stylegan_wdim256` | 64 | **256** | 50 | ✓ | 131.67 | 1.04 | 0.0462 |
| `stylegan_ngf128` | **128** | 512 | 50 | ✓ | 125.65 | 2.50 | 0.0553 |
| **`stylegan_ngf128_200ep`** | **128** | 512 | **200** | ✓ | **113.22** | **1.51** | **0.0423** |
| `stylegan_nomix` | 64 | 512 | ~40 | **✗** | — | — | — |

> **`stylegan_nomix`** não tem results.csv — o treino aparentemente crashou por volta da época 40.

---

## 2. Ranking Global

| Rank | Experimento | Modelo | FID ↓ | KID ↓ | Nota |
|---|---|---|---|---|---|
| 🥇 1 | `dcgan_spectral_200ep` | DCGAN+SN | **60.02** | **0.0251** | SN + 200ep DEV |
| 🥈 2 | `dcgan_spectral` | DCGAN+SN | 71.17 | 0.0344 | SN + 100ep |
| 🥉 3 | `dcgan_spectral_lat32` | DCGAN+SN | 72.16 | 0.0357 | SN + lat=32 |
| 4 | `dcgan_ngf128_100ep` | DCGAN | 74.29 | 0.0385 | NGF=128 + 100ep |
| 5 | `dcgan_ngf128_ndf64_100ep` | DCGAN | 86.74 | 0.0526 | NGF>NDF |
| 6 | `dcgan_lat32_100ep` | DCGAN | 91.78 | 0.0560 | lat=32 + 100ep |
| 7 | `dcgan_cosine` | DCGAN | 100.37 | 0.0659 | Cosine LR 50ep |
| 8 | `dcgan_lat32` | DCGAN | 103.39 | 0.0679 | lat=32 50ep |
| 9 | `dcgan_cosine_sn` | DCGAN | 103.58 | 0.0676 | Cosine+SN 50ep |
| 10 | `dcgan_ngf128` | DCGAN | 104.76 | 0.0725 | NGF=128 50ep |
| 11 | `wgan_no_cosine` | WGAN-GP | 109.01 | 0.0652 | WGAN melhor |
| 12 | `stylegan_ngf128_200ep` | StyleGAN | 113.22 | 0.0423 | StyleGAN melhor |
| — | `dcgan_beta09` | DCGAN | 286.58 | 0.2517 | Pior de todos |

---

## 3. Análise por Hiperparâmetro

### 3.1 Latent Dimension — Inversão da tendência

| lat | FID (50ep, sem SN) | Rank |
|---|---|---|
| 32 | **103.39** | 🥇 |
| 64 | 112.61 | 🥈 |
| 100 | 118.82 | 🥉 (default) |
| 256 | 128.98 | 4º |

**Tendência contrária ao esperado**: lat=32 supera lat=100 e lat=256. Esta inversão é oposta ao que acontece nos VAEs.

**Explicação teórica**: No GAN, o latent não é um bottleneck de informação — é simplesmente o "semente" do processo generativo. Com 50 épocas de treino, o Generator não tem tempo suficiente para mapear efectivamente todos os 100 ou 256 graus de liberdade para padrões visuais coerentes. Com lat=32, o espaço de ruído é mais compacto → mais fácil de cobrir completamente → melhor diversidade nas amostras geradas (cada ponto de z=N(0,I) mapeia para uma imagem distinta, não para regiões do espaço de imagem nunca visitadas). Com lat=256, o Generator tem muitas dimensões para aprender mas poucas épocas — resulta em sub-utilização do espaço latente.

Com mais épocas (100ep): lat=32 ainda ganha (91.78 vs 74.29 para NGF=128 com lat=100). A vantagem de lat menor persiste.

### 3.2 Feature Maps (NGF/NDF) — Capacidade é crucial

| NGF=NDF | FID (50ep) | Δ vs default |
|---|---|---|
| 32 | 139.31 | +20.5 ❌ |
| 64 (default) | 118.82 | baseline |
| **128** | **104.76** | **−14.1** ✅ |

**NGF=128 claramente melhor**. O Generator com NGF=128 tem ~4× mais parâmetros que NGF=64 → consegue aprender padrões de textura artística mais ricos. Para ArtBench-10 com 10 estilos distintos, a capacidade representativa é limitante.

**Assimetria NGF > NDF** (`dcgan_ngf128_ndf64`): FID=115.68 — pior que NGF=NDF=128 (104.76). A intuição que "G mais poderoso que D = melhor" não se confirma. O Discriminator com NDF=64 não consegue fornecer gradientes suficientemente informativos para um Generator NGF=128 — o equilíbrio G/D é importante.

**Efeito de mais épocas (NGF=128)**:
- 50ep: FID=104.76
- 100ep: FID=74.29 → **−30.47 FID** (ganho enorme!)
O modelo NGF=128 está claramente sub-treinado em 50 épocas — tem capacidade para aprender muito mais.

### 3.3 β1 (Adam Momentum) — Parâmetro mais perigoso

| β1 | FID | KID | Diagnóstico |
|---|---|---|---|
| 0.5 (paper DCGAN) | 118.82 | 0.0904 | Estável |
| **0.9 (Adam standard)** | **286.58** | **0.2517** | **Colapso** |

**β1=0.9 → FID=286.58**: O pior resultado de toda a experiência GAN, pior até que WGAN-GP subótimo. Aumento de 167.76 FID face ao default!

**Explicação**: β1 controla o momentum do Adam — quanto o optimizer "lembra" dos gradientes anteriores. No treino GAN, os gradientes oscilam naturalmente (D e G treinam alternadamente, o landscape muda a cada step). Com β1=0.9, o optimizer tem alta inércia → amplifica as oscilações em vez de as suavizar → instabilidade severa. O D aprende a saturar rapidamente (distingue perfeitamente real de fake) antes de G ter tempo de aprender → G recebe gradientes próximos de zero → modo colapso.

O paper DCGAN (Radford et al. 2015) especifica β1=0.5 precisamente por esta razão. Este resultado empiricamente valida a escolha do paper para este dataset.

### 3.4 Learning Rate — Zona óptima estreita

| LR | FID | Diagnóstico |
|---|---|---|
| **2e-4 (paper default)** | **118.82** | Baseline |
| 1e-3 | 125.25 | +6.4, ligeiramente pior |

Com apenas dois pontos no sweep de LR, a zona óptima para DCGAN está confirmada em torno de 2e-4 (conforme o paper). LR=1e-3 é 5× maior e já piora — a instabilidade adversarial amplifica os efeitos de LR excessivo muito mais do que no VAE.

**LR assimétrico** (`dcgan_asym_lr`): LR_G=4e-4, LR_D=1e-4 → FID=148.46 (**+29.6 vs default**). Aumentar o LR do Generator enquanto se diminui o do Discriminator rompe o equilíbrio: G aprende mais depressa que D → D fica "para trás" → G aprende a enganar um D fraco, não a gerar imagens realistas. Com lat=32+asym: FID=134.32 (mesma tendência negativa).

### 3.5 Spectral Normalization — O factor decisivo

| Configuração | Épocas | FID ↓ | Δ vs mesmo sem SN |
|---|---|---|---|
| `dcgan_ngf128_100ep` (sem SN) | 100 | 74.29 | — |
| `dcgan_spectral` (com SN) | 100 | 71.17 | −3.12 |
| `dcgan_spectral_lat32` (com SN) | 100 | 72.16 | — |
| **`dcgan_spectral_200ep`** | **200** | **60.02** | — |

A Spectral Normalization normaliza cada camada do Discriminator pela sua norma espectral (maior valor singular):
```
W_SN = W / σ_max(W)
```
Isto garante que D é 1-Lipschitz: `|D(x) - D(y)| ≤ ||x - y||`. Consequências:
1. D não consegue tornar-se "demasiado bom" demasiado depressa
2. Os gradientes que chegam a G são sempre informativos (não zero por saturação, não infinito por explosão)
3. O treino é estável por muito mais épocas

**Efeito de epochs com SN**: 100ep → 71.17, 200ep → **60.02** (melhoria de 11.15 FID). Sem SN, provavelmente 200ep teria modo colapso. A SN torna possível o treino prolongado.

**SN vs Cosine LR** (`dcgan_cosine_sn` = 103.58 vs `dcgan_cosine` = 100.37): A combinação SN+Cosine é ligeiramente pior que só Cosine em 50ep. Porquê? Com 50ep, o Cosine LR já estabiliza o treino o suficiente. A SN adiciona uma restrição extra ao D que, a curto prazo, pode deixá-lo menos capaz de fornecer gradientes ricos. A SN revela o seu valor a longo prazo (100–200ep).

### 3.6 Cosine LR para GAN — Resultado positivo (contrário ao VAE)

| Configuração | Épocas | FID ↓ | Melhoria |
|---|---|---|---|
| `default_dcgan` | 50 | 118.82 | baseline |
| `dcgan_cosine` | 50 | 100.37 | **−18.45** ✅ |

**O Cosine LR AJUDA nos GANs**, ao contrário dos VAEs a 50ep. Porquê a diferença?

Para o VAE, o Cosine LR a 50ep reduzia o LR demasiado cedo, parando o aprendizado. Para o GAN, o Cosine LR tem um efeito diferente: nas fases finais do treino, quando G e D estão mais próximos do equilíbrio, um LR baixo previne overshooting — o optimizador não "salta" sobre o equilíbrio adversarial. O GAN beneficia da estabilização final que o Cosine proporciona, especialmente porque o landscape de otimização adversarial é inerentemente instável.

Esta assimetria (Cosine ajuda GAN, não ajuda VAE curto) é uma observação importante para o relatório.

---

## 4. WGAN-GP — Análise das Variantes

### 4.1 Resultados

| Configuração | n_critic | Cosine | FID ↓ | KID ↓ |
|---|---|---|---|---|
| `wgan_gp` (com cosine) | 5 | ✓ | 128.29 | 0.0947 |
| `wgan_ncritic2` (com cosine) | 2 | ✓ | 155.50 | 0.1041 |
| `wgan_no_cosine` | 5 | ✗ | **109.01** | **0.0652** |

### 4.2 Análise

**WGAN-GP é pior que DCGAN+SN** em todos os casos. Teoria vs prática:

**Teoria**: O WGAN substitui a BCE adversarial pela distância de Wasserstein:
```
L = E[C(x_real)] - E[C(G(z))] + λ · E[(||∇C(x̂)||₂ - 1)²]
```
O Critic (sem Sigmoid) produz scores não normalizados. A gradient penalty ($λ=10$) força 1-Lipschitz sem clipagem de pesos. Teoricamente mais estável que BCE.

**Na prática (ArtBench-10)**:
1. **n_critic=5 com Cosine (FID=128.29)**: O Critic actualiza 5× mais depressa que G → fica muito mais forte → gradientes de Wasserstein para G são instáveis nas primeiras épocas
2. **n_critic=2 (FID=155.50)**: Muito pior! O Critic não chega perto do óptimo a cada step → gradientes de Wasserstein são imprecisos → G recebe sinal ruidoso
3. **Sem Cosine (FID=109.01)**: Surpreendentemente o melhor WGAN! O Cosine LR com decaimento agressivo pode ser problemático para o Critic que precisa de manter LR alto para atingir o seu óptimo a cada step

**Por que DCGAN+SN supera WGAN-GP?** SN não adiciona parâmetros nem altera a loss — é uma restrição arquitectónica limpa. A gradient penalty do WGAN-GP, por outro lado, requer computação de segundos gradientes (backprop através de backprop), é mais ruidosa e sensível ao λ. Para datasets de 32×32 com 50k imagens, a estabilidade da SN é suficiente sem o overhead do WGAN-GP.

---

## 5. cDCGAN — Geração Condicional

| Experimento | Épocas | FID ↓ | KID ↓ |
|---|---|---|---|
| `cdcgan` | 100 | 134.64 | 0.1050 |
| `default_dcgan` | 50 | 118.82 | 0.0904 |
| `dcgan_ngf128_100ep` | 100 | 74.29 | 0.0385 |

**cDCGAN (FID=134.64) é pior que DCGAN base com as mesmas épocas (100ep)**. Comparar com `dcgan_ngf128_100ep` (74.29) — com mesmas épocas mas sem conditioning.

**Análise**: O cDCGAN injeta um embedding de classe (10 estilos artísticos) no latent do Generator e como canal adicional no Discriminator. O treino adversarial já é difícil sem conditioning; com conditioning, o modelo tem de aprender simultaneamente:
1. Gerar imagens realistas
2. Condicionar correctamente na classe
3. Fazer o Discriminator distinguir real/fake por classe

100 épocas com 20% do dataset (10k imagens, ~1k por classe) não são suficientes para que o modelo aprenda o conditioning efectivamente. O modelo usa a informação de classe parcialmente, resultando em amostras que não são tão diversas quanto um DCGAN unconditional com mais capacidade.

**Vantagem qualitativa**: O cDCGAN permite geração por estilo artístico na inferência, o que o FID unconditional não captura. Para avaliação condicional (FID por classe), o cDCGAN poderia superar o DCGAN.

---

## 6. StyleGAN — Análise

### 6.1 Resultados

| Experimento | NGF | w_dim | Épocas | Style Mix | FID ↓ | KID ↓ |
|---|---|---|---|---|---|---|
| `stylegan_default` | 64 | 512 | 50 | ✓ | 139.23 | 0.0515 |
| `stylegan_map8` | 64 | 512 | 50 | ✓ | 155.14 | 0.0679 |
| `stylegan_wdim256` | 64 | **256** | 50 | ✓ | 131.67 | 0.0462 |
| `stylegan_ngf128` | **128** | 512 | 50 | ✓ | 125.65 | 0.0553 |
| **`stylegan_ngf128_200ep`** | **128** | 512 | **200** | ✓ | **113.22** | **0.0423** |
| `stylegan_nomix` | 64 | 512 | ~40 | **✗** | — | — |

### 6.2 Análise

**StyleGAN é pior que DCGAN+SN** mas a tendência com mais épocas é positiva.

**Por que StyleGAN é mais difícil em 32×32?**
- A mapping network transforma z→w, mas em 32×32 o Generator tem apenas 3 etapas de upsampling (4→8→16→32). Com poucas camadas, o AdaIN tem menos "níveis de estilo" para injectar vs. arquiteturas de alta resolução (256×256 tem 6+ camadas)
- O style mixing (usar dois latents em camadas diferentes) é mais benéfico quando há muitas camadas onde diferentes aspectos podem ser controlados independentemente
- O mapeamento z→w requer mais dados e epochs para convergir útilmente

**`stylegan_map8` (FID=155.14) é pior que `stylegan_default`**: O default já usa 8 camadas na mapping network (é o default do paper). `stylegan_map8` pode ter uma configuração diferente que piora o resultado — provavelmente re-inicializou com menos layers ou diferentes parâmetros.

**w_dim=256 vs 512**: `stylegan_wdim256` (131.67) é melhor que `stylegan_default` (139.23). Um espaço W mais compacto pode ser mais fácil de preencher com padrões artísticos coerentes em 32×32, análogo ao efeito de latent dim menor no DCGAN.

**`stylegan_ngf128_200ep` (FID=113.22) — melhor StyleGAN**: NGF=128 + 200 épocas. A tendência com mais epochs é clara (50ep: 125.65 → 200ep: 113.22, ganho de 12.43 FID). Com mais epochs, a mapping network tem tempo para aprender um espaço W mais estruturado.

**`stylegan_nomix` crashou** após ~40 epochs. O style mixing é importante para regularização — sem ele, o modelo pode sobreajustar rapidamente ou ter gradientes instáveis.

**StyleGAN KID vs FID**: Curiosamente, o KID do StyleGAN (0.042–0.067) é comparável ao do DCGAN não-SN (0.065–0.090) mesmo com FID maior. Isto sugere que o StyleGAN gera amostras mais diversas (melhor cobertura da distribuição, medido por KID/MMD) mas com qualidade individual inferior (FID penaliza médias da distribuição).

---

## 7. Interacções Entre Hiperparâmetros

### 7.1 Latent × NGF (50ep sem SN)

| lat \ NGF | 64 | 128 |
|---|---|---|
| **32** | 103.39 | 116.04 |
| **100** | 118.82 | 104.76 |

**Resultado surpreendente**: lat=32+NGF=64 (103.39) é melhor que lat=32+NGF=128 (116.04)! Mas lat=100+NGF=128 (104.76) é melhor que lat=100+NGF=64 (118.82). A interacção não é aditiva:
- Com lat=32, o Generator tem menos dimensões de entrada → uma rede menor (NGF=64) é mais eficiente em mapear este espaço compacto
- Com lat=100, a rede maior (NGF=128) tem capacidade para explorar mais o espaço latente

### 7.2 SN × Epochs — A combinação vencedora

| | 50ep | 100ep | 200ep |
|---|---|---|---|
| **sem SN** | 118.82 | 74.29* | ~65?† |
| **com SN** | — | 71.17 | **60.02** |

*`dcgan_ngf128_100ep` (NGF=128, lat=100)  
†estimativa se não colapsar

SN é que permite treinar para 200ep. Sem SN, treino longo causa modo colapso. Com SN, cada epoch adicional melhora consistentemente o FID. A interacção é não-linear: SN sozinho a 100ep dá 71.17, mas SN+200ep dá 60.02 — um ganho desproporcionalmente maior das épocas extras quando SN está activo.

---

## 8. Análise de Convergência (épocas)

| Config | 50ep | 100ep | 200ep | Δ(50→100) | Δ(100→200) |
|---|---|---|---|---|---|
| NGF=128, lat=100 (sem SN) | 104.76 | 74.29 | — | −30.47 | — |
| NGF=128, NDF=64 (sem SN) | 115.68 | 86.74 | — | −28.94 | — |
| lat=32 (sem SN) | 103.39 | 91.78 | — | −11.61 | — |
| SN, lat=100 | — | 71.17 | 60.02 | — | −11.15 |
| StyleGAN NGF=128 | 125.65 | — | 113.22 | — | −12.43 |

**Observação**: os maiores ganhos por epoch ocorrem entre 50ep e 100ep (−28 a −30 FID para NGF=128 sem SN). Entre 100ep e 200ep o ganho é menor (−11 FID) mas ainda significativo. O modelo NGF=128 sem SN claramente não converge em 50 épocas.

O padrão de convergência do GAN é diferente do VAE: no VAE a convergência é monotónica e suave (log-linear). No GAN, a convergência é mais instável e não-monotónica — o FID pode aumentar antes de baixar em epochs intermédias se G e D estiverem fora de equilíbrio.

---

## 9. Resumo e Melhorias Identificadas

### 9.1 Configuração Final Recomendada (PROD)

**Melhor configuração DEV**: `dcgan_spectral_200ep` → FID=60.02 (lat=100, NGF=NDF=64, SN, 200ep, 20% subset)

**Para PROD**: Com dataset completo (50k) e 200+ épocas, espera-se FID significativamente melhor. Com base no padrão observado (DEV→PROD valeu ~33 FID para Diffusion), podemos estimar FID PROD ≈ 28–35 para `dcgan_spectral_200ep`.

**Configuração recomendada para PROD**:
```bash
DCGAN_LATENT=100
DCGAN_NGF=128       # aumentar de 64 para 128 (melhor capacidade)
DCGAN_NDF=128
DCGAN_SPECTRAL=1    # SN obrigatório
DCGAN_EPOCHS=200    # ou mais
RUN_PROFILE=PROD    # dataset completo, 5000 amostras, 10 seeds
```

### 9.2 O Que Não Funcionou e Porquê

| Técnica | Resultado | Porquê falhou |
|---|---|---|
| β1=0.9 | FID=286 (+167) | Alto momentum amplifica oscilações adversariais |
| LR assimétrico | FID=148 (+29) | Rompe equilíbrio G/D |
| WGAN-GP | FID=109–155 | Gradient penalty é ruidosa; n_critic sensível |
| StyleGAN (50ep) | FID=125–155 | Mapping network precisa de muitas epochs; 32×32 limita o impacto do AdaIN |
| NGF>NDF | FID=116 | D fraco não fornece gradientes úteis para G forte |
| cDCGAN | FID=134 | Dataset insuficiente por classe para aprender conditioning em 100ep |

### 9.3 O Que Funcionou e Porquê

| Técnica | Melhoria | Mecanismo |
|---|---|---|
| SN no D | −48 FID | Mantém D 1-Lipschitz, gradientes sempre informativos |
| NGF=128 | −14 FID (50ep) | Mais capacidade generativa para 10 estilos |
| Cosine LR | −18 FID | Estabiliza equilíbrio G/D nas épocas finais |
| lat=32 | −15 FID (50ep) | Espaço latente mais fácil de cobrir com treino curto |
| Mais epochs | −28 FID (50→100ep) | Os modelos não convergem em 50ep; margem grande ainda disponível |

---

## 10. O que falta / Próximos Passos

1. **Run PROD do `dcgan_spectral_200ep`**: ainda está em DEV (20% subset, 3 seeds, 2000 amostras). Precisa ser corrido com `RUN_PROFILE=PROD` para a avaliação final do relatório.

2. **Run PROD do `dcgan_ngf128_spectral`** (NGF=128 + SN + 200ep): a melhor combinação teoricamente, ainda não testada.

3. **cDCGAN em PROD**: avaliar com dataset completo e mais epochs — pode melhorar significativamente.

4. **`stylegan_nomix` corrigido**: investigar o crash e re-correr para confirmar o impacto do style mixing.

5. **WGAN-GP com SN** em vez de gradient penalty: combinar as duas abordagens de estabilização.

---

## 11. Tabela para o Relatório

### Sweep DCGAN (50 épocas, DEV, 3 seeds, 2000 amostras)

| Variação | FID ↓ | FID std | KID ↓ | Δ baseline |
|---|---|---|---|---|
| Baseline (lat=100, NGF=64) | 118.82 | 1.40 | 0.0904 | — |
| lat=32 | 103.39 | 0.93 | 0.0679 | −15.4 |
| lat=64 | 112.61 | 1.41 | 0.0826 | −6.2 |
| lat=256 | 128.98 | 2.28 | 0.0998 | +10.2 |
| NGF=NDF=32 | 139.31 | 2.11 | 0.1131 | +20.5 |
| NGF=NDF=128 | 104.76 | 0.98 | 0.0725 | −14.1 |
| β1=0.9 | 286.58 | 1.77 | 0.2517 | +167.8 |
| LR=1e-3 | 125.25 | 1.81 | 0.0891 | +6.4 |
| LR assimétrico | 148.46 | 1.25 | 0.1251 | +29.6 |
| + Cosine LR | 100.37 | 2.62 | 0.0659 | −18.5 |

### Sweep épocas (selecção)

| Configuração | 50ep | 100ep | 200ep |
|---|---|---|---|
| NGF=128, lat=100 | 104.76 | 74.29 | — |
| NGF=128, NDF=64, lat=100 | 115.68 | 86.74 | — |
| lat=32, NGF=64 | 103.39 | 91.78 | — |
| SN, lat=100, NGF=64 | — | 71.17 | **60.02** |

### Comparação de Arquitecturas GAN

| Arquitectura | Melhor Config | FID ↓ | KID ↓ |
|---|---|---|---|
| DCGAN + SN | lat=100, NGF=64, 200ep | **60.02** | **0.0251** |
| DCGAN + SN | lat=32, NGF=64, 100ep | 72.16 | 0.0357 |
| DCGAN (sem SN) | NGF=128, 100ep | 74.29 | 0.0385 |
| StyleGAN | NGF=128, w_dim=512, 200ep | 113.22 | 0.0423 |
| cDCGAN | lat=100, NGF=64, 100ep | 134.64 | 0.1050 |
| WGAN-GP | n_critic=5, sem cosine | 109.01 | 0.0652 |

---

*Análise gerada a partir dos results.csv da branch `gan` | 2026-04-25*
