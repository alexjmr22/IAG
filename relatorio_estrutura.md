# Relatório de Investigação: ArtBench Generative Modeling

**Unidade Curricular:** Inteligência Artificial Generativa (2025/2026)  
**Instituição:** Faculdade de Ciências e Tecnologia, Universidade de Coimbra  

---

## 1. Abstract

<!-- SUGESTÃO: Escrever ~200 palavras cobrindo:
- Objetivo: estudo comparativo de 3 famílias de modelos generativos no ArtBench-10
- Metodologia: grid search sistemático (~45 experiências) com protocolo rigoroso de FID/KID
- Resultados chave: melhor VAE (FID 145.85, β=0.05), mencionar melhores GAN e Diffusion
- Conclusão em 1 frase: qual família venceu e porquê
-->

*A preencher após todos os resultados estarem completos.*

---

## 2. Introduction

<!-- SUGESTÃO: 
- Contextualizar ArtBench-10: 60k imagens, 10 estilos artísticos, 32×32 RGB
- Problema: gerar arte com diversidade estilística e fidelidade visual
- Mencionar as 3 famílias de modelos e a metodologia de grid search
- Contributo: abordagem metódica de sweep univariado → combinações → validação → produção
- ~1 página
-->

* **Contexto:** Descrever o dataset ArtBench-10 e a importância da síntese de arte na IA Generativa.
* **Problema:** A dificuldade de equilibrar fidelidade visual e diversidade em 10 estilos artísticos distintos.
* **Objetivo:** Estudo comparativo entre as famílias VAE, GAN e Diffusion.
* **Contributo:** Uma abordagem sistemática de exploração de hiperparâmetros para atingir o estado da arte no dataset.

---

## 3. Methodology

Nesta secção, descrevemos as especificidades técnicas das implementações realizadas.

### 3.1. Variational Autoencoders (VAE)

#### 3.1.1. Arquitetura — ConvVAE

A arquitetura base é uma **Convolutional VAE (ConvVAE)** adaptada do notebook 3 da disciplina, originalmente desenhada para MNIST (1 canal, 28×28) e modificada para imagens RGB 32×32 do ArtBench-10. O modelo segue a estrutura clássica encoder-decoder com reparameterização (Kingma & Welling, 2014):

| Componente | Camadas | Detalhes |
|:---|:---|:---|
| **Encoder** | 3× Conv2d (stride=2) + BatchNorm + ReLU | Reduz 32×32 → 16 → 8 → 4 (bottleneck 4×4×128) |
| **Projeção Latente** | 2× Linear (2048 → $d_z$) | Produz $\mu$ e $\log \sigma^2$ |
| **Reparameterização** | $z = \mu + \varepsilon \cdot \sigma$, $\varepsilon \sim \mathcal{N}(0, I)$ | Permite backpropagation através da amostragem |
| **Decoder** | Linear + 3× ConvTranspose2d + Tanh | Reconstrói 4→8→16→32 com saída em $[-1, 1]$ |

O encoder aplica convoluções com stride=2 para reduzir progressivamente a resolução espacial, enquanto o decoder utiliza convoluções transpostas para reconstruir a imagem. A ativação final Tanh garante que a saída está normalizada no intervalo $[-1, 1]$, consistente com a normalização aplicada ao input (média 0.5, desvio-padrão 0.5 por canal). O total de parâmetros treináveis é ~1.05M (para $d_z = 128$).

#### 3.1.2. Função de Perda — β-VAE (ELBO)

O treino do VAE maximiza a Evidence Lower Bound (ELBO, Kingma & Welling 2014), formulada como:

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \| x_i - \hat{x}_i \|^2}_{\text{Reconstrução (MSE)}} + \beta \cdot \underbrace{\left( -\frac{1}{2N}\sum_{i=1}^{N}\sum_{j=1}^{d_z} \left(1 + \log\sigma_{ij}^2 - \mu_{ij}^2 - \sigma_{ij}^2\right) \right)}_{\text{KL Divergence}}$$

Onde:
- **Reconstrução (MSE)**: Soma dos quadrados das diferenças, normalizada pelo batch size $N$. Mede a fidelidade pixel-a-pixel da reconstrução.
- **KL Divergence**: Mede a divergência entre a posterior aproximada $q(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ e a prior $p(z) = \mathcal{N}(0, I)$. Normalizada pelo batch size $N$ (sem normalização pela dimensão latente $d_z$).
- **$\beta$** (Higgins et al., 2017): Peso de regularização que controla o trade-off entre qualidade de reconstrução e organização do espaço latente.

<!-- NOTA IMPORTANTE PARA O RELATÓRIO:
A normalização do KL apenas por batch_size (e não por batch_size × latent_dim) é relevante:
- É a implementação do notebook original da disciplina
- Significa que o β efetivo escala com latent_dim (e.g., β=0.1 na nossa formulação ≈ β_per_dim = 0.1×128 = 12.8)
- Isto é uma decisão de design válida, não um bug — mas influencia a escala do β ótimo
- Durante a investigação, foi testada a normalização "canónica" (÷ batch×latent), mas verificou-se empiricamente que não melhorava os resultados (ver Secção 5.1.4)
-->

#### 3.1.3. Variantes Arquiteturais

Para além do β-VAE standard, foram implementadas e testadas duas variantes:

**Conditional VAE (CVAE):** Extensão do VAE que condiciona tanto o encoder como o decoder ao label de classe (estilo artístico) através de class embeddings (16 dimensões). O encoder concatena o embedding à representação convolucional antes da projeção latente, e o decoder concatena-o ao vetor latente $z$ antes da reconstrução. Esta arquitetura permite geração condicional por estilo artístico — uma capacidade única face ao VAE standard, particularmente relevante dado que o ArtBench-10 possui 10 classes de estilo perfeitamente balanceadas.

**VQ-VAE (Vector Quantized VAE):** Substitui a distribuição gaussiana contínua do espaço latente por um codebook discreto de 256 embeddings aprendidos (van den Oord et al., 2017). O encoder produz representações contínuas que são quantizadas para o embedding mais próximo no codebook via distância euclidiana, usando o *straight-through estimator* para permitir backpropagation. A loss inclui um termo de *commitment loss* ($\lambda = 0.25$) para encorajar o encoder a produzir representações próximas dos embeddings do codebook. Ao contrário do β-VAE, o VQ-VAE não utiliza regularização KL — a organização do espaço latente é imposta pela discretização.

<!-- SUGESTÃO: Incluir diagrama/figura comparativa das 3 arquiteturas VAE -->

### 3.2. Generative Adversarial Networks (GAN)

<!-- SUGESTÃO: 
- Descrever DCGAN: arquitetura do Generator (ConvTranspose2d) e Discriminator
- Mencionar hiperparâmetros específicos: latent_dim, ngf, ndf, betas do Adam
- NÃO mencionamos WGAN-GP nem Spectral Normalization — remover se não implementámos
- Técnicas de estabilização que realmente usámos (label smoothing, etc.)
-->

*A preencher com detalhes da implementação GAN.*

### 3.3. Diffusion Models

<!-- SUGESTÃO:
- Descrever PixelUNet: U-Net com sinusoidal position embeddings e ResNet blocks
- DDPM com schedule linear (β_start=1e-4, β_end=0.02)
- NÃO implementámos Latent Diffusion nem DDIM — ajustar ao que realmente fizemos
- Mencionar T (timesteps) como hiperparâmetro variável testado
-->

*A preencher com detalhes da implementação Diffusion.*

---

## 4. Experimental Setup

### 4.1. Dataset e Profiles de Execução

O dataset ArtBench-10 (Liao et al., 2022) contém ~60.000 imagens de arte classificadas em 10 estilos artísticos (Art Nouveau, Baroque, Expressionism, Impressionism, Post-Impressionism, Realism, Renaissance, Romanticism, Surrealism, Ukiyo-e). Todas as imagens foram redimensionadas para 32×32 pixels RGB.

Para otimizar o tempo de iteração experimental, definimos dois profiles de execução:

| Profile | Epochs (VAE) | Dataset | Amostras Avaliação | Seeds | Propósito |
|:---|:---|:---|:---|:---|:---|
| **DEV** | 30 (default, variável) | 20% subset (~10k imgs) | 2.000 | 3 | Grid search e iteração rápida |
| **PROD** | 200 | 100% dataset (~50k imgs) | 5.000 | 10 | Resultados finais para relatório |

O subset de 20% foi definido por um CSV fixo fornecido pelo enunciado, garantindo reprodutibilidade entre experiências. Os resultados DEV foram usados para seleção de hiperparâmetros, e apenas as configurações vencedoras foram treinadas em PROD.

### 4.2. Protocolo de Avaliação

Conforme especificado no enunciado, todos os modelos seguem o mesmo protocolo:

1. **Geração**: Criar $N$ amostras (2.000 em DEV, 5.000 em PROD) usando $z \sim \mathcal{N}(0, I)$
2. **Referência**: Amostrar $N$ imagens reais aleatoriamente do dataset de treino
3. **FID** (Fréchet Inception Distance, Heusel et al. 2017): Distância entre distribuições no espaço de features InceptionV3 (2048-d). Calculado sobre o conjunto completo de $N$ amostras.
4. **KID** (Kernel Inception Distance, Bińkowski et al. 2018): Distância baseada em MMD kernel, calculada em 50 subsets de 100 imagens → reporta-se média ± desvio-padrão.
5. **Repetição**: Todo o processo é repetido com $S$ seeds aleatórias ($S=3$ em DEV, $S=10$ em PROD). Reporta-se FID: média ± std e KID: média ± std.

O processamento (resize, crop, normalização) é **idêntico** para todas as famílias de modelos, garantindo comparabilidade.

### 4.3. Estratégia de Grid Search

A otimização de hiperparâmetros seguiu uma estratégia metódica em fases:

1. **Sweep Univariado** (Ronda 1): Variar cada hiperparâmetro individualmente, mantendo os restantes no valor default do notebook.
2. **Refinamento Dirigido** (Ronda 2): Explorar zonas promissoras identificadas nos sweeps + testar combinações dos melhores individuais.
3. **Validação** (Ronda 3): Confirmar resultados com mais epochs e configuração estável.
4. **Exploração de Técnicas Avançadas** (Ronda 4): Testar schedulers (Cosine Annealing LR) e arquiteturas alternativas.
5. **Produção** (Ronda Final): Treinar as configurações vencedoras no dataset completo com protocolo de avaliação maximal.

No total, foram conduzidas **~45 experiências** para o VAE, distribuídas por 12 configurações de PC (máquinas). Cada experiência gera automaticamente: curvas de treino, amostras intermediárias, reconstruções, interpolações latentes, checkpoint do modelo, e métricas FID/KID via um sistema de orquestração automatizado (`run_experiments.py`).

<!-- SUGESTÃO: Incluir tabela semelhante para hiperparâmetros testados em GAN e Diffusion -->

### 4.4. Tabela de Hiperparâmetros Explorados (VAE)

| Hiperparâmetro | Valores Testados | Default (notebook) | Melhor Encontrado |
|:---|:---|:---|:---|
| $\beta$ (peso KL) | 0.0, 0.02, 0.05, **0.1**, 0.15, 0.2, 0.5, 0.7, 1.0, 2.0 | 0.7 | **0.05** |
| Dimensão latente ($d_z$) | 16, 32, 64, 96, 128, 256 | 128 | **128** |
| Learning Rate | 1e-4, 5e-4, 1e-3, **2e-3**, 5e-3, 1e-2 | 1e-3 | **2e-3** |
| Epochs | 30, 50, 100, **150**, 200 | 30 | **150–200** |
| Batch Size | 128 (fixo) | 128 | 128 |
| Optimizer | Adam (fixo) | Adam | Adam |
| Cosine Annealing LR | true / false | false | **false** |

---

## 5. Results and Analysis

Esta secção apresenta a narrativa das nossas descobertas experimentais.

### 5.1. Otimização do VAE (Ablation Studies)

O VAE foi o modelo sujeito à exploração mais exaustiva, com ~45 experiências ao longo de 5 rondas de otimização. Apresentamos os resultados organizados por hiperparâmetro.

#### 5.1.1. Impacto do $\beta$: Reconstrução vs. Organização Latente

O peso $\beta$ da divergência KL revelou-se o **hiperparâmetro mais impactante**, com um swing de mais de 150 pontos de FID entre o pior e o melhor valor testado. A tabela seguinte resume os resultados do sweep de $\beta$ (30 epochs, 20% subset, $d_z = 128$, $lr = 10^{-3}$):

| $\beta$ | FID ↓ | KID ↓ | Regime |
|:---|:---|:---|:---|
| 0.0 | 336.76 ± 1.53 | 0.360 ± 0.013 | Autoencoder puro — sem regularização |
| 0.05 | 205.11 ± 0.75 | 0.200 ± 0.009 | Regularização leve |
| **0.1** | **185.26 ± 0.59** | **0.172 ± 0.008** | **Sweet spot (30ep)** |
| 0.2 | 187.86 ± 1.49 | 0.167 ± 0.008 | Plateau ótimo |
| 0.5 | 223.23 ± 1.93 | 0.206 ± 0.009 | Sobre-regularizado |
| 0.7 (default) | 241.58 ± 1.98 | 0.231 ± 0.010 | Default do notebook — subótimo |
| 1.0 | 262.29 ± 3.02 | 0.261 ± 0.012 | Posterior collapse parcial |
| 2.0 | 316.29 ± 1.46 | 0.331 ± 0.014 | Posterior collapse severo |

<!-- SUGESTÃO: Incluir gráfico FID vs β (curva em U invertido) — dados disponíveis no sweep acima + dados de 150ep abaixo -->

**Análise:** O trade-off reconstrução–regularização da ELBO manifesta-se claramente:

- **$\beta = 0$ (AE puro):** Sem penalidade KL, o espaço latente fica completamente desorganizado. As regiões de $z$ nunca visitadas pelo encoder geram ruído no decoder, resultando em FID 336.76. Embora a reconstrução seja excelente, a amostragem $z \sim \mathcal{N}(0, I)$ produz amostras fora da distribuição dos encodings.

- **$\beta \in [0.1, 0.2]$ (sweet spot):** A FID é mínima nesta zona, com um plateau largo — β=0.1 (FID 185.26) e β=0.2 (FID 187.86) são quase idênticos. A regularização é suficiente para organizar o espaço latente sem degradar excessivamente a reconstrução.

- **$\beta \geq 0.5$ (sobre-regularização):** O termo KL domina a loss, forçando $q(z|x) \approx \mathcal{N}(0, I)$ para todo $x$. O decoder recebe inputs pouco informativos e produz imagens blurry que se assemelham à média do dataset. Com $\beta = 2.0$, observa-se **posterior collapse** severo — fenómeno descrito por Bowman et al. (2016).

**Nota sobre o default do notebook:** O valor $\beta = 0.7$ foi herdado de uma implementação para MNIST (1 canal, estruturas simples). O ArtBench-10, com 3 canais RGB e texturas artísticas complexas, requer regularização substancialmente mais leve ($\beta = 0.05$–$0.1$). Esta observação sublinha a importância de nunca transferir hiperparâmetros entre datasets sem validação empírica.

#### 5.1.2. Dimensão do Espaço Latente

| $d_z$ | FID ↓ | KID ↓ | Observação |
|:---|:---|:---|:---|
| 16 | 356.26 ± 3.35 | 0.373 ± 0.018 | Capacidade insuficiente |
| 32 | 311.07 ± 1.55 | 0.326 ± 0.014 | Bottleneck informacional |
| **64** | **234.08 ± 1.48** | **0.223 ± 0.010** | Melhor com $\beta = 0.7$ |
| 128 (default) | 241.06 ± 2.07 | 0.230 ± 0.010 | Baseline |
| 256 | 240.92 ± 2.01 | 0.233 ± 0.010 | Saturação |

(Todos os testes com $\beta = 0.7$, $lr = 10^{-3}$, 30 epochs)

**Resultado contra-intuitivo:** $d_z = 64$ supera $d_z = 128$, apesar da menor capacidade. A explicação reside na interação com a normalização do KL: como a soma KL cresce linearmente com $d_z$ (sem normalização per-dimensão), $d_z = 128$ sofre uma penalidade KL 2× mais forte que $d_z = 64$, para o mesmo $\beta$. Com $\beta = 0.7$ (já excessivo), esta pressão extra degrada a reconstrução.

Contudo, após otimização do $\beta$ para 0.05–0.1, a dimensão $d_z = 128$ torna-se preferível por oferecer maior capacidade representacional sem sobre-regularização. **O ótimo de $d_z$ depende do $\beta$** — a otimização conjunta revelou $\{d_z = 128, \beta = 0.05\}$ como a melhor combinação.

#### 5.1.3. Learning Rate e Efeito de Epochs

**Learning Rate** (30 epochs, $\beta = 0.7$, $d_z = 128$):

| LR | FID ↓ | Observação |
|:---|:---|:---|
| $10^{-4}$ | 443.73 ± 1.96 | Convergência insuficiente em 30 epochs |
| $5 \times 10^{-4}$ | 296.55 ± 2.57 | Lento |
| $10^{-3}$ (default) | 241.58 ± 1.98 | Baseline |
| $2 \times 10^{-3}$ | 207.36 ± 2.01 | Bom compromisso estabilidade/velocidade |
| $5 \times 10^{-3}$ | 210.70 ± 2.00 | Rápido mas menos estável |
| $10^{-2}$ | 456.55 ± 1.29 | Instável — não converge |

A zona ótima situa-se em LR $\in [10^{-3}, 5 \times 10^{-3}]$, com $2 \times 10^{-3}$ selecionado como melhor compromisso entre velocidade de convergência e estabilidade a longo prazo.

**Efeito de epochs** (com configuração otimizada $\beta = 0.1$, $d_z = 128$, LR $= 2 \times 10^{-3}$):

| Epochs | FID ↓ | $\Delta$/epoch | Observação |
|:---|:---|:---|:---|
| 30 | 169.62 ± 1.84 | — | Início |
| 50 | 157.73 ± 1.42 | −0.59/ep | Convergência rápida |
| 100 | 146.13 ± 2.02 | −0.23/ep | Desaceleração |
| 150 | 141.28 ± 2.85 | −0.10/ep | Aproximando-se do limite |

A convergência segue um perfil log-linear: cada duplicação de epochs produz ganho decrescente, sugerindo que o limite arquitetural desta ConvVAE se situa em FID $\approx 135$–$140$.

<!-- SUGESTÃO: Incluir gráfico FID vs Epochs mostrando a curva de saturação -->

#### 5.1.4. Cosine Annealing LR: Resultado Negativo

O Cosine Annealing LR (Loshchilov & Hutter, 2017) foi testado com $T_{\max} = 50$ epochs, decaindo o LR de $2 \times 10^{-3}$ para $10^{-5}$.

| Configuração | Epochs | FID ↓ |
|:---|:---|:---|
| Sem scheduler (baseline) | 100 | 146.13 |
| Cosine Annealing LR | 50 | 164.26 |
| Sem scheduler (controlo 50ep) | 50 | ~158* |

*Estimado dos dados da Ronda 3.

**Conclusão:** O Cosine Annealing **não melhorou** os resultados. A explicação é que, com apenas 50 epochs, o schedule reduz o LR demasiado cedo (para $< 10^{-4}$ nas últimas ~15 epochs), impedindo o modelo de continuar a aprender. Na literatura, esta técnica é tipicamente avaliada com $T_{\max} \geq 200$ epochs (contexto ImageNet). Para VAEs com treinos curtos ($< 100$ epochs), um learning rate fixo é preferível.

#### 5.1.5. Investigação da Normalização KL: Uma Lição Metodológica

Durante a Ronda 5, foi investigada uma normalização alternativa da divergência KL: dividir por $N \times d_z$ (batch size × dimensão latente) em vez de apenas $N$, seguindo a formulação de Kingma & Welling (2014, Eq. 10). Os resultados foram inesperados:

| Normalização KL | $\beta$ | FID ↓ | Diferença |
|:---|:---|:---|:---|
| $\div N$ (original) | 0.1 | 146.13 | — |
| $\div (N \times d_z)$ | 0.1 | 285.53 | +139.4 ❌ |

A normalização "corrigida" reduziu o $\beta$ efetivo por um fator de ~128×, transformando o modelo num autoencoder sem regularização. Este resultado, embora inicialmente desconcertante, demonstra que **a escala absoluta do $\beta$ é arbitrária** — o que importa é o equilíbrio relativo entre o termo de reconstrução e a penalidade KL. Ambas as normalizações são matematicamente válidas; simplesmente requerem calibração diferente do $\beta$.

Esta investigação levou também à descoberta e correção de dois bugs no código: um off-by-one no warmup do KL Annealing (Bowman et al., 2016) e um device mismatch na Perceptual Loss — reforçando a importância do debugging meticuloso em sistemas complexos.

#### 5.1.6. Resultado Final do β Tuning (150 Epochs)

Com todos os bugs corrigidos e a normalização revertida para a original, foi conduzido o sweep definitivo de $\beta$ com 150 epochs:

| $\beta$ | FID ↓ | KID ↓ | Ranking |
|:---|:---|:---|:---|
| 0.02 | 148.19 ± 1.14 | 0.143 ± 0.008 | Sub-regularizado |
| **0.05** | **140.22 ± 1.38** | **0.130 ± 0.007** | 🥇 **Melhor VAE (DEV)** |
| 0.1 | 141.28 ± 2.85 | 0.129 ± 0.008 | 🥈 Quase empatado |

A diferença entre $\beta = 0.05$ e $\beta = 0.1$ é de apenas 1.06 FID — dentro da margem de erro entre seeds. Com $\beta = 0.02$, a regularização insuficiente já degrada o FID (+7.97 vs o melhor), confirmando o sweet spot em $\beta \in [0.05, 0.1]$.

#### 5.1.7. Comparação de Arquiteturas VAE

As três arquiteturas VAE foram comparadas nas suas versões otimizadas:

| Arquitetura | $\beta$ | LR | Epochs | FID ↓ | KID ↓ |
|:---|:---|:---|:---|:---|:---|
| **β-VAE** | 0.05 | 2e-3 | 150 | **140.22 ± 1.38** | **0.130 ± 0.007** |
| **CVAE** (Conditional) | 0.15 | 2e-3 | 100 | 151.90 ± 0.75 | 0.145 ± 0.008 |
| **VQ-VAE** (Quantized) | — | 5e-3 | 100 | 191.18 ± 2.04 | 0.177 ± 0.008 |

**β-VAE** obtém o melhor FID, beneficiando de mais epochs e da simplicidade do treino. **CVAE** é 11.68 FID pior, mas oferece uma capacidade que o FID não captura: **geração condicional por estilo artístico**. Cada classe do ArtBench-10 pode ser pré-selecionada ao gerar, o que tem valor prático significativo. O $\beta$ ligeiramente mais alto (0.15) é necessário porque o conditioning estabiliza o modelo, permitindo KL mais forte sem posterior collapse.

**VQ-VAE** apresenta o FID mais alto, possivelmente porque: (i) a amostragem $z \sim \mathcal{N}(0, I)$ não é ideal para codebooks discretos — deveria usar um prior autoregressivo (PixelCNN); (ii) 100 epochs pode ser insuficiente para convergência do codebook; (iii) o LR mais alto (5e-3) foi necessário para atualizações rápidas do codebook mas pode não ser ótimo para o encoder/decoder.

<!-- SUGESTÃO: Incluir grelha visual com amostras geradas por cada variante (β-VAE, CVAE, VQ-VAE) -->

### 5.2. Otimização da GAN (Ablation Studies)

<!-- SUGESTÃO:
- Sweep de latent_dim (32, 100, 256)
- Sweep de ngf/ndf (32, 64, 128) 
- Sweep de LR (1e-3, 2e-4)
- Sweep de beta1 Adam (0.5 vs 0.9)
- Discussão de mode collapse e estabilização
- ~1 página
-->

*A preencher com resultados GAN.*

### 5.3. Otimização do Diffusion (Ablation Studies)

<!-- SUGESTÃO:
- Sweep de T (timesteps: 100, 250, 500, 1000, 1500, 2000)
- Sweep de channels (32, 64, 96, 128)
- Sweep de LR (1e-3, 2e-4, 5e-5, 2e-5)
- Sweep de beta_end (0.01, 0.02, 0.04)
- Melhores combinações
- ~1 página
-->

*A preencher com resultados Diffusion.*

### 5.4. Comparação Quantitativa Final

| Família | Melhor Config | FID (Média ± Std) | KID (Média ± Std) |
|:---|:---|:---|:---|
| **VAE** | β=0.05, dz=128, lr=2e-3, 200ep | 145.85 ± 1.27 | 0.150 ± 0.009 |
| **GAN** | *a preencher* | ... | ... |
| **Diffusion** | *a preencher* | ... | ... |

<!-- SUGESTÃO: Gráfico de barras com FID e KID side-by-side, com barras de erro -->

Os resultados de VAE apresentados (FID 145.85) são os do treino **PROD** (dataset completo, 200 epochs, 5000 amostras, 10 seeds), representando a avaliação mais fiável.

### 5.5. Exploração Qualitativa

#### 5.5.1. Navegação no Espaço Latente (VAE)

O VAE permite interpolação linear no espaço latente entre dois pontos $z_0$ e $z_1$:

$$z_\alpha = (1 - \alpha) \cdot z_0 + \alpha \cdot z_1, \quad \alpha \in [0, 1]$$

As interpolações geradas mostram transições suaves entre estilos, confirmando que o espaço latente captura variações semânticas contínuas. O CVAE permite adicionalmente interpolações com estilo fixo — e.g., variar o conteúdo mantendo "Impressionism" como condição.

<!-- SUGESTÃO: Incluir strip de interpolação latente (z0 → z1) com 10 passos, do ficheiro latent_interpolation.png -->

#### 5.5.2. Reconstruções

<!-- SUGESTÃO: 
- Incluir grelha de reconstruções (real vs reconstruído) do ficheiro reconstructions.png
- Comentar sobre nível de blur e preservação de estrutura
-->

#### 5.5.3. Evolução das Amostras Durante o Treino

<!-- SUGESTÃO:
- Mostrar samples de epoch 10, 50, 100, 150 side-by-side
- Usa os ficheiros samples_epoch010.png, samples_epoch050.png, etc.
- Comentar sobre emergência progressiva de estrutura visual
-->

*A preencher com amostras geradas.*

---

## 6. Discussion: Strengths and Weaknesses

### 6.1. VAE

**Pontos Fortes:**
- **Estabilidade de treino**: O VAE demonstrou convergência fiável em todas as ~45 experiências. Ao contrário das GANs, não existe risco de mode collapse ou treino instável — mesmo com hiperparâmetros subótimos (e.g., β=2.0), o modelo converge para uma solução (embora de baixa qualidade).
- **Espaço latente estruturado**: A regularização KL garante uma organização suave do espaço latente, permitindo interpolações significativas e amostragem via $z \sim \mathcal{N}(0, I)$.
- **Velocidade de treino**: Treino mais rápido entre as 3 famílias (~2h para 150 epochs no subset 20%), permitindo a extensa exploração de hiperparâmetros realizada.
- **Geração condicional** (CVAE): Capacidade única de controlar o estilo artístico na geração.

**Limitações:**
- **Desfoque visual (blur)**: As amostras geradas pelo VAE sofrem de blur característico, consequência da loss MSE que otimiza a média dos pixels. Este efeito é visível nas amostras e é a principal razão pela qual o FID do VAE é tipicamente superior ao das GANs e Diffusion Models.
- **Sensibilidade ao β**: O FID varia >150 pontos entre o pior e o melhor β. Sem o sweep sistemático conduzido, o modelo teria ficado no β=0.7 subótimo do notebook.
- **Limite arquitetural**: A convergência log-linear sugere que esta ConvVAE com ~1M parâmetros atinge um FID mínimo de ~135–140, independentemente de epochs ou hiperparâmetros. Melhorias adicionais requerem arquiteturas mais poderosas (ResNet, attention layers).

### 6.2. GAN

<!-- SUGESTÃO:
- Nitidez superior ao VAE
- Instabilidade de treino: mode collapse, necessidade de tuning fino
- Sensibilidade à relação de capacidade G/D
-->

*A preencher com discussão GAN.*

### 6.3. Diffusion

<!-- SUGESTÃO:
- Qualidade visual superior (se confirmado)
- Custo computacional: T=1000 steps por amostra
- Tempo de treino mais longo
- Discussão de T como trade-off qualidade/velocidade
-->

*A preencher com discussão Diffusion.*

---

## 7. Conclusion

<!-- SUGESTÃO (~200 palavras):
- O VAE atingiu FID 145.85 (PROD) após otimização exaustiva de ~45 experiências
- O β revelou-se o hiperparâmetro mais crítico, com ótimo em [0.05, 0.1]
- As variantes CVAE e VQ-VAE oferecem vantagens qualitativas mas não superam o β-VAE em FID
- Mencionar qual família venceu overall (provavelmente Diffusion)
- Trabalhos futuros: arquiteturas maiores, perceptual loss otimizada, VQ-VAE com prior autoregressivo
- A investigação de bugs (normalização KL) foi uma lição valiosa sobre debugging científico
-->

*A preencher após todos os resultados estarem completos.*

---

## Sugestões de Visualização (Plots Recomendados)

1. **FID vs β (VAE):** Curva em U invertido mostrando o sweet spot β ∈ [0.05, 0.1]. Dados disponíveis para 30ep e 150ep.

2. **FID vs Epochs:** Curva de convergência log-linear (30→50→100→150 epochs).

3. **Curvas de Loss (VAE):** Gráficos de training_curves.png mostrando Total Loss, Reconstrução e KL ao longo do treino.

4. **Curvas de Loss (GAN):** Gráfico mostrando Generator vs Discriminator loss.

5. **Métricas de Desempenho:** Gráfico de barras com o FID/KID final das 3 famílias + variantes.

6. **Grelha Comparativa:** Comparação visual direta (VAE vs CVAE vs VQ-VAE vs GAN vs Diffusion).

7. **Latent Traversal (VAE):** Interpolação $z_0 \to z_1$ em 10 passos — ficheiro `latent_interpolation.png`.

8. **Reconstruções (VAE):** Real vs Reconstruído — ficheiro `reconstructions.png`.

9. **Evolução Temporal:** Amostras de epochs 10, 50, 100, 150 side-by-side.

10. **Análise de Denoising (Diffusion):** Visualização dos passos de denoising.

---

## Referências

- Bińkowski, M. et al. (2018). Demystifying MMD GANs. *ICLR*.
- Bowman, S.R. et al. (2016). Generating Sentences from a Continuous Space. *CoNLL*.
- Heusel, M. et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS*.
- Higgins, I. et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR*.
- Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- Liao, P. et al. (2022). The ArtBench Dataset: Benchmarking Generative Models with Artworks. *arXiv:2206.11404*.
- Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*.
- van den Oord, A. et al. (2017). Neural Discrete Representation Learning. *NeurIPS*.
- Slides e Notebooks da disciplina de IAG 2025/2026 (FCTUC).
- Documentação oficial do PyTorch e TorchMetrics.
