# Style-Based Generator Architecture for GANs
### Karras, Laine, Aila — NVIDIA, CVPR 2019

> Paper: "A Style-Based Generator Architecture for Generative Adversarial Networks"  
> Implementação neste projecto: `scripts/07_stylegan.py`

---

## Índice

1. [O problema com os geradores tradicionais](#1-o-problema-com-os-geradores-tradicionais)
2. [A arquitectura style-based — visão geral](#2-a-arquitectura-style-based--visão-geral)
3. [Mapping Network: Z → W](#3-mapping-network-z--w)
4. [Synthesis Network: constante → imagem](#4-synthesis-network-constante--imagem)
5. [AdaIN — Adaptive Instance Normalization](#5-adain--adaptive-instance-normalization)
6. [Noise Injection](#6-noise-injection)
7. [Style Mixing Regularization](#7-style-mixing-regularization)
8. [Discriminador — MiniBatch StdDev + Spectral Norm](#8-discriminador--minibatch-stddev--spectral-norm)
9. [R1 Regularization](#9-r1-regularization)
10. [Propriedades do espaço W — Disentanglement](#10-propriedades-do-espaço-w--disentanglement)
11. [Resultados do paper](#11-resultados-do-paper)
12. [Implementação neste projecto](#12-implementação-neste-projecto)

---

## 1. O problema com os geradores tradicionais

Num DCGAN ou ProGAN tradicional, o fluxo é simples:

```
z ∈ Z  →  [ConvTranspose layers]  →  imagem
```

O latente `z` é injectado **uma única vez**, na primeira camada. A partir daí, a rede convolucional processa `z` através de todas as resoluções sem voltar a consultá-lo.

**Problema fundamental: entanglement do espaço Z**

O espaço Z (tipicamente N(0,I)) tem de respeitar a distribuição de probabilidade dos dados de treino. Se certos atributos visuais raramente co-ocorrem no dataset (ex: homens com cabelo comprido), a zona do espaço Z que os representa tem de ser "encurvada" para os excluir. Isto força atributos distintos a ficarem enredados — mudar um factor de variação num ponto de Z afecta outros factores de forma imprevisível.

Consequência: **interpolação não linear** — ao interpolar linearmente entre dois pontos de Z, o caminho percorrido pode atravessar zonas de baixa probabilidade, criando artefactos ou transições abruptas.

---

## 2. A arquitectura style-based — visão geral

```
                   z ∈ Z (N(0,I))
                       │
                  PixelNorm
                       │
              ┌─────────────────┐
              │  Mapping Net f  │  8 camadas FC   (z → w)
              │  Z → W          │
              └─────────────────┘
                       │
                    w ∈ W  (espaço latente intermédio)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼  (mesmo w por camada, ou w diferente com style mixing)
   ┌───────────┐ ┌───────────┐ ┌───────────┐
   │ StyleBlk  │ │ StyleBlk  │ │ StyleBlk  │   ... (2 por resolução)
   │  4×4      │ │  8×8      │ │  16×16    │
   └───────────┘ └───────────┘ └───────────┘
         ▲             ▲             ▲
   Constante     Upsample      Upsample
   aprendida     bilinear      bilinear
   4×4×512
```

**Diferenças fundamentais face ao DCGAN:**

| Aspecto | DCGAN | StyleGAN |
|---|---|---|
| Input do gerador | z → primeira camada | Constante aprendida 4×4 |
| Como z é usado | Propagado pelas convoluções | Convertido em w → estilo em cada camada |
| Controlo de atributos | Implícito, entangled | Explícito, scale-specific |
| Variação estocástica | Vem de z | Ruído independente por camada |
| Espaço latente | Z (Gaussiano) | W (aprendido, mais linear) |

---

## 3. Mapping Network: Z → W

```python
# Paper: 8 camadas FC, dim=512
# Implementação: MAP_LAYERS camadas FC, dim=W_DIM

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, w_dim, n_layers):
        layers = [PixelNorm(), Linear(latent_dim, w_dim), LeakyReLU(0.2)]
        for _ in range(n_layers - 1):
            layers += [Linear(w_dim, w_dim), LeakyReLU(0.2)]
```

### PixelNorm

Antes de entrar na mapping network, `z` é normalizado pixel a pixel:

```
PixelNorm(z) = z / sqrt(mean(z²) + ε)
```

Evita que a magnitude de `z` afecte o treino — a rede aprende apenas a direcção, não a escala.

### Porquê a mapping network funciona

O espaço intermédio **W não está sujeito a nenhuma distribuição imposta**. Enquanto Z tem de ser amostrado de N(0,I), W pode organizar-se livremente para representar os factores de variação do dataset. A mapping network aprende a "deformação inversa" — transforma o Gaussiano esférico Z num espaço W onde atributos distintos ficam em direcções aproximadamente ortogonais.

Evidência do paper (Tabela 4): adicionar uma mapping network com 8 camadas melhora o FID de 5.25 → 4.40 em FFHQ, e a **perceptual path length** cai de 412 → 200, indicando interpolação muito mais suave.

---

## 4. Synthesis Network: constante → imagem

```
Constante aprendida 4×4×512
           │
    StyleBlock (Conv 3×3 → Noise → AdaIN)     ← w controla aqui
    StyleBlock (Conv 3×3 → Noise → AdaIN)     ← w controla aqui
           │
      Upsample 2× (bilinear)
           │
    StyleBlock (8×8)
    StyleBlock (8×8)
           │
      Upsample 2× (bilinear)
           │
    StyleBlock (16×16)
    StyleBlock (16×16)
           │
      Upsample 2× (bilinear)
           │
    StyleBlock (32×32)
    StyleBlock (32×32)
           │
     Conv 1×1 → Tanh
           │
        Imagem
```

**Observação notável do paper:** ao adicionar a mapping network e AdaIN, os autores descobriram que o gerador já **não beneficia** de receber z na primeira camada. Removeram completamente esse input e substituíram por uma constante aprendida 4×4×512. A rede sintetiza imagens exclusivamente através dos estilos injectados via AdaIN — a constante é apenas o ponto de partida estrutural.

---

## 5. AdaIN — Adaptive Instance Normalization

O mecanismo central do StyleGAN. Para cada bloco da synthesis network:

```
AdaIN(xᵢ, y) = yₛ,ᵢ · [(xᵢ - μ(xᵢ)) / σ(xᵢ)] + yᵦ,ᵢ
```

onde:
- `xᵢ` é o feature map do canal i (normalizado para μ=0, σ=1)
- `yₛ,ᵢ` é o scale (γ) derivado de w via transformação afim A
- `yᵦ,ᵢ` é o bias (β) derivado de w via transformação afim A
- A dimensão de y é `2 × número de canais` (metade para scale, metade para bias)

```python
class AdaIN(nn.Module):
    def forward(self, x, style):
        B, C, H, W = x.shape
        gamma = style[:, :C].view(B, C, 1, 1)
        beta  = style[:, C:].view(B, C, 1, 1)
        x_norm = F.instance_norm(x)        # normaliza cada canal
        return gamma * x_norm + beta        # aplica estilo

class StyleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim):
        self.conv  = Conv2d(in_ch, out_ch, 3, padding=1)
        self.style = Linear(w_dim, 2 * out_ch)   # afine A: w → (γ, β)
        self.adain = AdaIN()
```

### Localização do estilo

Cada AdaIN **sobrescreve** as estatísticas do feature map antes do bloco seguinte. Isto significa que o estilo `w` num bloco afecta apenas a convolução imediatamente a seguir — não se propaga para camadas posteriores. Esta propriedade é crucial: permite controlo **scale-specific**:

- **Resoluções baixas (4×4, 8×8):** atributos globais — pose, forma geral, estilo global
- **Resoluções médias (16×16, 32×32):** atributos intermédios — forma do rosto, cabelo, microestrutura  
- **Resoluções altas (64×64+):** detalhes finos — textura de pele, cor dos olhos, microdetalhes

---

## 6. Noise Injection

Cada StyleBlock recebe ruído estocástico independente **por camada**:

```python
class StyleBlock(nn.Module):
    def __init__(self, ...):
        self.noise_scale = nn.Parameter(torch.zeros(out_ch, 1, 1))  # B: escala por canal

    def forward(self, x, w):
        x     = self.conv(x)
        noise = torch.randn(B, 1, H, W, device=x.device)   # ruído espacial
        x     = x + self.noise_scale * noise                # escala aprendida por canal
        style = self.style(w)
        x     = self.adain(x, style)
        return self.act(x)
```

### O que o ruído controla

O ruído é adicionado independentemente por pixel e por camada. O discriminador penaliza qualquer tentativa de usar o ruído para controlo global (ex: pose), pois isso criaria inconsistências espaciais. O gerador aprende espontaneamente a usar o ruído apenas para **variação estocástica local**:

- Pelo cabelo, poros, pecas da pele, detalhes de fundo
- A identidade e pose mantêm-se idênticas entre realizações de ruído diferentes

Evidência visual do paper: a mesma pessoa com diferentes realizações de ruído tem cabelo exactamente diferente mas identidade, expressão e iluminação idênticas.

**Sem ruído:** o paper mostra que as imagens ficam com aparência "pictórica" e excessivamente suave — o gerador tenta criar variação estocástica a partir das activações da rede mas sem sucesso, resultando em texturas repetitivas.

---

## 7. Style Mixing Regularization

Durante o treino, com probabilidade `p` (paper: 90%), dois códigos z são usados em vez de um:

```python
def sample_with_mixing(generator, z, mix_prob, n_blocks):
    if random.random() < mix_prob:
        z2        = torch.randn_like(z)
        mix_layer = random.randint(1, n_blocks - 1)   # ponto de corte aleatório
        return generator(z, z2=z2, mix_layer=mix_layer)
    return generator(z)

# No forward do generator:
def forward(self, z, z2=None, mix_layer=None):
    w  = self.mapping(z)
    w2 = self.mapping(z2) if z2 is not None else None
    for i, block in enumerate(self.blocks):
        w_cur = w2 if (w2 is not None and i >= mix_layer) else w
        x = block(x, w_cur)
```

### Porquê funciona

Sem mixing regularization, a rede pode aprender correlações entre estilos de diferentes resoluções (ex: "se a pose é X, então a textura é Y"). O mixing força a rede a tratar os estilos de diferentes camadas como **independentes** — a resolução baixa de uma imagem pode combinar com a resolução alta de outra.

Consequência: o estilo de cada resolução fica mais "limpo" e controlável. O paper mostra (Figura 3) que é possível transferir apenas os estilos coarse (pose, forma geral) de uma imagem para outra mantendo os fine styles (cor, textura) — ou vice-versa.

**FID improvement (Tabela 2):** com 90% mixing vs 0% mixing, ao gerar com 2 latentes misturados, o FID melhora de 8.22 → 5.11. O modelo fica mais robusto a este tipo de operação porque foi treinado para a suportar.

---

## 8. Discriminador — MiniBatch StdDev + Spectral Norm

O paper herda o discriminador do ProGAN sem alterações significativas. A principal adição é o **Minibatch Standard Deviation**:

```python
class MiniBatchStdDev(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        G = min(self.group_size, B)
        # std entre grupos de G amostras, para cada posição e canal
        y = x.view(G, -1, C, H, W)
        y = (y - y.mean(0, keepdim=True)).pow(2).mean(0).add(1e-8).sqrt()
        y = y.mean([1, 2, 3], keepdim=True).expand(G, -1, H, W).reshape(B, 1, H, W)
        return torch.cat([x, y], dim=1)  # +1 canal: estatística de diversidade
```

Esta camada calcula a **desvio padrão médio entre amostras do batch** e adiciona-o como canal extra antes da última convolução. Dá ao discriminador informação sobre a diversidade das amostras geradas — se o gerador produz amostras muito semelhantes entre si (baixa std), o discriminador detecta-o e penaliza.

Efeito: o gerador é incentivado a manter diversidade entre amostras, mitigando o colapso de modo sem alterar a loss.

---

## 9. R1 Regularization

O paper usa non-saturating GAN loss (igual ao DCGAN padrão) mais R1 regularization (Mescheder et al. 2018):

```
R1 = γ/2 · E[||∇D(x_real)||²]
```

O gradiente é calculado **apenas em imagens reais** (não em imagens interpoladas como no WGAN-GP). O efeito é forçar o discriminador a ser localmente linear na vizinhança dos dados reais — se o discriminador é muito "sharp" (gradiente muito grande) em x_real, é penalizado.

```python
def r1_penalty(discriminator, real):
    real = real.requires_grad_(True)
    d_real = discriminator(real)
    grads  = torch.autograd.grad(d_real.sum(), real, create_graph=True)[0]
    return (grads.view(real.size(0), -1).norm(2, dim=1) ** 2).mean()

# Na loss do discriminador:
d_loss = bce(D(real), ones) + bce(D(fake), zeros)
if batch_idx % R1_EVERY == 0:
    d_loss += (R1_GAMMA / 2) * r1_penalty(D, real) * R1_EVERY
```

**R1 vs WGAN-GP:**

| Aspecto | WGAN-GP | R1 |
|---|---|---|
| Onde calcula o gradiente | Pontos interpolados real+fake | Apenas em reais |
| Objectivo | Penalizar ||∇D(x_interp)|| ≠ 1 | Penalizar ||∇D(x_real)|| grande |
| Complexidade | Requer amostras interpoladas | Só forward/backward nos reais |
| N_CRITIC | Tipicamente 5 | 1 (igual ao gerador) |
| Paper diz | FID continua a descer com R1 por muito mais tempo que com WGAN-GP | |

O paper nota explicitamente que com R1, o FID continua a decrescer durante muito mais tempo do que com WGAN-GP, permitindo treinos mais longos com ganhos consistentes.

---

## 10. Propriedades do espaço W — Disentanglement

O paper propõe duas métricas novas para medir disentanglement (não dependem de encoder):

### 10.1 Perceptual Path Length (PPL)

Mede quão "curvo" é o espaço latente — uma curva pequena indica transições suaves e lineares:

```
lW = E[(1/ε²) · d(g(lerp(w₁, w₂; t)), g(lerp(w₁, w₂; t+ε)))]
```

onde `d(·,·)` é distância perceptual (VGG16 weighted). Valores menores = interpolação mais suave.

**Resultados (Tabela 3):**
- Gerador tradicional (Z): PPL = 412.0
- StyleGAN sem noise (D): PPL = 200.5  
- StyleGAN com noise (E): PPL = 231.5 (noise aumenta ligeiramente a curvatura local)
- Com mixing 90% (F): PPL = 234.0

O espaço W é **cerca de 2× mais linear** que Z no gerador tradicional.

### 10.2 Linear Separability

Mede se os atributos visuais são linearmente separáveis no espaço latente. Para 40 atributos binários (CelebA), treina SVMs lineares e mede entropia condicional H(Y|X). Valores menores = melhor separabilidade.

**Resultados (Tabela 3):**
- Tradicional (Z): separabilidade = 10.78
- StyleGAN W (F): separabilidade = 3.79

W é **quase 3× mais separável** que Z — os atributos visuais estão muito mais linearmente organizados no espaço W.

### 10.3 Truncation Trick em W

O paper propõe aplicar a truncation trick **no espaço W** em vez de Z:

```
w_mean = E[f(z)]   (centro de massa de W)
w' = w_mean + ψ · (w - w_mean)    com ψ ∈ [-1, 1]
```

- `ψ = 1`: amostras normais (máxima diversidade, menor qualidade média)
- `ψ = 0`: "rosto médio" — geração determinística da moda do dataset
- `ψ = -1`: "anti-face" — inversão de atributos (pose oposta, cabelo oposto, etc.)

Vantagem: pode aplicar-se selectivamente só às resoluções baixas, mantendo detalhes finos inalterados.

---

## 11. Resultados do paper

### Ablation (Tabela 1 — FFHQ dataset)

| Config | Descrição | FID |
|---|---|---|
| A | ProGAN baseline | 7.79 |
| B | + bilinear up/down + tuning | 6.11 |
| C | + mapping network + AdaIN | 5.34 |
| D | + remove traditional input (→ constante) | 5.07 |
| E | + noise inputs | 5.06 |
| F | + mixing regularization (90%) | **5.17** |

Observações:
- A maior melhoria isolada é de A→C: apenas a mapping network e AdaIN reduzem o FID de 7.79 → 5.34 (-31%)
- Remover o input tradicional (D) não piora — a constante é suficiente
- Noise melhora subtilmente (E)
- Mixing regularization (F) sobe ligeiramente o FID global mas melhora robustez a operações de mixing

### Impacto da profundidade da mapping network (Tabela 4)

| Profundidade | FID | PPL |
|---|---|---|
| 0 (sem mapping) | 5.06 | 283.5 |
| 1 camada | 4.87 | 219.9 |
| 2 camadas | 4.87 | 217.8 |
| 8 camadas | **4.40** | **234.0** |

Mais camadas FC = melhor FID e melhor disentanglement. O learning rate da mapping network deve ser reduzido em 2 ordens de magnitude face ao resto da rede para evitar instabilidade.

---

## 12. Implementação neste projecto

### Adaptações para 32×32

O paper original gera imagens a 1024×1024 (FFHQ) ou 1024×1024 (CelebA-HQ). Para ArtBench-10 a 32×32:

| Componente | Paper (1024px) | Implementação (32px) |
|---|---|---|
| Mapping depth | 8 camadas FC | 4 camadas FC (MAP_LAYERS) |
| Dimensão w | 512 | 128 (W_DIM) |
| Resoluções | 4→8→16→32→…→1024 | 4→8→16→32 (3 upsamplings) |
| Canais (ngf=64) | 512, 512, 512, 512, 256, 128, 64, 32, 16 | 256, 256, 128, 64, 32 |
| Progressive growing | Sim (começa em 8×8) | Não (desnecessário a 32×32) |
| LR mapping net | 100× menor | **Não implementado** (fixo com o resto) |
| Loss | Non-sat + R1 (γ=10) | Non-sat + R1 (γ=10, a cada 16 batches) |
| Discriminador | ProGAN + MiniBatch StdDev | DCDiscriminator + SN + MiniBatch StdDev |

### Limitação conhecida: LR da mapping network

O paper reduz o LR da mapping network em 100× (`λ_mapping = 0.01 · λ`). Na implementação actual, a mapping network usa o mesmo LR do resto do gerador. Isto pode causar instabilidade com learning rates altos e é o principal desvio face ao paper. Se o treino for instável, reduzir LR_G para 1e-4 ou implementar LR diferenciado por grupo de parâmetros.

### Parâmetros configuráveis (env vars)

| Env var | Default | Significado |
|---|---|---|
| `STYLEGAN_WDIM` | 128 | Dimensão do espaço W |
| `STYLEGAN_MAP_LAYERS` | 4 | Camadas da mapping network |
| `STYLEGAN_R1_GAMMA` | 10.0 | Força da R1 regularization |
| `STYLEGAN_R1_EVERY` | 16 | Frequência do cálculo R1 (batches) |
| `STYLEGAN_MIX_PROB` | 0.9 | Probabilidade de style mixing |
| `DCGAN_NGF` | 64 | Base de canais do gerador |
| `DCGAN_NDF` | 64 | Base de canais do discriminador |
| `DCGAN_LATENT` | 100 | Dimensão do z de entrada |
| `DCGAN_EPOCHS` | cfg.dcgan_epochs | Epochs de treino |

### Testes no PC7

| Experimento | NGF | W_DIM | FID esperado | Baseline comparação |
|---|---|---|---|---|
| `stylegan_default` | 64 | 128 | ~80–100 | default_dcgan (118.82) |
| `stylegan_ngf128` | 128 | 128 | ~65–85 | dcgan_spectral (71.17) |

### O que observar nos resultados

**Style mixing demo** (`style_mixing.png`): grid onde linhas = estilo coarse (pose, forma) e colunas = estilo fine (textura, cor). Se o modelo aprendeu style mixing, cada célula da grid deve combinar a pose da linha com a textura da coluna de forma coerente.

**Latent walk** (`latent_interpolation.png`): interpolação em Z (que passa pela mapping network → interpolação em W). Deve ser mais suave que a interpolação de um DCGAN standard — sem transições abruptas ou artefactos a meio do caminho.

**Curvas de treino** (`training_curves.png`): dois painéis — adversarial losses (D e G) e R1 penalty. O R1 deve decrescer ao longo do treino (discriminador a convergir para regiões mais lineares nas reais). Um R1 que não decresce indica que o discriminador tem dificuldade em memorizar os dados reais.

---

## Sumário executivo

| Pergunta | Resposta |
|---|---|
| O que é o StyleGAN? | GAN onde o gerador aplica estilos (via AdaIN) em cada resolução, a partir de um espaço latente intermédio W mais linear |
| Qual a inovação central? | Mapping network Z→W + AdaIN por camada + input constante (sem z directo no gerador) |
| Porquê W é melhor que Z? | W não está constrangido a seguir N(0,I) — pode organizar-se livremente → 2× mais linear, 3× mais separável |
| O que o noise injection faz? | Separa variação estocástica (textura, poros) dos atributos globais (pose, identidade) |
| O que é style mixing? | Regularização que força estilos de resoluções diferentes a serem independentes — permite controlo scale-specific |
| Como é mais simples que WGAN-GP? | R1 regularization: gradiente só em reais, sem interpolação, sem N_CRITIC=5 |
| O que esperar em ArtBench 32×32? | FID possivelmente competitivo com dcgan_spectral (71.17); a principal vantagem é qualitativa — controlo de estilo |

---

*Documento criado em 2026-04-24 com base em Karras et al. 2019 (arXiv:1812.04948v3).*
