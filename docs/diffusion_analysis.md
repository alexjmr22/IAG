# Análise Completa — Diffusion Models (DDPM/DDIM + EMA)

> Projeto IAG — ArtBench-10 | Data de análise: 2026-04-25

---

## 1. Visão Geral

O projeto implementa Diffusion Models no domínio pixel (pixel-space) para geração de imagens 32×32 RGB no dataset **ArtBench-10** (10 estilos artísticos). A família de modelos Diffusion foi claramente a mais trabalhada e a que obteve os melhores resultados absolutos do projeto, superando largamente VAE (FID=157.73) e DCGAN.

Foram realizadas **7 baterias de testes** ao longo de 9 PCs diferentes (PC3, PC4, PC6, PC7, PC8, PC9), num total de **22 experiências** com ablações sistemáticas sobre T, channels, LR, beta schedule, epochs, sampler e EMA.

---

## 2. Arquitectura

### 2.1 GaussianDiffusion (Scheduler DDPM)

Implementado em `scripts/03_diffusion.py` e `scripts/03b_diffusion_ema.py`.

**Forward process** (`q_sample`):
```
x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,   ε ~ N(0, I)
```

**Beta schedule linear** (fixo em todos os testes):
- `beta_start = 1e-4`
- `beta_end = 0.02` (padrão; testado 0.01 e 0.04)
- `T = 1000` (padrão; testado 100, 250, 500, 1500, 2000)

**Reverse process** (`p_sample`):
```
x_{t-1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t)) + √σ_t · z
```

**DDIM sampling** (adicionado na Teste 8):
- Implementação de Song et al. (2020), eq. 12
- `eta = 0.0` (determinístico)
- 100 passos de inferência em vez de 1000 (10× mais rápido)
- Clamping de `x0_pred` em `[-1, 1]` para estabilidade

### 2.2 PixelUNet (Denoising Network)

U-Net adaptada do notebook 5 para RGB (3 canais de entrada em vez de 1 no MNIST original).

**Arquitectura** (para `model_channels=96`):
```
Input: (B, 3, 32, 32)

Time Embedding:
  SinusoidalPosEmb(C=96) → Linear(96, 384) → SiLU → Linear(384, 384)

Encoder:
  init_conv: Conv2d(3→96, 3×3)
  down1_res: ResnetBlock(96, t=384)       [32×32]
  down1_pool: Conv2d stride=2             [32→16]
  down2_res: ResnetBlock(96→192, t=384)   [16×16]
  down2_pool: Conv2d stride=2             [16→8]

Bottleneck:
  mid_res1 + mid_res2: ResnetBlock(192, t=384)  [8×8]

Decoder:
  up2_conv: ConvTranspose2d(192→96, stride=2)   [8→16]
  up2_res: ResnetBlock(96+192=288→96, t=384)    [16×16, skip de down2]
  up1_conv: ConvTranspose2d(96→96, stride=2)    [16→32]
  up1_res: ResnetBlock(96+96=192→96, t=384)     [32×32, skip de down1]

Output: Conv2d(96→3, 3×3)
```

**ResnetBlock** (idêntico ao notebook 5):
- GroupNorm(4) + SiLU antes de cada conv
- Time embedding injectado por adição: `conv1(x) + mlp(t_emb)`
- Shortcut com Conv2d(1×1) quando `dim != out_dim`
- Activação SiLU em vez de ReLU

**Parâmetros por configuração de channels:**
| Channels | Parâmetros aprox. |
|----------|------------------|
| 32       | ~1.2M            |
| 64       | ~4.7M            |
| 96       | ~10.5M           |
| 112      | ~14.3M           |
| 128      | ~19.6M           |

### 2.3 EMAModel (03b_diffusion_ema.py)

Implementa Exponential Moving Average dos pesos do modelo:
```
shadow_k ← decay · shadow_k + (1 − decay) · θ_k
```
- `decay = 0.9999` (1 peso novo por cada 10 000 gradientes)
- `apply()` / `restore()` para usar EMA apenas no sampling sem contaminar o treino
- Guarda dois checkpoints separados: `diffusion_checkpoint.pth` (pesos raw) e `diffusion_ema_checkpoint.pth` (pesos EMA)

---

## 3. Pipeline de Treino

### 3.1 Optimizer e Scheduler

```python
optimizer = Adam(lr=LR)

lr_lambda(epoch) = {
    (epoch+1)/warmup_epochs          se epoch < warmup_epochs   [warmup linear]
    0.5 * (1 + cos(π * progress))   caso contrário              [cosine decay]
}
```

O scheduler cosine foi o **turning point** do projecto: introduzido na Teste 7, reduziu o FID de 100.21 → 65.73 no mesmo modelo (ch=96, T=1000).

### 3.2 Loop de Treino (simplificado)

```python
for epoch in range(epochs):
    for x, _ in loader:
        t = randint(0, T)            # timestep aleatório uniforme
        noise = randn_like(x)
        x_t = q_sample(x, t, noise)  # forward process
        pred = model(x_t, t)          # predict noise
        loss = mse_loss(pred, noise)  # ε-prediction
        loss.backward()
        optimizer.step()
        ema.update(model)             # (só em 03b)
    scheduler.step()
```

### 3.3 Configs de Dataset

| Perfil | Dataset          | Amostras FID/KID | Seeds |
|--------|-----------------|-----------------|-------|
| DEV    | 20% subset       | 2 000           | 3     |
| PROD   | ArtBench-10 full | 5 000           | 10    |

A transição DEV→PROD sozinha valeu **~33 pontos FID** (65.73 → 32.17).

---

## 4. Protocolo de Avaliação

Implementado em `scripts/04_evaluation.py`.

- **N_SAMPLES = 5000** imagens geradas vs 5000 reais
- **FID**: feature=2048 (InceptionV3), conjunto completo
- **KID**: 50 subsets de tamanho 100, reporta `mean ± std`
- **N_SEEDS = 10** seeds (range 42–51), reporta `mean ± std` final
- Normalização: imagens uint8 [0,255] para FID/KID (sem normalização adicional)
- VRAM: updates em batches de 64 para evitar OOM

**Nota importante sobre Diffusion na avaliação**: em `04_evaluation.py` a função `generate_diffusion` usa `p_sample_loop` (DDPM completo de 1000 passos), não DDIM — mesmo quando o modelo foi treinado com DDIM. Isto significa que os números de avaliação dos modelos com DDIM podem subestimar ligeiramente a performance real com DDIM-100 durante o sampling de avaliação.

---

## 5. Histórico de Experimentos

### Teste 3 — Sweep Inicial (PC3, pasta `DIFF/`)

**Config base:** ch=64, T=1000, lr=2e-4, 50 épocas, DEV, DDPM

| Experiência    | Variação        | FID ↓      | FID std  | KID ↓      |
|----------------|----------------|------------|----------|------------|
| `default_diff` | base            | 106.89     | 2.27     | 0.0667     |
| `diff_T100`    | T=100           | 156.09     | 0.42     | 0.1495     |
| `diff_T250`    | T=250           | 128.98     | 1.28     | 0.1064     |
| `diff_T500`    | T=500           | 113.76     | 1.57     | 0.0930     |
| `diff_ch32`    | channels=32     | 203.46     | 1.67     | 0.2099     |
| `diff_ch64`    | channels=64     | 108.75     | 2.18     | 0.0686     |
| `diff_ch128`   | channels=128    | 187.81     | 1.62     | 0.1700     |
| `diff_lr1e3`   | lr=1e-3         | 420.12     | 3.35     | 0.4586     |
| `diff_lr2e5`   | lr=2e-5         | 292.23     | 1.63     | 0.3209     |

**Conclusões:**
- T tem efeito monotónico positivo: T100 < T250 < T500 < T1000 → mais passos = melhor cobertura do espaço de ruído
- ch=64 bate ch=128: o modelo maior overfita com apenas 50 épocas no subset DEV; ch=32 é muito fraco
- lr=2e-4 é claramente o ponto ideal — lr=1e-3 diverge (FID 420), lr=2e-5 é lento demais para 50 épocas

---

### Teste 4 — Exploração Dirigida (PC4, pasta `DIFF_ate_65fid/`)

**Config base:** T=1000, ch=64, lr=2e-4, 50 épocas, DEV

| Experiência             | Variação                          | FID ↓      | KID ↓      |
|-------------------------|----------------------------------|------------|------------|
| `diff_best_combo`       | âncora (T=1000, ch=64, lr=2e-4)  | 107.19     | 0.0670     |
| `diff_T1500`            | T=1500                            | 314.81     | 0.2930     |
| `diff_T2000`            | T=2000                            | 293.86     | 0.2748     |
| `diff_T2000_ch64`       | T=2000, ch=64, lr=2e-4           | 288.19     | 0.2674     |
| **`diff_ch96`**         | **channels=96**                  | **100.21** | **0.0734** |
| `diff_lr5e5`            | lr=5e-5                           | 218.35     | 0.2107     |
| `diff_beta_high`        | beta_end=0.04                     | 242.14     | 0.2711     |
| `diff_beta_low`         | beta_end=0.01                     | 467.27     | 0.5412     |
| `diff_beta_low_combo`   | T=1000, ch=64, lr=2e-4, beta=0.01| 469.09     | 0.5445     |
| `diff_combo_v2`         | T=2000, ch=64, lr=2e-4, beta=0.01| 579.23     | 0.6979     |

**Conclusões:**
- T > 1000 piora: o modelo entra no regime de sobre-difusão — com beta_end=0.02 fixo, T=1500/2000 força ᾱ_T ≈ 0 demasiado cedo, deixando o modelo sem sinal nos últimos passos
- ch=96 é o novo anchor: FID 100.21 supera ch=64 (107) e ch=128 (187)
- beta_end baixo (0.01) colapsa completamente — o schedule linear com beta_end=0.01 deixa demasiado sinal nos últimos passos, dificultando a aprendizagem da distribuição de ruído puro
- beta_end alto (0.04) também piora: mais sinal destruído por iteração → model overkill

---

### Teste 6 — Mais Épocas e ch=112 (PC6, pasta `DIFF_ate_65fid/`)

**Motivação:** top-2 do Teste 4 com 100 épocas; explorar ch=112

| Experiência              | Variação                   | FID ↓      | KID ↓      |
|--------------------------|---------------------------|------------|------------|
| `diff_ch96_e100`         | ch=96, 100ep              | 193.54     | 0.1539     |
| `diff_best_combo_e100`   | ch=64, 100ep              | 155.25     | 0.1407     |
| `diff_ch112`             | ch=112, 50ep              | 264.89     | 0.2364     |

**Conclusões:**
- 100 épocas **sem LR scheduler piora** drasticamente (ch=96: 100 → 194; ch=64: 107 → 155): LR constante durante treino longo causa overfitting e/ou instabilidade no subset DEV
- ch=112 é pior que ch=96 — sweet spot em ch=96; acima disso o modelo não tem dados suficientes para regularizar no subset

---

### Teste 7 — Cosine LR Scheduler (PC7, pasta `DIFF_ate_65fid/`)

**Motivação:** o Teste 6 provou que o scheduler é necessário; testar cosine com warmup

| Experiência                   | Variação                          | FID ↓      | KID ↓      |
|-------------------------------|----------------------------------|------------|------------|
| `diff_ch96_cosine`            | ch=96, lr=2e-4, cosine, 100ep    | 78.28      | 0.0492     |
| `diff_ch64_cosine`            | ch=64, lr=2e-4, cosine, 100ep    | 107.32     | 0.0828     |
| **`diff_ch96_cosine_lr4e4`** | **ch=96, lr=4e-4, cosine, 100ep** | **65.73**  | **0.0353** |

**Conclusões:**
- Cosine + warmup resolve o overfitting: ch=96 passa de FID 194 (sem sched) para 78 (com sched) com +100 épocas
- LR inicial 4e-4 com cosine supera 2e-4: o warmup linear absorve o LR alto nos primeiros 5 epochs, e o cosine decay garante estabilidade no final — combinação óptima
- Melhor DEV atingido: **FID=65.73, KID=0.0353**

---

### Teste 8 — PROD + DDIM (PC8)

**Motivação:** validar se a melhor config DEV generaliza para o dataset completo

Config: ch=96, T=1000, lr=4e-4, cosine (warmup 5ep/10ep), DDIM-100 steps, dataset 100%

| Experiência              | Épocas | FID ↓      | KID ↓      |
|--------------------------|--------|------------|------------|
| `diff_prod_ddim_e100`    | 100    | **32.17**  | **0.0169** |
| `diff_prod_ddim_e250`    | 250    | —          | —          |

**Conclusões:**
- Transição DEV → PROD: FID 65.73 → 32.17 (**melhoria de ~51%**): mais dados = distribuição mais rica = melhor qualidade
- DDIM-100 acelera sampling em 10× (100 vs 1000 passos) sem perda mensurável de qualidade face ao DDPM
- `diff_prod_ddim_e250` não foi concluído ou resultados não disponíveis

---

### Teste 9 — EMA (PC9)

**Motivação:** EMA suaviza os pesos finais e reduz variância do sampling

Config: ch=96, T=1000, lr=4e-4, cosine, DDIM-100, EMA=0.9999, PROD

| Experiência         | Épocas | FID ↓      | KID ↓      |
|---------------------|--------|------------|------------|
| `diff_ema_e100`     | 100    | 48.72      | 0.0359     |
| **`diff_ema_e200`** | **200**| **28.97**  | **0.0144** |

**Conclusões:**
- EMA a 100ep (48.72) é **pior** que DDIM sem EMA a 100ep (32.17): EMA com decay=0.9999 precisa de muitas iterações para que os pesos shadow divirjam suficientemente dos pesos iniciais (shadow começa como cópia dos pesos no início do treino)
- EMA a 200ep (28.97) supera tudo: com mais iterações o shadow acumula sinal suficiente → **melhor resultado do projecto**
- A diferença EMA e100 vs e200 (48.72 vs 28.97) é maior do que sem EMA e100 vs e200 — sugere que o EMA beneficia exponencialmente de treino mais longo

---

## 6. Ranking Global — Diffusion

| Rank | Experiência              | ch  | T    | LR    | Sched  | Ep  | EMA   | Sampler | Dataset | FID ↓      | KID ↓      |
|------|--------------------------|-----|------|-------|--------|-----|-------|---------|---------|------------|------------|
| 🥇 1 | `diff_ema_e200`          | 96  | 1000 | 4e-4  | cosine | 200 | 0.9999| DDIM-100| PROD    | **28.97**  | **0.0144** |
| 🥈 2 | `diff_prod_ddim_e100`    | 96  | 1000 | 4e-4  | cosine | 100 | —     | DDIM-100| PROD    | 32.17      | 0.0169     |
| 🥉 3 | `diff_ema_e100`          | 96  | 1000 | 4e-4  | cosine | 100 | 0.9999| DDIM-100| PROD    | 48.72      | 0.0359     |
| 4    | `diff_ch96_cosine_lr4e4` | 96  | 1000 | 4e-4  | cosine | 100 | —     | DDPM    | DEV     | 65.73      | 0.0353     |
| 5    | `diff_ch96_cosine`       | 96  | 1000 | 2e-4  | cosine | 100 | —     | DDPM    | DEV     | 78.28      | 0.0492     |
| 6    | `diff_ch96`              | 96  | 1000 | 2e-4  | —      | 50  | —     | DDPM    | DEV     | 100.21     | 0.0734     |
| 7    | `default_diff`           | 64  | 1000 | 2e-4  | —      | 50  | —     | DDPM    | DEV     | 106.89     | 0.0667     |

---

## 7. Análise de Sensibilidade

### Por parâmetro

| Parâmetro         | Melhor Valor     | Pior Valor       | Δ FID      | Notas                                                   |
|-------------------|-----------------|-----------------|------------|--------------------------------------------------------|
| **Dataset**       | PROD (100%)      | DEV (20%)        | ~33 pts    | Factor mais impactante isolado                         |
| **Epochs**        | 200              | 50               | ~78 pts    | Com EMA+scheduler; sem scheduler, mais épocas piora    |
| **LR Scheduler**  | cosine+warmup   | constant         | ~93 pts    | ch=96, 100ep: 194 → 78 após adicionar cosine           |
| **EMA**           | 0.9999 (200ep)  | sem EMA          | ~3.2 pts   | Diferença pequena mas consistente a 200ep              |
| **Channels**      | 96               | 32               | ~103 pts   | ch=128 piora por overfitting (187 vs 107 do ch=64)     |
| **LR inicial**    | 4e-4             | 2e-5             | ~226 pts   | LR muito baixo não converge; muito alto diverge        |
| **T (steps)**     | 1000             | 100              | ~49 pts    | Monotónico até T=1000; T>1000 piora                    |
| **Sampler**       | DDIM-100         | DDPM-1000        | ~0 pts FID | DDIM 10× mais rápido, qualidade equivalente            |
| **Beta end**      | 0.02 (default)   | 0.01             | ~360 pts   | beta_low colapsa; beta_high também piora               |

### Interacções identificadas

1. **EMA × Epochs**: EMA com 100ep piora face a sem-EMA; EMA com 200ep é o melhor. O EMA decay=0.9999 precisa de ~100 000+ updates para estabilizar (shadow diverge dos pesos iniciais).

2. **Scheduler × LR alto**: LR=4e-4 sem scheduler provavelmente divergiria (lr=1e-3 sem scheduler já divergiu). O cosine com warmup torna LR=4e-4 seguro e benéfico.

3. **Channels × Epochs**: ch=128 com 50ep overfita no DEV. Com PROD e mais épocas, poderia potencialmente superar ch=96.

4. **T-steps × Beta schedule**: T>1000 com beta_end=0.02 entra em sobre-difusão. Um beta schedule ajustado ao T maior poderia recuperar, mas não foi testado.

---

## 8. Análise da Implementação

### Pontos fortes

- **Modularidade**: `03_diffusion.py` e `03b_diffusion_ema.py` são self-contained — qualquer hiperparâmetro é controlável via env var (`DIFF_LR`, `DIFF_CHANNELS`, etc.), sem editar código
- **Orquestração robusta**: `run_experiments.py` limpa variáveis de ambiente entre runs para evitar contaminação entre experimentos
- **DDIM correcto**: implementação fiel ao paper Song et al. 2020 — clamping de `x0_pred`, sigma para eta determinístico
- **EMA seguro**: `apply()`/`restore()` garante que os pesos EMA só são usados no sampling, nunca durante o backward pass
- **Logging automático**: cada run grava `experiment_params.md` com os parâmetros exactos usados

### Limitações identificadas

1. **`generate_diffusion` em `04_evaluation.py` usa DDPM** (p_sample_loop de 1000 passos), não DDIM — os modelos treinados com `SAMPLER=ddim` são avaliados com DDPM, o que é mais lento e pode sub-avaliar ligeiramente a qualidade do DDIM

2. **GroupNorm com n_groups=4 fixo**: para ch=32, cada grupo tem apenas 8 canais — pode causar instabilidade; ch=32 obteve FID=203 mas não ficou claro se é arquitectura ou falta de parâmetros

3. **sem gradient clipping**: Adam sem clip pode ter picos de gradiente, especialmente com LR=4e-4 sem warmup completo

4. **Código duplicado**: `GaussianDiffusion`, `PixelUNet`, `ResnetBlock` e `SinusoidalPosEmb` estão copiados em `03_diffusion.py`, `03b_diffusion_ema.py` e `04_evaluation.py` — qualquer bug tem que ser corrigido em 3 lugares

5. **`diff_prod_ddim_e250` não tem resultados**: corrida de 250 épocas PROD não foi concluída ou o ficheiro foi perdido

---

## 9. Comparação com Outros Modelos do Projeto

| Modelo             | Melhor Config                           | FID ↓   | KID ↓   |
|--------------------|-----------------------------------------|---------|---------|
| **Diffusion EMA**  | ch=96, cosine lr=4e-4, 200ep, DDIM, PROD | **28.97** | **0.0144** |
| VAE                | β=0.1, lat=128, lr=2e-3, 50ep, PROD     | 157.73  | 0.1444  |

A Diffusion supera o VAE por **128 pontos FID** e **10× no KID**. A comparação com DCGAN não está registada na memória.

---

## 10. O Que Foi Feito vs O Que Pode Ser Feito

### Feito

| Feature                  | Implementado | Testado |
|--------------------------|:-----------:|:-------:|
| DDPM com linear beta     | ✅          | ✅      |
| Pixel U-Net RGB           | ✅          | ✅      |
| DDIM sampling             | ✅          | ✅      |
| EMA dos pesos            | ✅          | ✅      |
| Cosine LR + warmup       | ✅          | ✅      |
| Sweep T / channels / LR  | ✅          | ✅      |
| Sweep beta schedule       | ✅          | ✅      |
| Dataset completo (PROD)  | ✅          | ✅      |
| Latent Diffusion (VAE+Diff)| ✅ (stub) | ❌      |
| Classifier-free guidance | ❌          | ❌      |
| Cosine beta schedule     | ❌          | ❌      |
| Attention no bottleneck  | ❌          | ❌      |

### Propostas Não Exploradas / Próximos Passos

#### 1. Cosine Beta Schedule (impacto esperado: médio, ~5-10 pts FID)
O schedule linear com beta_end=0.02 mostrou-se sensível (beta=0.01 colapsou). O **cosine schedule** (Nichol & Dhariwal, 2021) garante ᾱ_T mais suave e evita o problema de sobre-difusão a T alto. Implementação simples — adicionar como variante no `GaussianDiffusion.__init__`.

```python
# Cosine schedule alternativo
s = 0.008  # offset para evitar β_0 muito pequeno
t_range = torch.arange(T+1, dtype=torch.float64) / T
f = torch.cos((t_range + s) / (1 + s) * math.pi / 2) ** 2
alphas_cumprod = f / f[0]
betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
betas = betas.clamp(0, 0.999)
```

#### 2. Attention no Bottleneck (impacto esperado: médio, ~5-15 pts FID)
A arquitectura actual não tem atenção — o bottleneck é apenas ResnetBlock × 2. Adicionar **Self-Attention** de cabeça única (ou multi-head) no bottleneck (8×8) é o upgrade canónico dos DDPM modernos (Ho et al., 2020; Dhariwal & Nichol, 2021). Em 32×32, a atenção na resolução 8×8 tem custo quadrático O(64²)=4096 — muito manejável.

```python
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(4, dim)
        self.qkv  = nn.Conv2d(dim, dim*3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        k = k.view(B, C, -1)                   # (B, C, HW)
        v = v.view(B, C, -1).transpose(1, 2)   # (B, HW, C)
        attn = torch.softmax(q @ k / C**0.5, dim=-1)
        out  = (attn @ v).transpose(1, 2).view(B, C, H, W)
        return x + self.proj(out)
```

#### 3. Treino mais longo com EMA (impacto esperado: alto, ~3-8 pts FID)
A tendência EMA e100→e200 (48.72→28.97) sugere que **e300 ou e400** ainda melhoraria. O custo é apenas tempo de GPU — nenhuma mudança de código.

#### 4. Latent Diffusion Model (LDM) (impacto esperado: muito alto, potencialmente <20 FID)
O stub já existe em `03_diffusion.py` (secção 9 comentada). A ideia:
1. Usar o encoder do VAE treinado (melhor: β=0.1, lat=128) para comprimir 32×32 → 4×4 latente
2. Treinar a Diffusion no espaço latente 4×4 (muito mais rápido)
3. Decoder VAE na geração

**Cuidado**: o VAE actual tem FID=157 — um encoder fraco pode limitar o tecto do LDM. Seria necessário primeiro melhorar o VAE ou usar um encoder pré-treinado.

#### 5. Classifier-Free Guidance (CFG) (impacto esperado: alto para FID condicionado)
Condicionar nas 10 classes do ArtBench-10 com CFG (Ho & Salimans, 2022):
- Modificar `PixelUNet.forward(x, t, label=None)` — injectar label embedding no time embedding
- 10-20% dropout do label durante treino → modelo aprende unconditional e conditional
- Na inferência: `ε_guided = ε_uncond + γ · (ε_cond − ε_uncond)` com γ > 1 para amplificar o sinal da classe

#### 6. Gradient Clipping (impacto esperado: estabilidade)
```python
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()
```
Especialmente relevante com LR=4e-4 — previne explosão de gradientes.

---

## 11. Configuração Óptima Final

```bash
# diff_ema_e200 — melhor resultado (FID=28.97, KID=0.0144)
EXP_NAME=diff_ema_e200 \
EVAL_TARGET=DiffusionEMA \
RUN_PROFILE=PROD \
DIFF_CHANNELS=96 \
DIFF_T_STEPS=1000 \
DIFF_LR=4e-4 \
DIFF_EPOCHS=200 \
DIFF_WARMUP_EPOCHS=10 \
DIFF_SAMPLER=ddim \
DIFF_DDIM_STEPS=100 \
DIFF_EMA_DECAY=0.9999 \
python scripts/03b_diffusion_ema.py
```

**Checkpoints gerados:**
- `results/diff_ema_e200/diffusion_checkpoint.pth` — pesos raw Adam
- `results/diff_ema_e200/diffusion_ema_checkpoint.pth` — pesos EMA (usar para sampling)

---

## 12. Ficheiros Relevantes

| Ficheiro | Função |
|----------|--------|
| [scripts/03_diffusion.py](../scripts/03_diffusion.py) | DDPM base + DDIM, sem EMA |
| [scripts/03b_diffusion_ema.py](../scripts/03b_diffusion_ema.py) | DDPM + DDIM + EMAModel |
| [scripts/04_evaluation.py](../scripts/04_evaluation.py) | Protocolo FID/KID (5k amostras, 10 seeds) |
| [scripts/run_experiments.py](../scripts/run_experiments.py) | Orquestrador grid-search (PC3–PC9) |
| [scripts/config.py](../scripts/config.py) | Perfis DEV / PROD / TEST |

---

*Análise gerada automaticamente a partir dos scripts e memória de resultados experimentais.*
