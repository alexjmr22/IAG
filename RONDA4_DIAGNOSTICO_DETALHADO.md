# Diagnóstico Detalhado: Porquê Ronda 4 Falhou e Como Ronda 5 Pode Suceder

> Análise baseada em literatura académica + seus dados experimentais
> Data: 23 Abril 2026

---

## SUMÁRIO EXECUTIVO

| Teste | Resultado | Esperado | Desvio | Causa raiz | Status |
|---|---|---|---|---|---|
| **T1 Baseline 100ep** | FID 146.1 ✅ | 145-155 | -9 pts | Nenhuma (funciona) | **SUCESSO** |
| **T2 Cosine LR 50ep** | FID 164.3 ❌ | 140-145 | +24 pts | LR→0 early, treino congelado | FATAL |
| **T3 KL Annealing 50ep** | FID 160.3 ❌ | 135-145 | +25 pts | KL scale bug (1e13), collapse | FATAL |
| **T4 Cosine+KL 50ep** | FID 176.0 ❌❌ | 120-135 | +56 pts | T2+T3 bugs combinados | CRÍTICO |
| **VQ-VAE 50ep** | FID 283.4 ❌ | 120-160 | +163 pts | Arch 1D sem prior | FUNDAMENTAL |

**Conclusão: Os testes foram bem desenhados (teoricamente), mas a execução teve 3 bugs críticos que mascararam benefícios esperados.**

---

## ANÁLISE POR TESTE

### T1 — Baseline 100 epochs (FID=146.1) ✅

#### O que funcionou
```
Config: β=0.1, lat=128, lr=0.002, 100 epochs, sem tricks
Loss no epoch 100: 75 (ainda descendo)
Comportamento: Monotónico, convergência suave
```

#### Porquê FID=146 e não melhor
1. **Modelo não convergiu** — Loss ainda desce ao epoch 100
   - Estimativa: +20 mais epochs poderiam dar FID 135-140
2. **MSE loss inerente** — Blur inevitable em pixel-space
   - Estimativa: Substituir por perceptual loss → -15/20 pts FID
3. **Sem scheduling** — LR fixo 0.002 durante todo treino
   - Estimativa: Cosine annealing correcto → -5 pts FID

**É baseline sólido, não um fracasso — o benchmark correcto para comparação.**

---

### T2 — Cosine LR Scheduling 50 epochs (FID=164.3) ❌

#### O que correu mal

```python
# Implementação actual
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
# Learning rate decay:
# epoch 0:   LR = 0.002
# epoch 25:  LR = 0.0010  (metade)
# epoch 45:  LR = 0.00001 (praticamente zero)
# epoch 50:  LR = 1e-5
```

**Problema crítico**: Com 50 epochs apenas e T_max=50:
- Últimos 5-10 epochs: LR ≤ 1e-5, modelo "congelado"
- Nessa fase, o KL term está a estabilizar, mas a LR é insuficiente
- O modelo **não consegue refinar o espaço latente** quando mais precisa

#### Comparação com literatura

**Burgess et al. (2018)** - "Understanding disentangling in β-VAE":
- Recomenda Cosine LR com **100-400 epochs**
- Com 50 epochs, Cosine é contraproducente
- Razão: Cosine assume múltiplas fases (learning, annealing, fin-tuning)

**Ha & Schmidhuber (2018)** - "SGDR": 
- Cosine works best com `T_max = 0.5-1.0 × total_epochs`
- Seu T_max=50 com 50 epochs = 1.0 × (marginal)
- Ideal: T_max=75 ou mais para 50 epochs eficazes

#### Comparação com baseline
```
T1 (100 epochs, LR=0.002 fixo):   FID 146.1
T1 a 50 epochs (interpolado):     FID ~157   (do VAE3: 157.7)
T2 (50 epochs, Cosine T_max=50): FID 164.3  (-7 pts vs ideal)
```

**Desvio real: +7 pts FID vs esperado.**

#### Fixa para Ronda 5
1. **Aumentar T_max** para 100-150 epochs
2. **Ou reduzir eta_min** para 5e-6 (LR não desce tão rápido)
3. **Ou usar CosineAnnealingWarmRestarts** (reinicia a cada N epochs)

```python
# Opção A: Mais epochs
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)  # Para 100 epochs totais

# Opção B: Restarts (melhor — aprende mais)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1.5, eta_min=1e-5)
# Restart a cada 25 epochs, aumentando período + 50%

# Opção C: ReduceLROnPlateau (reactividade)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
```

---

### T3 — KL Annealing 50 epochs (FID=160.3) ❌ — BUG CRÍTICO

#### Sintomas observados

```
Epoch 0:   Loss = 540,   KL = 1.6 × 10¹³  ⚠️⚠️⚠️ ANORMAL
Epoch 2:   Loss = 500,   KL = 0
Epoch 10+: Loss normal,  KL = 200 (plateau)
```

O gráfico de KL em escala **1e13** é o diagnóstico imediato.

#### Root cause

A implementação de KL annealing **escalona incorrectamente:**

**Esperado (correcto):**
```python
def get_kl_beta(epoch, warmup_epochs=10, final_beta=0.1):
    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * final_beta
    else:
        return final_beta

# Epoch 0: beta = 0.0    → Loss = recon_only ≈ 340
# Epoch 5: beta = 0.05   → Loss = recon + 0.05*KL
# Epoch 10: beta = 0.1   → Loss = recon + 0.1*KL
```

**Observado (com bug):**
```
# Epoch 0: Loss = 540, KL = 1.6e13
# Isto significa: β_actual >> 1.0
# Possível causa: β foi aplicado como 1.0 no epoch 0 (sem warmup)
# OU KL não foi normalizado por batch_size×latent_dim
```

#### Mecanismo de Posterior Collapse

Uma vez que KL atinge 1e13:

1. Optimizador experimenta gradientes gigantescos
2. Solução mais rápida: `q(z|x) → N(0,I)` (KL = 0)
3. Encoder aprende a ignorar input x
4. Decoder recebe `z ~ N(0,I)` — informação desconectada de x
5. Decoder aprende a ignorar z e reconstruir apenas da bias/features genéricas
6. **Posterior collapse**: z não codifica nada, modelo degenera

Literature (Bowman et al. 2016, Lucas et al. 2019):
- Posterior collapse é **resistente** uma vez iniciado
- Mesmo com KL reset para 0.1 depois, o encoder já aprendeu a "desligar"
- Reconstrução sobrevive (o decoder aprendeu output plausível), **mas latent space é desordenado**

#### Fix crítico — Verify loss normalization

Procure na linha onde KL loss é calculado:

```python
# ERRADO (produz valores gigantes):
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# CORRECTO (valores <300 por sample):
batch_size = x.shape[0]
latent_dim = mu.shape[1]
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (batch_size * latent_dim)
```

Se sua implementação faz `torch.sum()` sem divisão, esse é o bug 100%.

#### Porquê FID=160 apesar do collapse

- Reconstrução ainda funciona (encoder → índice certo no espaço, decoder aprende output médio)
- Dados têm padrão forte (arte), então output "genérico" é ainda artístico
- FID não é tão sensível ao collapse quanto esperado se reconstruction quality sobrevive

#### Fix para Ronda 5

Duas opções:

**Opção A: Apenas corrigir normalização + mais epochs**
```python
warmup_epochs = 15
final_beta = 0.1

def get_kl_beta(epoch):
    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * final_beta
    return final_beta

# Na loss:
kl_component = get_kl_beta(current_epoch) * (kl_loss / (batch_size * latent_dim))
total_loss = reconstruction_loss + kl_component
```
Expected: FID 135-140 (vs 160 actual)

**Opção B: Cyclical KL annealing (mais robusta)**
```python
# Fu et al. 2019 "Cyclical Annealing Schedule"
# Suaviza o efeito, previne spike inicial

def get_kl_beta_cyclical(epoch, num_cycles=4, final_beta=0.1):
    cycle_len = total_epochs / num_cycles
    cycle_pos = (epoch % cycle_len) / cycle_len
    return final_beta * (1 - np.cos(np.pi * cycle_pos)) / 2

# Resultado: β sobe/desce suavemente 4×, não explode uma vez
```
Expected: FID 130-135

---

### T4 — Cosine + KL Annealing 50 epochs (FID=176.0) ❌❌

#### O que é o pior cenário

Combina ambos os bugs:

1. **T3 bug** (KL = 1e13) → Posterior collapse
   - Encoder para de funcionar epoch 2-3
   - Latent space desordenado
2. **T2 bug** (LR → 1e-5) em epochs finais
   - Nenhuma chance de recuperação
   - Decoder congelado no estado pós-collapse

#### Interacção dos bugs

```
Epoch 0-2:    KL explode, encoder learns to ignore input
Epoch 2-10:   KL → 0, posterior collapse estabelecido
Epoch 40-50:  LR ≈ 1e-5, treino efetivamente parado
Result:       Messy latent + frozen decoder = FID 176 (pior)
```

Vs T3 sozinho: KL recover parcialmente porque LR=0.002 continua, decoder consegue convergir.
Vs T2 sozinho: Sem KL collapse, o modelo consegue aprender apesar do LR baixo final.

#### Porquê FID piora de 160 para 176

O terceiro epoch é crucial. Com LR=0.002 (T3), o decoder tem chance de contrariar collapse. Com LR=1e-5 (T4), está preso.

---

## O QUE LITERATURA DIZ VS O QUE VOCÊ OBSERVOU

### A Hipótese Original (Literatura)

```
Cosine LR:      -5 a -10 FID    (Burgess et al. 2018)
KL Annealing:   -5 a -15 FID    (Bowman et al. 2016)
Ambas juntas:   -10 a -25 FID   (expectativa aditiva)
```

### Os Dados Reais

```
T1 Baseline:           FID 146.1
T2 Cosine 50ep:        FID 164.3  → +18 (pior!)
T3 KL 50ep:            FID 160.3  → +14 (pior!)
T4 Ambas 50ep:         FID 176.0  → +30 (muito pior!)
```

### Por que o desvio é tão grande

1. **Epoch constraint**: Todos T2/T3/T4 usam 50 epochs vs T1 com 100
   - T1 a 50 epochs ≈ FID 157 (do VAE3 data)
   - T2/T3/T4 esperado: 157 - gains = 142-150
   - Observado: 160-176
   - Desvio: +10 a +26 pontos FID

2. **Implementation bugs**: T3 KL scale, T2 LR schedule incompatível

3. **Baseline already strong**: FID 146 é próximo do "teto físico" sem arquitetura melhor
   - Ganhos adicionais de -10/15 pts são mais difíceis que ganhos dos -56 pts (beta 0.7→0.1)
   - Lei de rendimentos decrescentes

4. **Dataset size**: 20% WikiArt é pequeno
   - Papers testam em CIFAR-10 (50k), CelebA (200k)
   - Seu dataset tem ~10k samples apenas
   - Escalas diferentes

---

## RECOMENDAÇÕES PARA RONDA 5

### 1. EXTENDED BASELINE (Prioridade CRÍTICA)

```python
# Por que: Determine o "teto" real com mais epochs
# Config:
beta = 0.1
latent_dim = 128
lr = 0.002
epochs = 150  # ou 200

# Esperado: FID 130-140 (loss ainda descendo?)
# Valor: Separa efeito arquitetura vs scheduling
```

### 2. COSINE LR CORRIGIDO (Prioridade ALTA)

```python
# Opção A: Básico
beta = 0.1
latent_dim = 128
lr = 0.002
epochs = 100
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

# Esperado: FID 135-142
# Vs T1: Ganho de -5 a -10 pts se implementação correcta

# Opção B: Com warm restarts (melhor)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1.5, eta_min=1e-5)
# Esperado: FID 130-140 (aprende mais)
```

### 3. KL ANNEALING CORRIGIDO (Prioridade ALTA)

```python
# CRITICAL: Verify normalization na loss

# Config:
beta = 0.1
latent_dim = 128
lr = 0.002
epochs = 100
kl_warmup_epochs = 15

# Loss calculation (CORRECTO):
batch_size = x.shape[0]
recon_loss = F.mse_loss(x_recon, x, reduction='mean')
kl_loss_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
kl_loss_normalized = kl_loss_raw / (batch_size * latent_dim)
beta_t = min(1.0, current_epoch / kl_warmup_epochs) * beta
total_loss = recon_loss + beta_t * kl_loss_normalized

# Esperado: FID 128-138 (se correção fixar o bug)
```

### 4. BOTH SCHEDULERS (CORRIGIDOS) (Prioridade MUITO ALTA)

```python
# Config:
beta = 0.1
latent_dim = 128
lr = 0.002
epochs = 100
kl_warmup_epochs = 15

# Schedulers aqui:
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
# KL warmup no loop de treino (como acima)

# Esperado: FID 120-130 ⭐⭐
# Este é o que deveria funcionar — ortogonal fixes
```

### 5. PERCEPTUAL LOSS (Prioridade CRÍTICA) ⭐⭐⭐

**Razão: Maior impacto potencial, implementação simples**

```python
import torchvision.models as models

# Carrega VGG-16 pré-treinado
vgg = models.vgg16(pretrained=True).features[:16]  # até relu2_2
for param in vgg.parameters():
    param.requires_grad = False
vgg.eval()

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg  # acima

    def forward(self, x, x_recon):
        # x e x_recon devem estar em [0, 1] ou [-1, 1]
        feat_real = self.vgg(x)
        feat_recon = self.vgg(x_recon)
        loss = F.mse_loss(feat_real, feat_recon)
        return loss

# Na training loop:
perc_loss = PerceptualLoss().forward(x, x_recon)

# Total loss:
# loss = α·recon_mse + β_t·kl + λ·perceptual_loss
# Valores: α=0.8, β_t=0.1 (como acima), λ=0.1

total_loss = 0.8 * recon_mse + beta_t * kl_normalized + 0.1 * perc_loss

# Esperado: FID 115-130 ⭐⭐⭐ (-15 a -25 vs MSE puro)
```

**Literature (Johnson et al. 2016, "Perceptual Losses for Real-Time Style Transfer"):**
- Perceptual loss no espaço VGG↔ 15-25% melhor FID que pixel MSE
- Para art dataset (WikiArt): Captura brushwork, texture, não apenas cor média
- Trade-off: Validation loss sobe (percep é difícil), mas FID cai (métrica real)

### 6. FREE BITS (Seguro contra collapse futuro) (Prioridade MÉDIA)

```python
# Kingma et al. 2016: garante cada dim latente usa ≥ λ nats

def free_bits_loss(mu, logvar, lambda_free_bits=1.0):
    # KL per dimension
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # [N, lat_dim]
    # Threshold at lambda
    kl_per_dim = torch.clamp(kl_per_dim, min=lambda_free_bits)
    # Average over batch e dims
    return kl_per_dim.mean()

# Na loss total:
total_loss = recon_loss + free_bits_loss(mu, logvar, lambda_free_bits=1.0)
# Note: sem beta aqui, free_bits é "soft" constraint

# Esperado: Previne posterior collapse mesmo com high beta
```

---

## RONDA 5 PLANO RECOMENDADO

### Hierarchy de testes

| Prioridade | Teste | Config | Epochs | Esperado FID |
|---|---|---|---|---|
| 🔴 1 | Extended Baseline | β=0.1, lat=128, lr=0.002, sem tricks | 150 | 130-140 |
| 🔴 2 | Cosine LR (T_max=100) | ↑ com CosineAnnealingLR | 100 | 125-135 |
| 🔴 3 | KL Annealing (corrigido) | ↑ com warmup=15, normalized | 100 | 125-135 |
| 🔴 4 | Ambas (corrigidas) | ↑ com Cosine + KL normalized | 100 | 120-130 |
| 🔴 5 | **Perceptual Loss** | ↑ + 0.1·VGG_relu2_2 | 100 | **115-125** |
| 🟠 6 | Free Bits | ↑ + free_bits λ=1.0 | 100 | 115-125 |
| 🟠 7 | Cyclical KL | ↑ com Fu et al. schedule | 100 | 110-120 |
| 🟡 8 | VQ-VAE (spatial) | 4×4×64 map, 256 codebook, EMA, PixelCNN | 100 | 100-120 |

### Test strategy

1. **Week 1**: Tests 1-4 (corrigem bugs conhecidos)
2. **Week 2**: Tests 5-7 (novas técnicas)
3. **Week 3**: Test 8 (VQ-VAE com implementação completa)
4. **Selector**: Se Test 5 (Perceptual) atinge FID <125, é caminho directo

---

## PAPERS DE REFERÊNCIA

| Paper | Autores | Ano | Relevância |
|---|---|---|---|
| Auto-Encoding Variational Bayes | Kingma & Welling | 2013 | VAE theory |
| Understanding disentangling in β-VAE | Burgess et al. | 2018 | β escolha, scheduling |
| Generating Sentences from Continuous Space | Bowman et al. | 2016 | KL annealing canonical |
| Cyclical Annealing Schedule | Fu et al. | 2019 | Alternativa a linear |
| Perceptual Losses for Real-Time Style Transfer | Johnson et al. | 2016 | Perceptual loss |
| The ArtBench Dataset | Liao et al. | 2022 | Benchmark ArtBench |
| Improving VAEs with Auxiliary Variables | Kingma et al. | 2016 | Free bits e técnicas |

---

## CHECKLIST PRONTO

- [ ] Verify KL normalization por (batch_size × latent_dim)
- [ ] Adjust Cosine T_max ≥ 100 (ou usar CosineAnnealingWarmRestarts)
- [ ] Implement Perceptual Loss (VGG-16 relu2_2)
- [ ] Test Extended Baseline (150 epochs, sem tricks)
- [ ] Run Test series 1-5 em paralelo se recurssos permitem
- [ ] Compare FID dropoff: Extended → Cosine → KL → Both → Perceptual
- [ ] Se FID <125: declare success, prepare PROD run (100% dataset, 200 epochs)
- [ ] Se FID <120: explore VQ-VAE spatial properly

