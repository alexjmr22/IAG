# 🎯 RONDA 4: PLANO DE TESTES DETALHADO

## 1. RESUMO: O Que Vamos Testar

Base Fixa: **β=0.1, lat=128, lr=0.002, 20% dataset**

| # | Teste | Tipo | Epochs | Modificação | ROI | Tempo |
|---|-------|------|--------|-------------|-----|-------|
| **T1** | Cosine Annealing | Scheduler | 30 | CosineAnnealingLR | ⭐⭐⭐ | 2h |
| **T2** | KL Annealing | Beta Schedule | 30 | β: 0→0.1 em 10ep | ⭐⭐⭐ | 2h |
| **T3** | Both (Cosine + KL) | Dual | 30 | Ambos combinados | ⭐⭐⭐ | 2h |
| **T4** | 80 Epochs | Baseline | 80 | Só mais tempo | ⭐⭐ | 3h |
| **T5** | VQ-VAE | Architecture | 30 | Latent discreto | ⭐⭐⭐⭐ | 3h |

---

## 2. ANÁLISE INDIVIDUAL: Vale a Pena Testar?

### ✅ **T1: Cosine Annealing LR (CosineAnnealingLR)**

**O que é:**
```python
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
# LR vai de 0.002 em epoch 0 → 0.00001 em epoch 30
# Segue uma curva cosenoidal (meia volta de 0° a 180°)
```

**ROI: ⭐⭐⭐ (RECOMENDADO)**
- Impacto estimado: **-3 a -5 FID**
- Razão: Evita overshooting final (cuando o modelo está perto do ótimo)
- Risco: Muito baixo (técnica standard, já usada em billions de modelos)
- Tempo: ~2h
- Conclusão: **VALE MUITO A PENA**

**Porquê funciona:**
```
Cenário SEM scheduler (lr=0.002 fixo):
  Epoch 25: Loss=45, está perto do ótimo
  Epoch 26: Toma passo de 0.002 → oscila, sai do buraco!
  Epoch 27: Oscila pior ainda
  Epoch 30: Fica preso em mínimo local

Cenário COM CosineAnnealing:
  Epoch 25: Loss=45, lr ainda ≈0.0015
  Epoch 26: lr≈0.001, passo mais pequeno, fica no mínimo
  Epoch 27: lr≈0.0005, ainda mais pequeno
  Epoch 30: lr≈0.00001, pisadinha minúscula
  → Permanece no mínimo profundo!
```

---

### ✅ **T2: KL Annealing (β Schedule)**

**O que é:**
```python
# Epoch 0-9: β vai de 0 → 0.1 linearmente
# Epoch 10-30: β fica fixo em 0.1

def get_beta(epoch):
    if epoch < 10:
        return 0.1 * (epoch / 10)  # Sobe suavemente
    return 0.1
```

**ROI: ⭐⭐⭐ (MUITO RECOMENDADO)**
- Impacto estimado: **-5 a -8 FID**
- Razão: Evita KL collapse (encoder collapsa para Gaussian padrão sem aprender nada)
- Risco: Muito baixo (padrão industrial, usado em Transformer-based VAEs, GPT-VAE, etc.)
- Tempo: ~2h
- Conclusão: **EXTREMAMENTE VALE A PENA**

**Porquê funciona:**
```
Cenário SEM annealing (β=0.1 desde epoch 0):
  Epoch 0: Encoder está RANDOM
  Epoch 1: KL loss = 0.1 * KLD  ← Penalidade FORTE logo de início!
  Problema: Encoder é forçado a ficar perto de N(0,1) quando ainda não sabe o que fazer
  Resultado: KL collapse, encoder não aprende representações úteis
  
Cenário COM annealing:
  Epoch 0: β=0, KL loss = 0 (LIVRE para encoder aprender!)
  Epoch 1: β=0.01, pequena penalidade
  Epoch 5: β=0.05, penalidade média
  Epoch 10: β=0.1, penalidade full, mas encoder já tem boas reps
  Resultado: Encoder aprende bem, depois KL regulariza
  → Melhor taxa de reconstrução + bom latent space
```

---

### ✅ **T3: Combinado (Cosine + KL Annealing)**

**ROI: ⭐⭐⭐⭐ (MÁXIMA PRIORIDADE)**
- Impacto estimado: **-8 a -13 FID** (efeitos são ortogonais!)
- Racional: Ambos resolve problemas DIFERENTES:
  - KL Annealing: Evita collapse inicial (encoder aprende)
  - Cosine: Evita overshooting final (permanece no ótimo)
- Risco: Baixo (combinação de técnicas standard)
- Tempo: ~2h
- Conclusão: **CRÍTICO TESTAR — PODE SER O MELHOR RESULTADO**

**Sinergia:**
```
T1 (Cosine) resolve:    ┌─────────────────────┐
                        │ Convergência final  │
                        └─────────────────────┘

T2 (KL Ann) resolve:    ┌─────────────────────┐
                        │ Aprendizado inicial │
                        └─────────────────────┘

                        ✅ Não há sobreposição!
```

---

### ⭐⭐ **T4: 80 Epochs**

**ROI: ⭐⭐ (SEM SCHEDULER É SUBÓTIMO)**
- Impacto estimado: **-1 a -2 FID** (saturação obviamente)
- Racional: Apenas extensão timeput-wise
- Risco: Sem scheduler, vai oscilar entre epoch 50-80
- Tempo: ~3h
- Conclusão: **Vale testar DEPOIS de T1+T2+T3, não antes**

**Padrão esperado:**
```
Epochs  FID (sem scheduler)  FID (com CosinAnnealing)
──────────────────────────────────────────────────────
30      169.62              ~165 (estimado)
50      157.73              ~155 (estimado)
80      155-157 (oscila)    ~152 (steady)

Com scheduler continua melhorando; sem scheduler fica platô/oscila.
```

**Melhor praticar: Executar T4 COM scheduler (T1) já implementado!**

---

### 🔥 **T5: VQ-VAE (Vector Quantized VAE)**

**ROI: ⭐⭐⭐⭐ (Potencial MUITO ALTO)**
- Impacto estimado: **-15 a -25 FID** (mudança arquitetural)
- Racional: 
  - Latent space DISCRETO em vez de contínuo
  - Ótimo para arte (tem estrutura discreta: estilos, técnicas)
  - Melhor generação de estruturas (linhas, formas)
  - Melhor KID (melhor disentanglement)
- Risco: Médio (é outra arquitetura, precisa debug)
- Tempo: ~3h desenvolvimento + 2h treino = 5h
- Conclusão: **MUITO PROMISSOR, MAS REQUER IMPLEMENTAÇÃO**

**Por que melhor para Arte:**
```
β-VAE (Atual):
  ├─ Latent: Contínuo, 128-dim Gaussian
  ├─ Força: Suave, boa interpolação
  └─ Fraqueza: Pode gerar "blur" em estruturas, menos discreto

VQ-VAE:
  ├─ Latent: Discreto (256 símbolos x 128-dim)
  ├─ Força: Sharp, estruturado, sem blur
  ├─ Aprende "vocabulário" de padrões visuais
  └─ Melhor para arte (que TEM estrutura discreta!)
```

**Exemplo melhoria:**
```
β-VAE output: [0.2345, -0.1293, 0.5124, ...]  ← Contínuo, pode gerar transições fuzzy
VQ-VAE output: [5, 124, 32, ...]              ← Discreto, escolhe "código visual" exato
```

---

## 3. ROADMAP RECOMENDADO

### 🔴 **FASE 1: Quick Wins (Semana 1, ~6-8h total)**

```bash
# T1: Cosine Annealing
python3 scripts/01_vae.py \
  --exp-name vae_r4_beta01_lat128_lr2e3_cosine30 \
  --scheduler cosine \
  --epochs 30

# T2: KL Annealing
python3 scripts/01_vae.py \
  --exp-name vae_r4_beta01_lat128_lr2e3_kl30 \
  --kl-annealing 10 \
  --epochs 30

# T3: Ambos combinados ← MÁXIMA PRIORIDADE!
python3 scripts/01_vae.py \
  --exp-name vae_r4_beta01_lat128_lr2e3_both30 \
  --scheduler cosine \
  --kl-annealing 10 \
  --epochs 30
```

**Esperar resultados antes de continuar!**

---

### 🟡 **FASE 2: Extensions (Se T1+T2+T3 forem bem, ~5h)**

```bash
# T4: 80 Epochs com Cosine (reusa código de T1)
python3 scripts/01_vae.py \
  --exp-name vae_r4_beta01_lat128_lr2e3_cosine80 \
  --scheduler cosine \
  --epochs 80

# Só se T1 mostrar melhoria significativa! (>3 FID)
```

---

### 🟢 **FASE 3: Game-Changer (Se tiver MUITA vontade, ~8h)**

```bash
# T5: VQ-VAE (requer novo arquivo: vq_vae.py)
python3 scripts/vq_vae.py \
  --exp-name vq_vae_r4_30epochs \
  --epochs 30
```

---

## 4. IMPLEMENTAÇÃO: O QUE PRECISA MUDAR

### Alterações em `01_vae.py`:

#### A) Adicionar argumentos de CLI
```python
import argparse

parser.add_argument('--scheduler', choices=['none', 'cosine'], default='none')
parser.add_argument('--kl-annealing', type=int, default=0, 
                    help='Warmup epochs for KL annealing (0=disabled)')
```

#### B) KL Annealing
```python
def get_kl_beta(epoch, warmup_epochs, final_beta=0.1):
    if warmup_epochs == 0:
        return final_beta
    if epoch < warmup_epochs:
        return final_beta * (epoch / warmup_epochs)
    return final_beta

# No training loop:
for epoch in range(num_epochs):
    beta = get_kl_beta(epoch, args.kl_annealing, cfg.vae_beta)
    # E usar beta no cálculo do loss: loss = recon + beta * kld
```

#### C) Cosine Annealing LR
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

if args.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# No training loop:
for epoch in range(num_epochs):
    train(...)
    if args.scheduler == 'cosine':
        scheduler.step()
```

---

## 5. PREDIÇÃO: RANKING ESPERADO

```
Pos │ Configuração                          │ FID Estimado │ Nota
────┼───────────────────────────────────────┼──────────────┼──────────────────────
  1 │ T3: Cosine + KL, 30ep                 │ 152-158      │ ⭐⭐⭐ ESPERANÇA MÁXIMA
  2 │ T2: KL Annealing, 30ep                │ 161-165      │ ⭐⭐⭐ Muito promissor
  3 │ T1: Cosine, 30ep                      │ 165-168      │ ⭐⭐⭐ Bem sólido
  4 │ Baseline (sem changes), 30ep          │ 169.62       │ R3 TESTE 1
  5 │ T4: Cosine, 80ep                      │ 150-154      │ ⭐⭐ Se T1 good
  6 │ T5: VQ-VAE, 30ep                      │ 140-150(?)   │ 🔥 Tiro no escuro alto
```

---

## 6. A QUESTÃO: "E Outro Tipo de VAE?"

### Sim, definitivamente! Aqui estão as opções:

| Variante | Descrição | ROI | Tempo | Melhor Para |
|----------|-----------|-----|-------|------------|
| **VQ-VAE** | Latent discreto (256 tokens) | ⭐⭐⭐⭐ Alta | 5h | Arte, estrutura discreta |
| **VAE-GAN** | VAE + Discriminador | ⭐⭐⭐ Alta | 6h | FID otimizado |
| **Gumbel-VAE** | Latent categórico suave | ⭐⭐ Média | 4h | Estilos artísticos |
| **β-TCVAE** | β-VAE totalmente disentangled | ⭐⭐ Média | 2h | Análise fatores |
| **ψ-VAE** | Psi-VAE, outro regularizador | ⭐ Baixa | 3h | Pesquisa |

### 🏆 **RECOMENDAÇÃO: VQ-VAE**

**Por que:**
1. Latent space discreto = "vocabulário" de padrões
2. Ótimo para arte (tem estrutura, estilo, técnica → tudo discreto)
3. Historicamente dá **15-20% FID improvement** vs β-VAE
4. Código open-source disponível (não é do zero)
5. É o próximo passo natural após β-VAE

**Expectativa: FID 140-150** (vs atual 157.73 best)

---

## 7. PLANO FINAL RESUMIDO

```
SEMANA 1 (High ROI):
  ✅ T1: CosineAnnealingLR (30 epochs)
  ✅ T2: KL Annealing (30 epochs)
  ✅ T3: Ambos combinados (30 epochs) ← PRIORIDADE 1
  
  Tempo total: ~6h
  Impacto esperado: -8 a -13 FID (157.73 → 145-150)

SEMANA 2 (Confirmação):
  ✅ T4: 80 epochs (se T1 bom)
  
  Tempo total: ~3h
  Impacto esperado: -2 a -3 FID

SEMANA 3 (Game-Changer):
  🔥 T5: VQ-VAE (nova arquitetura)
  
  Tempo total: ~8h
  Impacto esperado: -15 a -25 FID (157.73 → 135-145)
```

---

## 8. RESPOSTAS DIRETAS

> **Achas que vale a pena testar a cena do cosine?**

**SIM!** ⭐⭐⭐ Impacto -3 a -5 FID, risco baixo, é técnica padrão.

> **E KL Annealing?**

**SIM!** ⭐⭐⭐⭐ Impacto -5 a -8 FID, muito promissor, padrão industrial.

> **E LR scheduling?**

CosineAnnealingLR = "LR scheduling". Já respondido acima. Se queres outro tipo (StepLR, ExponentialLR), CosineAnnealing é o mais científico para VAEs.

> **Com 80 épocas?**

**DEPOIS de testar T1+T2+T3**, para não desperdiçar tempo. Sem scheduler vai oscilar; com scheduler pode melhorar mais 2-3 FID.

> **Outro tipo de VAE?**

**SIM, VQ-VAE!** 🔥 É o próximo natural. Pode dar -15 a -25 FID. Mas é implementação nova, ~8h total.

---
