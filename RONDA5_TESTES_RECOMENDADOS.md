# Ronda 5 — Todos os Testes Recomendados

> Tabela completa com prioridades e configurações prontas para copiar

---

## TABELA RESUMIDA: SELECIONE OS QUE QUER

| # | Nome do Teste | Prioridade | ROI | Tempo CPU* | FID Esperado | Custo Computacional | Status |
|---|---|---|---|---|---|---|---|
| **T1** | Extended Baseline 150ep | 🔴 CRÍTICA | ⭐ | 4-5h | **130-140** | Médio | Determina teto |
| **T2** | Cosine LR (T_max=100) | 🟠 ALTA | ⭐⭐ | 2-3h | 135-142 | Baixo | Valida scheduler |
| **T3** | KL Annealing (Fixed) | 🟠 ALTA | ⭐⭐ | 2-3h | 128-138 | Baixo | Previne collapse |
| **T4** | Cosine + KL (Both) | 🔴 CRÍTICA | ⭐⭐⭐ | 2-3h | **120-130** | Baixo | Combina efeitos |
| **T5** | Perceptual Loss | 🔴 CRÍTICA | ⭐⭐⭐⭐ | 2-3h | **115-125** | Médio | 🏆 Maior impacto |
| **T6** | Free Bits | 🟡 MÉDIA | ⭐⭐ | 2-3h | 125-135 | Baixo | Seguro collapse |
| **T7** | Cyclical KL Annealing | 🟡 MÉDIA | ⭐⭐ | 2-3h | 120-130 | Baixo | Alternativa T3 |
| **T8** | All Techniques (Combo) | 🟡 MÉDIA | ⭐ | 2-3h | 110-120? | Médio | Investigação |

*Tempo de wall clock em Mac M1/M2 com RUN_PROFILE=DEV

---

## RECOMENDAÇÕES POR CENÁRIO

### 📊 Cenário A: Quero apenas QUICK gains (1-2 testes)
**Tempo total: 4-6 horas**
```
✓ T1 Extended Baseline (determina se mais epochs ajuda)
✓ T5 Perceptual Loss (maior ROI isolado)
```
**Expected:** T1 = 130-140, T5 = 115-125 (melhoria de -15 a -19 pts vs T1)

---

### 📊 Cenário B: Quero entender os schedulers (3-4 testes)
**Tempo total: 8-12 horas**
```
✓ T1 Extended Baseline (baseline)
✓ T2 Cosine LR (scheduler simples)
✓ T3 KL Annealing (warm-up simples)
✓ T4 Cosine + KL (combo)
```
**Expected:** Validar teoria de literatura. T4 deve ser ~10 pts melhor que T1.

---

### 📊 Cenário C: Quero o MÁXIMO de ganho (5-6 testes)
**Tempo total: 12-18 horas**
```
✓ T1 Extended Baseline
✓ T4 Cosine + KL (Best puro scheduler)
✓ T5 Perceptual Loss (Best loss function)
✓ T6 Free Bits (Segurança contra collapse)
✓ T7 Cyclical KL (vs linear KL)
✓ T8 All Techniques (Melhor possível?)
```
**Expected:** Encontrar combinação ótima. T8 esperado < 120 FID.

---

### 📊 Cenário D: Quero TUDO (todas 8)
**Tempo total: 16-24 horas**
```
✓ Todos T1-T8
```
**Expected:** Comparação completa, paper-ready results.

---

## CONFIGURAÇÕES PRONTAS PARA COPIAR

Abaixo estão as configurações **exatas** para adicionar ao `run_experiments.py`:

---

## PC 8 — RONDA 5 COMPLETA (Todos os 8 testes)

Copie esta secção inteira para o dicionário `EXPERIMENTS`:

```python
'8': [ # PC 8 — Ronda 5: Técnicas avançadas (β=0.1, lat=128, lr=0.002)
    # ==================== BASELINE ====================
    # T1: Extended Baseline 150 epochs (determina teto absoluto)
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    
    # ==================== SCHEDULERS ====================
    # T2: Cosine LR Scheduling (T_max=100, não 50!)
    {'id': 'ronda5_t2_cosine_lr', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true'}},
    
    # T3: KL Annealing (warmup=15 epochs, normalized)
    {'id': 'ronda5_t3_kl_annealing_fixed', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    
    # T4: Cosine + KL Annealing (Both, 100 epochs)
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    
    # ==================== LOSS FUNCTIONS ====================
    # T5: Perceptual Loss (0.8*MSE + 0.1*VGG + 0.1*KL)
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
    
    # ==================== COLLAPSE PREVENTION ====================
    # T6: Free Bits (per-dimension KL ≥ 1.0)
    {'id': 'ronda5_t6_free_bits', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_FREE_BITS': '1.0'}},
    
    # T7: Cyclical KL Annealing (Fu et al. 2019)
    {'id': 'ronda5_t7_cyclical_kl', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_CYCLICAL_KL': 'true'}},
    
    # ==================== COMBO (ALL) ====================
    # T8: All Techniques Combined (Cosine + KL + Perceptual + Free Bits)
    {'id': 'ronda5_t8_all_techniques', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15', 'VAE_PERCEPTUAL_LOSS': '0.1', 'VAE_FREE_BITS': '1.0'}},
]
```

---

## VARIANTES REDUZIDAS

### Se quer apenas **Quick Start** (Cenário A)

```python
'8_quick': [
    # T1: Extended Baseline
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    
    # T5: Perceptual Loss (maior ROI)
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
]
```

### Se quer **Entender Schedulers** (Cenário B)

```python
'8_schedulers': [
    # T1: Baseline
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    
    # T2: Cosine only
    {'id': 'ronda5_t2_cosine_lr', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true'}},
    
    # T3: KL only
    {'id': 'ronda5_t3_kl_annealing_fixed', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    
    # T4: Cosine + KL
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
]
```

### Se quer **Maximum Gains** (Cenário C)

```python
'8_maxgains': [
    # T1: Baseline
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    
    # T4: Best scheduler combo
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    
    # T5: Perceptual loss
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
    
    # T6: Free Bits
    {'id': 'ronda5_t6_free_bits', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_FREE_BITS': '1.0'}},
    
    # T7: Cyclical KL
    {'id': 'ronda5_t7_cyclical_kl', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_CYCLICAL_KL': 'true'}},
    
    # T8: All Techniques
    {'id': 'ronda5_t8_all_techniques', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15', 'VAE_PERCEPTUAL_LOSS': '0.1', 'VAE_FREE_BITS': '1.0'}},
]
```

---

## DETALHES DE CADA TESTE

### T1: Extended Baseline 150 epochs

**O quê:** Run mais longo do baseline para ver se loss continua a descer

**Config:**
```
Epochs: 150
Beta: 0.1
Latent dim: 128
LR: 0.002
Sem tricks adicionais
```

**Porquê:** 
- Ronda 4 T1 a 100 epochs tinha loss ainda descendo
- 150 epochs vai determinar teto "puro" sem técnicas

**Expected FID:** 130-140

**ROI:** ⭐ Baseline — é o baseline comparativo obrigatório

---

### T2: Cosine LR Scheduling

**O quê:** Learning rate decresce com função coseno durante treino

**Config:**
```
Epochs: 100
Cosine Annealing: T_max = 100 (CRÍTICO!)
eta_min = 1e-5
Resto igual a T1
```

**Porquê:**
- Ronda 4 T2 falhou porque T_max=50 (errado)
- Com T_max=100, LR desce suavemente 0.002 → 1e-5 ao longo de 100 epochs
- Literatura: -5 a -10 FID

**Expected FID:** 135-142

**ROI:** ⭐⭐ Valida scheduler

---

### T3: KL Annealing (Fixed)

**O quê:** KL weight sobe gradualmente 0 → 0.1 primeiras 15 epochs

**Config:**
```
Epochs: 100
KL Annealing warmup: 15 epochs
Normalization: MUST divide por (batch_size × latent_dim)
Resto igual a T1
```

**Porquê:**
- Ronda 4 T3 falhou por bug de scale (1e13)
- Com normalização correcta, previne posterior collapse
- Literatura: -5 a -15 FID

**Expected FID:** 128-138

**ROI:** ⭐⭐ Valida annealing

---

### T4: Cosine + KL Annealing (Both)

**O quê:** Combina T2 (Cosine LR) + T3 (KL warmup)

**Config:**
```
Epochs: 100
VAE_COSINE_LR: true (T_max=100)
VAE_KL_ANNEALING_EPOCHS: 15
Resto igual
```

**Porquê:**
- Efeitos são orthogonais (não se antagonizam)
- T2 resolve convergência final, T3 resolve collapse inicial
- Ronda 4 T4 falhou por bugs em T2 e T3, agora corrigidos

**Expected FID:** 120-130 ⭐⭐⭐

**ROI:** ⭐⭐⭐ Best scheduler combo

---

### T5: Perceptual Loss ⭐⭐⭐

**O quê:** Substitui MSE pixel-space por VGG feature-space loss

**Config:**
```
Epochs: 100
Loss: 0.8 × MSE + 0.1 × VGG_relu2_2 + 0.1 × KL
VAE_PERCEPTUAL_LOSS: 0.1 (weight)
Resto igual
```

**Porquê:**
- Johnson et al. 2016 mostra -15 a -25% melhor FID
- MSE sozinho produz blur, VGG captura texturas/brushwork
- Para art dataset (WikiArt) é especificamente bom

**Expected FID:** 115-125 ⭐⭐⭐⭐

**ROI:** ⭐⭐⭐⭐ MAIOR IMPACTO ISOLADO

**Nota:** Requer implementação em 01_vae.py (ver RONDA5_CODIGO_PRONTO.md)

---

### T6: Free Bits

**O quê:** Garante que cada dimensão latente usa ≥ λ nats de KL

**Config:**
```
Epochs: 100
VAE_FREE_BITS: 1.0
Resto igual
```

**Porquê:**
- Kingma et al. 2016 — previne posterior collapse
- Mais robusto que KL annealing sozinho
- Seguro adicional contra dropout de dimensões

**Expected FID:** 125-135

**ROI:** ⭐⭐ Medida de segurança

---

### T7: Cyclical KL Annealing

**O quê:** KL weight segue ciclos suave (sawtooth pattern)

**Config:**
```
Epochs: 100
VAE_CYCLICAL_KL: true
Num cycles: 4 (25 epochs cada)
Resto igual
```

**Porquê:**
- Fu et al. 2019 — melhor que linear annealing
- Re-expõe modelo a β=0 periodicamente para recuperação
- Previne collapse melhor que linear

**Expected FID:** 120-130

**ROI:** ⭐⭐ Alternativa interessante a T3

---

### T8: All Techniques Combined

**O quê:** Combina Cosine LR + KL (cyclical) + Perceptual Loss + Free Bits

**Config:**
```
Epochs: 100
VAE_COSINE_LR: true
VAE_KL_ANNEALING_EPOCHS: 15 (ou VAE_CYCLICAL_KL: true)
VAE_PERCEPTUAL_LOSS: 0.1
VAE_FREE_BITS: 1.0
```

**Porquê:**
- Investigação: efeitos cumulativos funcionam?
- Pode haver antagonismo entre técnicas
- Possível overfitting de tuning

**Expected FID:** 110-120?

**ROI:** ⭐ Investigação (incerto)

**Nota:** Ordem de prioridade é lower — testa depois de saber quais técnicas funcionam isoladas

---

## COMO EXECUTAR

### Opção 1: Máquina local (todos 8 testes em sequência)

```bash
cd /Users/duartepereira/IAG
python3 scripts/run_experiments.py --pc 8
```

Tempo total: 16-24 horas

### Opção 2: Quick start (T1 + T5 apenas)

```bash
cd /Users/duartepereira/IAG
python3 scripts/run_experiments.py --pc 8_quick
```

Tempo total: 4-6 horas

### Opção 3: Entender schedulers (T1-T4)

```bash
cd /Users/duartepereira/IAG
python3 scripts/run_experiments.py --pc 8_schedulers
```

Tempo total: 8-12 horas

### Opção 4: Máximo ganho (T1, T4-T8)

```bash
cd /Users/duartepereira/IAG
python3 scripts/run_experiments.py --pc 8_maxgains
```

Tempo total: 12-18 horas

---

## FERRAMENTAS NECESSÁRIAS (Se não implementadas ainda)

### Obrigatório para T5 (Perceptual Loss):
- Adicionar `perceptual_loss.py` ou integrar em `01_vae.py`
- Ver `RONDA5_CODIGO_PRONTO.md` Parte 2

### Obrigatório para corrigir T2-T4:
- Verificar KL normalization (Parte 1 de `RONDA5_CODIGO_PRONTO.md`)
- Fix Cosine T_max (Parte 3)

### Obrigatório para T6-T7:
- Implementar free bits logic
- Implementar cyclical KL logic
- (Ambos em `RONDA5_CODIGO_PRONTO.md`)

---

## CHECKLIST DE DECISÃO

Responda sim/não para cada:

```
□ Quero apenas 1-2 testes rápidos? → CENÁRIO A (T1+T5)
□ Quero entender schedulers em profundidade? → CENÁRIO B (T1-T4)
□ Tenho tempo e recursos para 5-6 testes? → CENÁRIO C (T1,T4-T8)
□ Quero ALL the data? → CENÁRIO D (Todos T1-T8)
□ Tenho implementado Perceptual Loss? □ Sim □ Não (necessário para T5)
□ Tenho tempo para esperar 20+ horas? □ Sim □ Não
□ Quero focar único teste de maior impacto? → T5 PERCEPTUAL LOSS
□ Quero comparação científica completa? → Todos T1-T8
```

---

## RESUMO EXECUTIVO: O QUE ESCOLHER

| Tempo disponível | Recomendação |
|---|---|
| **1-2 horas (setup)** | Implementar fixes + cópiar configs |
| **4-6 horas (CPU)** | Executar CENÁRIO A (T1+T5) |
| **8-12 horas (CPU)** | Executar CENÁRIO B (T1-T4) |
| **12-18 horas (CPU)** | Executar CENÁRIO C (T1,T4-T8) |
| **20+ horas (CPU)** | Executar CENÁRIO D (Todos T1-T8) |

**Recomendação padrão:** CENÁRIO C (máximo ganho, tempo razoável)

