# Ronda 5 — DECISION TREE (Escolhe em 30 segundos)

---

## 🎯 RESPONDA RÁPIDO

### Tempo total que tens para testes?

```
⏱️  4-6 horas (aprox 1 dia de CPU)      → Salta para QUICK START
⏱️  8-12 horas (2-3 dias de CPU)        → Salta para COMPRENSIVE
⏱️  16+ horas (vários dias de CPU)      → Salta para ALL IN
```

---

## QUICK START (4-6 horas CPU)

**Objetivo:** Validar o maior ganho possível rapidamente

### Testes a executar:
```
✓ T1: Extended Baseline 150ep      (4-5h) → FID 130-140
✓ T5: Perceptual Loss              (2-3h) → FID 115-125
```

### Código para adicionar a run_experiments.py:
```python
'8_quick': [
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
]
```

### Executar:
```bash
python3 scripts/run_experiments.py --pc 8_quick
```

### Esperado:
```
T1: FID 130-140 (teto puro, sem tricks)
T5: FID 115-125 (com perceptual loss)
Ganho: -15 a -25 pontos FID
Status: ✅ Se T5 < 125, é sucesso
```

**Decisão pós-resultado:**
- Se T5 < 125: Implementar T5 em PROD (100% dataset)
- Se T5 ≥ 125: Tentar COMPREHENSIVE para encontrar combo melhor

---

## COMPREHENSIVE (8-12 horas CPU)

**Objetivo:** Entender contribuição de cada técnica

### Testes a executar:
```
✓ T1: Extended Baseline            (4-5h) → FID 130-140
✓ T2: Cosine LR                    (2-3h) → FID 135-142
✓ T3: KL Annealing                 (2-3h) → FID 128-138
✓ T4: Cosine + KL (Both)           (2-3h) → FID 120-130
```

### Código:
```python
'8_comprehensive': [
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    
    {'id': 'ronda5_t2_cosine_lr', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true'}},
    
    {'id': 'ronda5_t3_kl_annealing_fixed', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
]
```

### Executar:
```bash
python3 scripts/run_experiments.py --pc 8_comprehensive
```

### Esperado:
```
T1: 130-140  (baseline)
T2: 135-142  (Cosine ajuda? -5 a +7)
T3: 128-138  (KL ajuda? -2 a +10)
T4: 120-130  (Combo ajuda? -10 a -20) ← Deve ser melhor que T1
```

**Análise:**
- Se T4 < T1 - 10: Schedulers funcionam, boa base para PROD
- Se T4 > T1: Schedulers não funcionaram, investigar bugs
- Se algum singular piora vs T1: Antagoismo

---

## ALL IN (12-18 horas CPU)

**Objetivo:** Encontrar combinação óptima, paper-ready

### Testes a executar:
```
✓ T1: Extended Baseline            (4-5h)
✓ T4: Cosine + KL (Best scheduler) (2-3h)
✓ T5: Perceptual Loss              (2-3h)
✓ T6: Free Bits                    (2-3h)
✓ T7: Cyclical KL                  (2-3h)
✓ T8: All Techniques               (2-3h)
```

### Código:
```python
'8_all_in': [
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
    
    {'id': 'ronda5_t6_free_bits', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_FREE_BITS': '1.0'}},
    
    {'id': 'ronda5_t7_cyclical_kl', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_CYCLICAL_KL': 'true'}},
    
    {'id': 'ronda5_t8_all_techniques', 'target': 'VAE', 
     'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15', 'VAE_PERCEPTUAL_LOSS': '0.1', 'VAE_FREE_BITS': '1.0'}},
]
```

### Executar:
```bash
python3 scripts/run_experiments.py --pc 8_all_in
```

### Esperado:
```
T1: 130-140  (baseline)
T4: 120-130  (scheduler combo)
T5: 115-125  (perceptual)
T6: 125-135  (free bits)
T7: 120-130  (cyclical KL)
T8: 110-120? (all together — incerto)

Winner: Provavelmente T5 ou T6
```

---

## ULTRA COMPREHENSIVE (16+ horas = TUDO)

Apenas se quer data completa para paper/publicação.

```python
'8_all': [
    # Use a config completa de RONDA5_TESTES_RECOMENDADOS.md
    # Todos os T1-T8
]
```

---

## ANTES DE EXECUTAR: CHECKLIST

Antes de rodar qualquer teste, garanta que tem implementado:

```
□ SETUP FIXES (obrigatório):
  □ Verificou KL normalization?    (Parte 1 RONDA5_CODIGO_PRONTO.md)
  □ Fixed Cosine T_max = 100?       (Parte 3)
  □ Adicionou VAE_KL_ANNEALING_EPOCHS ao env cleanup? (run_experiments.py)

□ T5 PERCEPTUAL LOSS (se quer T5):
  □ Implementou perceptual_loss.py ou integrou em 01_vae.py? (Parte 2)
  □ Adicionou VAE_PERCEPTUAL_LOSS ao env cleanup?

□ T6 FREE BITS (se quer T6):
  □ Implementou free_bits logic em 01_vae.py?
  □ Adicionou VAE_FREE_BITS ao env cleanup?

□ T7 CYCLICAL (se quer T7):
  □ Implementou cyclical KL schedule?
  □ Adicionou VAE_CYCLICAL_KL ao env cleanup?
```

---

## TABELA RÁPIDA: O QUE ESPERAR

| Cenário | Testes | Tempo | FID Esperado | Winner |
|---|---|---|---|---|
| **QUICK** | T1, T5 | 4-6h | T1:130-140, T5:115-125 | T5 (-15 pts) |
| **COMPREHENSIVE** | T1-T4 | 8-12h | T1:130-140, T4:120-130 | T4 (-10 pts) |
| **ALL IN** | T1,T4-T8 | 12-18h | T1:130-140, T8:110-120? | T5 ou T8 (-20+ pts) |
| **FULL** | T1-T8 | 16+ | Dados completos | Science! |

---

## RECOMENDAÇÃO FINAL

### Se tens <6 horas → **QUICK (T1+T5)**
- Maior ROI por tempo
- Valida maior ganho possível
- Suficiente para decisão PROD

### Se tens 8-12 horas → **COMPREHENSIVE (T1-T4)**
- Entende schedulers em detalhe
- Publication-quality ablation
- Fornece baseline para comparação futura

### Se tens 12-18 horas → **ALL IN (T1,T4-T8)**
- Comparação quase completa
- Encontra combinação optimizada
- Pronto para paper

### Se tens 20+ horas → **FULL (T1-T8)**
- Todos os dados
- Estudar interações
- Publicação de qualidade

---

## EXECUTE AGORA

Escolhe uma das linhas abaixo e copia para terminal:

```bash
# QUICK START
python3 /Users/duartepereira/IAG/scripts/run_experiments.py --pc 8_quick

# COMPREHENSIVE
python3 /Users/duartepereira/IAG/scripts/run_experiments.py --pc 8_comprehensive

# ALL IN
python3 /Users/duartepereira/IAG/scripts/run_experiments.py --pc 8_all_in
```

**Dica:** Primeiro adiciona à `run_experiments.py` o dictionary escolhido, depois executa.

