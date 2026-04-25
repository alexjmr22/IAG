# 🎯 RONDA 5 — CHEAT SHEET (1 página)

---

## RESUMO DOS 8 TESTES

| T | Nome | Técnica | Tempo | FID Esperado | ROI |
|---|---|---|---|---|---|
| **1** | Baseline 150ep | Mais epochs | 4-5h | 130-140 | ⭐ Baseline |
| **2** | Cosine LR | Scheduler | 2-3h | 135-142 | ⭐⭐ |
| **3** | KL Annealing | Warmup | 2-3h | 128-138 | ⭐⭐ |
| **4** | Cosine+KL | Combo | 2-3h | 120-130 | ⭐⭐⭐ |
| **5** | Perceptual Loss | Loss fn | 2-3h | **115-125** | ⭐⭐⭐⭐ |
| **6** | Free Bits | Segurança | 2-3h | 125-135 | ⭐⭐ |
| **7** | Cyclical KL | Alt KL | 2-3h | 120-130 | ⭐⭐ |
| **8** | All Together | Combo | 2-3h | 110-120? | ⭐ (incerto) |

---

## ESCOLHA RÁPIDO

```
Tempo          Cenário        Testes    Código              Linha Comando
─────────────────────────────────────────────────────────────────────
4-6h           QUICK          T1, T5    8_quick             --pc 8_quick
8-12h          COMPREHENSIVE T1-T4     8_comprehensive     --pc 8_comprehensive
12-18h         ALL IN         T1,T4-T8  8_all_in            --pc 8_all_in
20+h           FULL           T1-T8     8_full              --pc 8_full
```

---

## 3 PASSOS

### 1️⃣ COPIAR CÓDIGO
- Abra [RONDA5_COPIAR_COLAR.md](RONDA5_COPIAR_COLAR.md)
- Copie o código da opção escolhida
- Cola em `scripts/run_experiments.py` (no dicionário `EXPERIMENTS`)

### 2️⃣ ACTUALIZAR PARSER
```python
# Em run_experiments.py, mude:
choices=['1', '2', '3', '4', '5', '6', '7']
# Para:
choices=['1', '2', '3', '4', '5', '6', '7', '8_quick']  # ou outra opção
```

### 3️⃣ EXECUTAR
```bash
cd /Users/duartepereira/IAG
python3 scripts/run_experiments.py --pc 8_quick  # ou 8_comprehensive, 8_all_in, 8_full
```

---

## PRÉ-REQUISITOS

Antes de executar, implementou isto em `01_vae.py`?

```
✓ KL Normalization:  kl_loss / (batch_size * latent_dim)
✓ Cosine T_max:      CosineAnnealingLR(T_max=100, não 50!)
✓ KL Warmup:         beta_t = (epoch / warmup_epochs) * beta   (se T3+)
✓ Perceptual Loss:   0.8*MSE + 0.1*VGG_relu2_2 + 0.1*KL        (se T5+)
✓ Free Bits:         max(kl_per_dim, 1.0)                      (se T6+)
✓ Cyclical KL:       Beta sobe/desce em ciclos                 (se T7+)
```

Ver: [RONDA5_CODIGO_PRONTO.md](RONDA5_CODIGO_PRONTO.md) para implementação

---

## RESULTADO ESPERADO

### QUICK (4-6h)
```
T1 (Baseline):        FID 130-140
T5 (Perceptual):      FID 115-125
Ganho:                -15 a -25 pts ✅
```

### COMPREHENSIVE (8-12h)
```
T1 (Baseline):        FID 130-140
T4 (Cosine+KL):       FID 120-130
Ganho:                -10 a -20 pts ✅
```

### ALL IN (12-18h)  ← RECOMENDADO
```
T1 (Baseline):        FID 130-140
T5 (Perceptual):      FID 115-125  ← Provavelmente melhor
T8 (All):             FID 110-120? (ajusta depois)
Winner:               T5 ou T8
```

---

## SUCESSO = FID < 125

```
FID < 120  →  🏆 Excelente! Implementar em PROD (100% dataset)
120 < FID < 125 → ✅ Bom, suficiente para PROD
125 < FID < 130 → ⚠️ OK, mas investigar mais
FID > 130 → ❌ Algo errado, rever bugs
```

---

## FICHEIROS PARA CONSULTAR

| O que preciso? | Ficheiro |
|---|---|
| Decidir qual teste | [RONDA5_QUICK_DECISION.md](RONDA5_QUICK_DECISION.md) |
| Detalhe de cada teste | [RONDA5_TESTES_RECOMENDADOS.md](RONDA5_TESTES_RECOMENDADOS.md) |
| Código para colar | [RONDA5_COPIAR_COLAR.md](RONDA5_COPIAR_COLAR.md) |
| Implementação 01_vae.py | [RONDA5_CODIGO_PRONTO.md](RONDA5_CODIGO_PRONTO.md) |
| Diagnóstico R4 | [RONDA4_DIAGNOSTICO_DETALHADO.md](RONDA4_DIAGNOSTICO_DETALHADO.md) |
| Validação com literatura | [RONDA5_VALIDACAO_LITERATURA.md](RONDA5_VALIDACAO_LITERATURA.md) |

---

## TIMELINE

```
Hoje:        Implementar fixes + cópiar código (1-2h)
Amanhã:      Rodar QUICK ou COMPREHENSIVE (4-12h)
Dia 3:       Analisar resultados, escolher winner
Dia 4+:      Implementar winner em PROD (100% dataset)
```

---

## PERGUNTA FINAL

**Qual opção escolhes?**

- [ ] QUICK (4-6h) — Valida maior ganho
- [ ] COMPREHENSIVE (8-12h) — Entende schedulers
- [ ] ALL IN (12-18h) — Encontra combo optimal ← **RECOMENDADO**
- [ ] FULL (20+h) — Tudo para paper

Se não tens claro, escolhe **ALL IN** (melhor custo-benefício).

---

## GO! 🚀

```bash
# 1. Copia código de RONDA5_COPIAR_COLAR.md
# 2. Actualiza run_experiments.py
# 3. Roda:
python3 scripts/run_experiments.py --pc 8_all_in
# 4. Espera...
# 5. Celebra! 🎉
```

