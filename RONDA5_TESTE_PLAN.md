# 🚀 RONDA 5: PLANO DE TESTES — SCHEDULER + ARQUTETURAS NOVAS

## 📋 Resumo Executivo

Base fixa para TODOS os testes: **β=0.1, lat=128, lr=0.002, 50 epochs, 20% dataset**

Vamos testar **5 configurações** em ordem de complexidade/impacto.

---

## 🧠 OS 5 TESTES

| # | Nome | Tipo | Scheduler | KL Ann | Arquivo | Tempo | ROI |
|---|------|------|-----------|--------|---------|-------|-----|
| **T1** | β-VAE Base | Baseline | ❌ | ❌ | `01_vae.py` | ~2h | ⭐⭐ |
| **T2** | β-VAE + Cosine | Scheduler | ✅ Cosine | ❌ | `01_vae.py` | ~2h | ⭐⭐⭐ |
| **T3** | β-VAE + KL Ann | Scheduler | ❌ | ✅ (10ep) | `01_vae.py` | ~2h | ⭐⭐⭐⭐ |
| **T4** | β-VAE + Ambos | Scheduler | ✅ Cosine | ✅ (10ep) | `01_vae.py` | ~2h | ⭐⭐⭐⭐ |
| **T5** | CVAE | Arquitetura | ✅ Cosine | ✅ (10ep) | `03_cvae.py` | ~2h | ⭐⭐⭐⭐⭐ |
| **T6** | VQ-VAE | Arquitetura | ✅ Cosine | ❌ (não aplica) | `04_vq_vae.py` | ~2h | ⭐⭐⭐⭐⭐ |

**Total: ~12 horas CPU** (ideal para correr em paralelo/background)

---

## 📊 COMANDOS DE EXECUÇÃO

Todos com: `β=0.1, lat=128, lr=0.002, 50 epochs, 20% dataset`

```bash
cd /Users/duartepereira/IAG

# T1: β-VAE Base (Baseline R3)
EXP_NAME=vae_r5_t1_baseline \
  VAE_BETA=0.1 VAE_LATENT_DIM=128 VAE_LR=2e-3 \
  python3 scripts/01_vae.py

# T2: β-VAE + Cosine Annealing
EXP_NAME=vae_r5_t2_cosine \
  VAE_BETA=0.1 VAE_LATENT_DIM=128 VAE_LR=2e-3 \
  VAE_COSINE_LR=true \
  python3 scripts/01_vae.py

# T3: β-VAE + KL Annealing
EXP_NAME=vae_r5_t3_kl_annealing \
  VAE_BETA=0.1 VAE_LATENT_DIM=128 VAE_LR=2e-3 \
  VAE_KL_ANNEALING_EPOCHS=10 \
  python3 scripts/01_vae.py

# T4: β-VAE + Cosine + KL Annealing (THE ONE!)
EXP_NAME=vae_r5_t4_both \
  VAE_BETA=0.1 VAE_LATENT_DIM=128 VAE_LR=2e-3 \
  VAE_COSINE_LR=true VAE_KL_ANNEALING_EPOCHS=10 \
  python3 scripts/01_vae.py

# T5: CVAE (Conditional VAE)
EXP_NAME=cvae_r5_t5_conditional \
  VAE_BETA=0.1 VAE_LATENT_DIM=128 VAE_LR=2e-3 \
  VAE_COSINE_LR=true VAE_KL_ANNEALING_EPOCHS=10 \
  python3 scripts/03_cvae.py

# T6: VQ-VAE (Vector Quantized)
EXP_NAME=vq_vae_r5_t6_quantized \
  VAE_LATENT_DIM=128 VAE_LR=2e-3 \
  VAE_COSINE_LR=true \
  python3 scripts/04_vq_vae.py
```

---

## 📈 PREDIÇÕES (FID esperado)

```
Ranking esperado:

Pos  │ Teste           │ FID Estimado │ Comentário
─────┼─────────────────┼──────────────┼─────────────────────────────
 1   │ VQ-VAE (T6)     │ 130-145      │ 🔥 Maior ganho potencial
 2   │ β-VAE Both (T4) │ 145-155      │ ⭐ Scheduler perfeito
 3   │ CVAE (T5)       │ 150-160      │ ✨ Controlo de classe
 4   │ KL Ann (T3)     │ 158-165      │ ⭐ Beta estratégico
 5   │ Cosine (T2)     │ 162-168      │ ⭐ LR refinement
 6   │ Base (T1)       │ 169.62       │ R3 Baseline
─────┴─────────────────┴──────────────┴─────────────────────────────
```

---

## 🎯 INTERPRETAÇÃO DOS RESULTADOS

### Se T4 (β-VAE Both) < 155 FID ✅
**Conclusão**: Schedulers funcionam extremamente bem!
→ Usar T4 config como base para tudo daqui em frente

### Se T5 (CVAE) < 160 FID ✅
**Conclusão**: Informação de classe ajuda!
→ CVAE é viável para geração condicionada

### Se T6 (VQ-VAE) < 140 FID ✅
**Conclusão**: Discretização é golden bullet!
→ VQ-VAE é verdadeiramente superior para arte

### Best Case Scenario:
```
T6 (VQ-VAE): FID 130-140
→ Representa ~20-25% melhoria vs R3 best (157.73)
→ Paper sólido, PROD-ready!
```

---

## 📝 PRÓXIMOS PASSOS APÓS TESTES

### Phase A: Confirmação (2-3h)
Se **T4 (Both) < 155 FID**, correr:
- Dataset 100% com T4 config
- Esperado: FID 125-135

### Phase B: Exploração (4-5h se T6 bom)
Se **T6 (VQ-VAE) < 140 FID**, correr:
- VQ-CVAE combo (discreto + condicional)
- Esperado: FID 120-130

### Phase C: PROD
Treinar melhor config com **100% dataset, 100 epochs, com schedulers**
- Esperado: FID 110-125

---

## 🔧 IMPLEMENTATION NOTES

### Ficheiros Modificados
- ✅ `01_vae.py`: Added `VAE_COSINE_LR`, `VAE_KL_ANNEALING_EPOCHS` flags
  - Default: sem schedulers (compatível backward)
  - Support: env vars ✅

### Ficheiros Novos
- ✅ `03_cvae.py`: ConditionalVAE (10 classes)
  - Encoder/Decoder recebem classe como input
  - Same loss como β-VAE
  
- ✅ `04_vq_vae.py`: VQVAE
  - VectorQuantizer layer (256 embeddings)
  - Commitment loss + reconstruction loss
  - No KL loss (discreto não precisa)

---

## 📋 ORDEM RECOMENDADA DE EXECUÇÃO

```
SEMANA 1: T1 + T2 + T3 + T4 (em paralelo, ~6-8h total CPU)
  └─ Benchmark β-VAE e validar schedulers

SEMANA 2: T5 + T6 (em paralelo, ~4-5h total CPU)
  └─ Explorar arquiteturas novas

SEMANA 3: Best configs com dataset 100%
  └─ PROD preparation
```

---

## ✅ Checklist de Validação

Antes de correr, verificar:
- [ ] `01_vae.py` tem flags de scheduler
- [ ] `03_cvae.py` criado com ConditionalVAE
- [ ] `04_vq_vae.py` criado com VQVAE
- [ ] Todos rescebem env vars corretos
- [ ] Dataset carrega sem errors
- [ ] Checkpoints gardados em `/results/EXP_NAME/`

---

## 🎓 SCIENTIFIC INSIGHTS

### T2 vs T1 (Cosine Effect)
```
Se ΔT2-T1 < 2 FID: Scheduler baixo impacto (esperado ~-3 a -5)
Se ΔT2-T1 ≥ 3 FID: Scheduler alto impacto (recomendado sempre!)
```

### T3 vs T1 (KL Annealing Effect)
```
Se ΔT3-T1 < 3 FID: KL annealing baixo impacto
Se ΔT3-T1 ≥ 5 FID: KL annealing alto impacto (recomendado!)
```

### T4 vs T2+T3 (Synergy)
```
Se ΔT4 > ΔT2 + ΔT3: Effects são sinérgicos! (ideal)
Se ΔT4 ≈ ΔT2 + ΔT3: Effects são aditivos (esperado)
Se ΔT4 < ΔT2 + ΔT3: Conflito entre schedulers (improvável)
```

### T5 vs T4 (Condicionamento)
```
Se ΔT5-T4 < 5 FID: Classe info não melhora FID mas melhora controlo
Se ΔT5-T4 ≥ 5 FID: Condicionamento ajuda FID também!
```

### T6 vs T4 (Vector Quantization)
```
Se ΔT6-T4 < 10 FID: VQ modesto improvement (5-15%)
Se ΔT6-T4 ≥ 15 FID: VQ transformador! (15-25%)
Se ΔT6-T4 < 0 (T6 pior): VQ não adequado para ArtBench (improvável)
```

---

