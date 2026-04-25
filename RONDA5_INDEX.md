# 📚 RONDA 5 — DOCUMENTAÇÃO COMPLETA (Index)

## Todos os Testes Recomendados

Documentei **8 testes** em 4 cenários:

### ⚡ RÁPIDO (4-6 horas)
- **T1**: Extended Baseline 150ep (FID 130-140)
- **T5**: Perceptual Loss (FID 115-125) ⭐ Melhor isolado

### 🔬 COMPREENSIVO (8-12 horas)
- **T1-T4**: Todos os schedulers isolados e combinados
- Valida: Cosine LR, KL annealing, efeito combinado

### 💪 MÁXIMO GANHO (12-18 horas) **← RECOMENDADO**
- **T1, T4, T5, T6, T7, T8**: Todos exceto T2-T3 individuais
- Encontra melhor combinação

### 🎓 COMPLETO (20+ horas)
- **T1-T8**: Tudo = data científica completa

---

## 📖 FICHEIROS CRIADOS (Lê nesta ordre)

### 1️⃣ **RONDA5_START_NOW.md** ← COMEÇA AQUI
```
├─ Fluxo de decisão visual
├─ 8 passos com checklist
├─ Timeline realista
└─ Potenciais problemas & fixes
```
**Tempo:** 5 minutos (ler) + 30 minutos (implementar)

---

### 2️⃣ **RONDA5_CHEATSHEET.md** ← Se tens pressa
```
├─ Tabela dos 8 testes (1 linha cada)
├─ Qual cenário escolher (4 opções)
├─ 3 passos (copiar → actualizar → executar)
└─ Resultado esperado vs FID real
```
**Tempo:** 2 minutos (ler)

---

### 3️⃣ **RONDA5_QUICK_DECISION.md** ← Se indeciso
```
├─ Decision tree por tempo disponível
├─ 4 cenários com configs prontas
├─ Código para cada (copiar-colar)
└─ Tabela rápida: O que esperar
```
**Tempo:** 5 minutos (decidir)

---

### 4️⃣ **RONDA5_COPIAR_COLAR.md** ← Se quer código
```
├─ 4 configs prontas (8_quick, comprehensive, all_in, full)
├─ Instruções de como modificar run_experiments.py
├─ Sintaxe exacta para argparse e cleanup
└─ Como executar cada variante
```
**Tempo:** 10-15 minutos (copiar e actualizar)

---

### 5️⃣ **RONDA5_TESTES_RECOMENDADOS.md** ← Se quer detalhe
```
├─ Tabela completa: 8 testes × prioridade × ROI
├─ Explicação detalhada de cada teste (T1-T8)
├─ Código de cada um
├─ Por quê (teoria) vs porquê não (bugs R4)
└─ Success criteria
```
**Tempo:** 20 minutos (ler completo)

---

### 6️⃣ **RONDA5_CODIGO_PRONTO.md** ← Se quer implementar
```
├─ Código pronto para cada técnica
├─ Perceptual Loss (copiar-colar)
├─ KL Normalization (1 linha fix)
├─ Cosine Scheduler (copiar-colar)
├─ KL Annealing (copiar-colar)
├─ Free Bits (copiar-colar)
├─ Cyclical KL (copiar-colar)
└─ Test suite script
```
**Tempo:** 1-2 horas (implementar tudo)

---

### 7️⃣ **RONDA5_VALIDACAO_LITERATURA.md** ← Se quer bases científicas
```
├─ Papers de referência + resultados
├─ Comparação literature vs seus dados
├─ Test matrix com predições
├─ Diagnostic checks (KL, active units, etc)
└─ Success criteria baseados em papers
```
**Tempo:** 30 minutos (ler)

---

### 8️⃣ **RONDA4_DIAGNOSTICO_DETALHADO.md** ← Se quer entender falhas
```
├─ Análise linha-a-linha de cada teste R4
├─ Os 3 bugs específicos (KL scale, Cosine T_max, etc)
├─ Porquê FID 176 (vs -25 esperado)
├─ Comparação com literatura (por quê desvio)
└─ Recomendações para R5
```
**Tempo:** 30 minutos (ler)

---

## 🎯 FLUXO RECOMENDADO

```
PASSO 1 (5 min):  Lê RONDA5_CHEATSHEET.md
                  ↓ Decide cenário (QUICK / COMPREHENSIVE / ALL IN / FULL)

PASSO 2 (10 min): Lê RONDA5_START_NOW.md
                  ↓ Segue o checklist

PASSO 3 (30 min): Implementa (se necessário)
                  ↓ Usa RONDA5_CODIGO_PRONTO.md

PASSO 4 (5 min):  Cópias config
                  ↓ Usa RONDA5_COPIAR_COLAR.md

PASSO 5 (imediato): Executa
                  ↓ python3 scripts/run_experiments.py --pc 8_all_in

PASSO 6 (18h):    Sistema roda sozinho
                  ↓ Vais para casa, dorme, volta amanhã

PASSO 7 (30 min): Analisa resultados
                  ↓ Escolhe winner

PASSO 8 (prep):   Implementa em PROD
```

---

## 📊 OS 8 TESTES (Resumo)

| # | Nome | FID Expect | Tempo | ROI | Pq funciona |
|---|---|---|---|---|---|
| 1 | Extended 150ep | 130-140 | 4-5h | ⭐ | Mais epochs = mais convergência |
| 2 | Cosine LR | 135-142 | 2-3h | ⭐⭐ | LR agenda suave melhora final |
| 3 | KL Annealing | 128-138 | 2-3h | ⭐⭐ | Beta warmup previne collapse |
| 4 | Cosine+KL | 120-130 | 2-3h | ⭐⭐⭐ | Combo efeitos orthogonais |
| 5 | Perceptual ⭐ | **115-125** | 2-3h | ⭐⭐⭐⭐ | VGG features > pixel MSE |
| 6 | Free Bits | 125-135 | 2-3h | ⭐⭐ | Per-dim KL ≥ 1.0 segurança |
| 7 | Cyclical KL | 120-130 | 2-3h | ⭐⭐ | β ciclos melhor que linear |
| 8 | All Combo | 110-120? | 2-3h | ⭐ | Investigação (incerto) |

---

## 🚀 COMEÇA AGORA (Copy-paste)

### Opção A: Quer ler tudo?
```bash
for f in /Users/duartepereira/IAG/RONDA5_*.md /Users/duartepereira/IAG/RONDA4_*.md; do
  echo "=== $(basename $f) ==="
  wc -l $f
done
# Total: ~3000 linhas de documentation 📚
```

### Opção B: Quer começar JÁ (10 min)?
```bash
# 1. Lê este (2 min)
cat /Users/duartepereira/IAG/RONDA5_CHEATSHEET.md

# 2. Lê isto (5 min)
cat /Users/duartepereira/IAG/RONDA5_QUICK_DECISION.md

# 3. Cópias código (3 min)
# Vai a RONDA5_COPIAR_COLAR.md, lê a variante escolhida

# 4. Executa
python3 /Users/duartepereira/IAG/scripts/run_experiments.py --pc 8_all_in
```

### Opção C: Quer tudo pronto mas não sabe como fazer?
```bash
# Lê isto em ordem:
1. RONDA5_START_NOW.md        (+ importante, guia passo-a-passo)
2. RONDA5_CODIGO_PRONTO.md    (+ importante, implementar)
3. RONDA5_COPIAR_COLAR.md     (+ importante, configs)
```

---

## ✅ CHECKLIST FINAL

Antes de executar `python3 run_experiments.py`:

```
☐ Leste RONDA5_START_NOW.md?
☐ Decidiste qual cenário (QUICK / COMPREHENSIVE / ALL IN / FULL)?
☐ Verificaste que 01_vae.py tem as implementations:
  ☐ KL normalization: kl_loss / (batch_size * latent_dim)?
  ☐ Cosine scheduler suportado (VAE_COSINE_LR env var)?
  ☐ KL annealing suportado (VAE_KL_ANNEALING_EPOCHS env var)?
  ☐ Perceptual loss suportado (VAE_PERCEPTUAL_LOSS env var)?
☐ Cópiaste config em scripts/run_experiments.py?
☐ Actualizaste argparse choices?
☐ Actualizaste _exp_vars cleanup?
☐ Testaste: python3 -m py_compile scripts/run_experiments.py?
☐ Pronto para: python3 scripts/run_experiments.py --pc 8_all_in?
```

---

## 🏆 RESULTADO ESPERADO

```
Se tudo correr bem:

  Ronda 4 melhor: FID 146.1 (T1 baseline 100ep)
  Ronda 5 esperado: FID 115-125 (T5 perceptual loss)
  Ganho: -21 a -31 pontos FID ✅
  
Status: SUCESSO → Implementar em PROD (100% dataset)
```

---

## 📞 SE TIVER DÚVIDAS

Consulta este Index:

| Pergunta | Ficheiro |
|---|---|
| Qual teste devo fazer? | RONDA5_CHEATSHEET.md |
| Como começo? | RONDA5_START_NOW.md |
| Indeciso entre opções? | RONDA5_QUICK_DECISION.md |
| Quer todos os detalhes? | RONDA5_TESTES_RECOMENDADOS.md |
| Quer código pronto? | RONDA5_COPIAR_COLAR.md |
| Quer implementar fixes? | RONDA5_CODIGO_PRONTO.md |
| Quer base científica? | RONDA5_VALIDACAO_LITERATURA.md |
| Quer entender R4? | RONDA4_DIAGNOSTICO_DETALHADO.md |

---

## 💡 ÚLTIMA DICA

**A MELHOR ABORDAGEM:**

1. Não gastes tempo a ler tudo de uma vez
2. Lê: RONDA5_CHEATSHEET.md (2 min)
3. Escolhe: QUICK ou ALL IN (depende do tempo)
4. Implementa: Segue RONDA5_START_NOW.md (30 min)
5. Executa: `--pc 8_all_in` (lança e vai dormir)
6. Analisa: Volta no dia seguinte (30 min)

**Total tempo teu: ~1 hora, distribuído em 2 dias**
**Tempo de máquina: 12-18 horas**

---

## 🎉 RESUMO: TÈM 8 TESTES

Organizados em 4 cenários por tempo disponível:

```
⏱️ 4-6h   (QUICK)        → T1, T5
⏱️ 8-12h  (COMPREHENSIVE) → T1-T4
⏱️ 12-18h (ALL IN)        → T1, T4-T8  ← Recomendado
⏱️ 20+h   (FULL)         → T1-T8
```

**Escolhe uma, segue o guia, e vê os resultados.**

Boa sorte! 🚀

