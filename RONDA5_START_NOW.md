# 📋 RONDA 5 — PRÓXIMOS PASSOS IMEDIATOS

> O QUE FAZER AGORA (Passo a Passo)

---

## FLUXO DE DECISÃO

```
INÍCIO
  │
  └─→ Qual o teu tempo disponível?
      │
      ├─→ < 6 horas         → QUICK (T1+T5)
      ├─→ 8-12 horas        → COMPREHENSIVE (T1-T4)
      ├─→ 12-18 horas       → ALL IN (T1,T4-T8) ← RECOMENDADO
      └─→ 20+ horas         → FULL (T1-T8)

  │
  └─→ Tens implementado em 01_vae.py:
      ├─→ KL Normalization?           □ Sim □ Não
      ├─→ Cosine Scheduler support?   □ Sim □ Não
      ├─→ KL Annealing support?       □ Sim □ Não
      ├─→ Perceptual Loss support?    □ Sim □ Não (necessário se T5+)
      └─→ Free Bits support?          □ Sim □ Não (necessário se T6+)

  │
  └─→ Se faltam implementations:
      1. Lê: RONDA5_CODIGO_PRONTO.md
      2. Copia os fixes para 01_vae.py
      3. Testa: python3 01_vae.py --help
      4. Verifica que env vars são aceitas

  │
  └─→ Cópiar config para run_experiments.py:
      1. Abre: scripts/run_experiments.py
      2. Lê: RONDA5_COPIAR_COLAR.md
      3. Copia a config escolhida (8_quick, 8_comprehensive, etc)
      4. Actualiza argparse choices
      5. Actualiza _exp_vars cleanup

  │
  └─→ Executar:
      python3 scripts/run_experiments.py --pc 8_quick  (ou outra)

  │
  └─→ Esperar + Analisa resultados
      ├─→ FID < 125?  → ✅ Sucesso! Prep PROD run
      └─→ FID > 125?  → ⚠️ Debug, ver RONDA4_DIAGNOSTICO.md
```

---

## CHECKLIST: FAZES ISTO AGORA?

### PASSO 1: DECIDIR (5 min)

```
□ Li o RONDA5_CHEATSHEET.md completo?
□ Escolhi o cenário (QUICK / COMPREHENSIVE / ALL IN / FULL)?
```

**Minha recomendação:** ALL IN (12-18h, máximo ganho, tempo razoável)

### PASSO 2: VERIFICAR IMPLEMENTAÇÕES (10-30 min)

Se tens dúvida se implementações estão prontas:

```bash
# Verifica se KL normalization está presente
grep -n "batch_size \* latent_dim" /Users/duartepereira/IAG/scripts/01_vae.py
# Deve retornar sim (se vazio, tem erro)

# Verifica se Cosine LR support existe
grep -n "CosineAnnealingLR" /Users/duartepereira/IAG/scripts/01_vae.py
# Deve retornar sim (se vazio, tem erro)

# Verifica se KL Annealing support existe
grep -n "VAE_KL_ANNEALING_EPOCHS" /Users/duartepereira/IAG/scripts/01_vae.py
# Deve retornar sim (se vazio, tem erro)
```

Se está tudo vazio ou com erro:
```
→ Lê: RONDA5_CODIGO_PRONTO.md
→ Implementa os fixes em 01_vae.py
→ Re-testa com comandos acima
```

### PASSO 3: ADICIONAR CONFIGS (10-15 min)

Edita `/Users/duartepereira/IAG/scripts/run_experiments.py`:

```python
# Linha 1: Acha a posição correta (após fechamento de '7')
# Procura: '7': [ ... ]

# Linha 2: Copia o código de RONDA5_COPIAR_COLAR.md
# (Escolhe a variante: 8_quick, 8_comprehensive, 8_all_in, ou 8_full)

# Linha 3: Cola no dicionário EXPERIMENTS
# Exemplo posição correta:
'7': [
    ...
],  # ← Fechamento de 7
'8_all_in': [  # ← Adicionas isto aqui
    {'id': 'ronda5_t1_baseline_150ep', ...},
    ...
],  # ← Fechamento de 8_all_in
```

### PASSO 4: ACTUALIZAR PARSER (5 min)

Na mesma file, procura:
```python
parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5', '6', '7'], ...)
```

Muda para:
```python
parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5', '6', '7', '8_all_in'], ...)
```

(Usa a variante escolhida: 8_quick, 8_comprehensive, 8_all_in, ou 8_full)

### PASSO 5: ACTUALIZAR CLEANUP (2 min)

Ainda em run_experiments.py, procura:
```python
_exp_vars = [
    'VAE_LATENT_DIM', 'VAE_BETA', 'VAE_LR', 'VAE_EPOCHS', 'VAE_COSINE_LR', 'VAE_KL_ANNEALING_EPOCHS',
    'DCGAN_LATENT', ...
]
```

Adiciona (se for usar T5+):
```python
_exp_vars = [
    'VAE_LATENT_DIM', 'VAE_BETA', 'VAE_LR', 'VAE_EPOCHS', 
    'VAE_COSINE_LR', 'VAE_KL_ANNEALING_EPOCHS',
    'VAE_PERCEPTUAL_LOSS', 'VAE_FREE_BITS', 'VAE_CYCLICAL_KL',  # ← ADICIONA
    'DCGAN_LATENT', ...
]
```

### PASSO 6: EXECUTAR (Início de teste!)

No terminal:
```bash
cd /Users/duartepereira/IAG

# Verifica sintaxe
python3 -m py_compile scripts/run_experiments.py

# Se OK, executa
python3 scripts/run_experiments.py --pc 8_all_in

# Senta-te e espera (12-18 horas de wall clock em M1/M2)
```

Monitor:
```bash
# Num outro terminal, verifica progresso
watch -n 60 'ls -ltr /Users/duartepereira/IAG/results/ | tail -5'
```

### PASSO 7: ANALISA RESULTADOS (30 min)

Após completar:

```bash
# Coleta FID de todos os testes
python3 << 'EOF'
import pandas as pd
from pathlib import Path

results = {}
for exp_dir in Path('/Users/duartepereira/IAG/results').glob('ronda5_*'):
    csv_file = exp_dir / 'results.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        fid = df[df['Metric'] == 'FID']['Value'].values[0] 
        results[exp_dir.name] = fid

# Sort by FID (melhor = menor)
print("\n" + "="*60)
print("  RONDA 5 RESULTS")
print("="*60)
for name, fid in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:40s}  FID = {fid:.1f}")
print("="*60 + "\n")
EOF
```

### PASSO 8: DECISÃO (5 min)

Baseado nos resultados:

```
Se FID < 120:
  ✅ MUITO BEM! Temos um vencedor.
  → Próximo: Implementar em PROD (100% dataset, 200+ epochs)

Se 120 ≤ FID < 125:
  ✅ BOM! Suficiente ganho.
  → Próximo: Preparar PROD run

Se 125 ≤ FID < 130:
  ⚠️ OK mas esperámos mais.
  → Próximo: Re-correr T5 isolado com mais epochs (150)
            Ou testar VQ-VAE com spatial maps

Se FID ≥ 130:
  ❌ Algo errado.
  → Próximo: Rever RONDA4_DIAGNOSTICO_DETALHADO.md
            Debug KL normalization
            Verificar se env vars aplicadas corretamente
```

---

## TIMELINE REALISTA

```
Hoje (Dia 1):
  └─ 9:00 - Lê documentação (30 min)
  └─ 9:30 - Implementa fixes em 01_vae.py (30-60 min)
  └─ 10:30 - Testa implementação (10 min)
  └─ 10:45 - Adiciona config + startup (15 min)
  └─ 11:00 - Inicia python3 run_experiments.py --pc 8_all_in
  └─ 11:05 - Training starts... (deixa correr)

Dia 2-3:
  └─ Processa: P3 a processar..., coffee ☕, beer 🍺

Dia 4 (Manhã):
  └─ 9:00 - Training completo
  └─ 9:30 - Analisa resultados (30 min)
  └─ 10:00 - Escolhe winner (5 min)
```

---

## POTENCIAIS PROBLEMAS & FIXES

### Problema 1: "ModuleNotFoundError: No module named 'torchvision'"
```
Fix: pip install torchvision
(ou verificar que está no venv correto)
```

### Problema 2: "CUDA out of memory"
```
Fix: Reduzir batch size em config.py
     BATCH_SIZE = 64 (em vez de 128)
     VAE_EPOCHS = 50 (em vez de 150)
```

### Problema 3: FID não melhora após primeiro teste
```
Fix: Verificar que env vars são realmente aplicadas
     python3 -c "import os; print(os.getenv('VAE_COSINE_LR'))"
     Se vazio, env var não foi passada correctamente
```

### Problema 4: KL fica 1e13 no primeiro epoch
```
Fix: Normalization bug ainda presente
     Verifica: grep -A2 "kl_loss" 01_vae.py | grep "batch_size"
     Deve ter a divisão
```

---

## SUCESSO SIGNS

✅ **Sabes que está a funcionar se:**

```
T1 baseline completa com FID 130-140
├─ KL no epoch 0: ~150-200 (não 1e13)
├─ Loss curva suave descenso
└─ Samples razoáveis (não blur severo)

T5 perceptual completa com FID 115-125
├─ Melhoria vs T1: -15 a -25 pontos
├─ Samples mais sharp vs T1
└─ Training 2x mais lento (por VGG forward pass)

T8 all_techniques completa
├─ Melhor ou igual a melhor single test
└─ Não há erro de NaN ou crashes
```

---

## DOCUMENTO REFERENCE RÁPIDO

| Preciso de... | Ficheiro |
|---|---|
| Entender o que fazer | Este documento |
| Decidir qual teste | RONDA5_CHEATSHEET.md |
| Decision tree | RONDA5_QUICK_DECISION.md |
| Código para colar | RONDA5_COPIAR_COLAR.md |
| Implementar fixes | RONDA5_CODIGO_PRONTO.md |
| Entender bugs R4 | RONDA4_DIAGNOSTICO_DETALHADO.md |
| Literatura | RONDA5_VALIDACAO_LITERATURA.md |

---

## RECOMENDAÇÃO FINAL

**Faz isto AGORA:**

1. Lê RONDA5_CHEATSHEET.md (2 min)
2. Cópias config de RONDA5_COPIAR_COLAR.md (5 min)
3. Implementa em run_experiments.py (10 min)
4. Executa: `python3 scripts/run_experiments.py --pc 8_all_in` (inicia treino)
5. Vai tomar café ☕ (18 horas de CPU tempo...)
6. Volta ao dia seguinte, analisa resultados

**Total de tempo teu: ~30 minutos hoje, 30 minutos no dia seguinte.**
**Tempo de CPU: 12-18 horas.**

---

## 🎯 GO! 

Coma isto para começar AGORA:

```bash
# 1. Lê os 2 docs principais (10 min total)
cat /Users/duartepereira/IAG/RONDA5_CHEATSHEET.md
cat /Users/duartepereira/IAG/RONDA5_QUICK_DECISION.md

# 2. Escolhe aqui qual quer fazer (5 min)
# (default recomendação: 8_all_in)

# 3. Cópias config (5 min)
# Ver RONDA5_COPIAR_COLAR.md

# 4. Inicia treino
python3 /Users/duartepereira/IAG/scripts/run_experiments.py --pc 8_all_in

# 5. Vê progresso
watch -n 60 'ls -ltr /Users/duartepereira/IAG/results/ronda5_* | tail -2'

# FIM. Sistema roda sozinho. Volta-se amanhã.
```

💪 Boa sorte! (vais conseguir!) 🚀

