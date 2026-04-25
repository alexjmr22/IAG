# Ronda 5 — Código Pronto para Copiar em run_experiments.py

> Copy-paste direto para o dicionário EXPERIMENTS

---

## OPÇÃO 1: QUICK START (2 testes, 4-6 horas)

Copie isto para `EXPERIMENTS` em `run_experiments.py`:

```python
'8_quick': [ # PC 8_quick — Ronda 5 Quick: Baseline vs Perceptual Loss
    # T1: Extended Baseline (determina teto)
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    # T5: Perceptual Loss (maior ROI isolado)
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
],
```

---

## OPÇÃO 2: COMPREHENSIVE (4 testes, 8-12 horas)

Copie isto para `EXPERIMENTS`:

```python
'8_comprehensive': [ # PC 8_comprehensive — Ronda 5: Entender schedulers (T1-T4)
    # T1: Extended Baseline 150 epochs
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    # T2: Cosine LR Scheduling (T_max=100, não 50!)
    {'id': 'ronda5_t2_cosine_lr', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true'}},
    # T3: KL Annealing (warmup=15, normalized)
    {'id': 'ronda5_t3_kl_annealing_fixed', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    # T4: Cosine + KL Annealing (Both)
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
],
```

---

## OPÇÃO 3: ALL IN (6 testes, 12-18 horas) ⭐ RECOMENDADO

Copie isto para `EXPERIMENTS`:

```python
'8_all_in': [ # PC 8_all_in — Ronda 5: Máximo ganho (T1,T4-T8)
    # T1: Extended Baseline 150 epochs
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    # T4: Cosine + KL Annealing (Best scheduler combo)
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    # T5: Perceptual Loss
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
    # T6: Free Bits (Kingma et al. 2016)
    {'id': 'ronda5_t6_free_bits', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_FREE_BITS': '1.0'}},
    # T7: Cyclical KL Annealing (Fu et al. 2019)
    {'id': 'ronda5_t7_cyclical_kl', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_CYCLICAL_KL': 'true'}},
    # T8: All Techniques Combined
    {'id': 'ronda5_t8_all_techniques', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15', 'VAE_PERCEPTUAL_LOSS': '0.1', 'VAE_FREE_BITS': '1.0'}},
],
```

---

## OPÇÃO 4: FULL (8 testes, 16+ horas)

Copie isto para `EXPERIMENTS`:

```python
'8_full': [ # PC 8_full — Ronda 5 Completa: Todos os 8 testes
    # T1: Extended Baseline 150 epochs (determina teto absoluto)
    {'id': 'ronda5_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '150'}},
    # T2: Cosine LR Scheduling (T_max=100, não 50!)
    {'id': 'ronda5_t2_cosine_lr', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true'}},
    # T3: KL Annealing (warmup=15 epochs, normalized)
    {'id': 'ronda5_t3_kl_annealing_fixed', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    # T4: Cosine + KL Annealing (Both, 100 epochs)
    {'id': 'ronda5_t4_cosine_kl_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
    # T5: Perceptual Loss (0.8*MSE + 0.1*VGG + 0.1*KL)
    {'id': 'ronda5_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
    # T6: Free Bits (per-dimension KL ≥ 1.0)
    {'id': 'ronda5_t6_free_bits', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_FREE_BITS': '1.0'}},
    # T7: Cyclical KL Annealing (Fu et al. 2019)
    {'id': 'ronda5_t7_cyclical_kl', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_CYCLICAL_KL': 'true'}},
    # T8: All Techniques Combined (Cosine + KL + Perceptual + Free Bits)
    {'id': 'ronda5_t8_all_techniques', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15', 'VAE_PERCEPTUAL_LOSS': '0.1', 'VAE_FREE_BITS': '1.0'}},
],
```

---

## PASSO DE IMPLEMENTAÇÃO

### 1. Editar run_experiments.py

Abra [scripts/run_experiments.py](scripts/run_experiments.py)

Procure a linha:
```python
'7': [ # PC 7 — Ronda 5: Schedulers + Nova Arquiteturas
```

Após a **closing do array de '7'** (a linha com `]`), adicione uma das opções acima.

Exemplo:
```python
'7': [ # PC 7 — Ronda 5: ... (mantém existente)
    ...
],
'8_quick': [ # ← ADICIONA ISTO AQUI
    {'id': 'ronda5_t1_baseline_150ep', ...},
    ...
],
```

### 2. Actualizar argparse choices

Procure:
```python
parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5', '6', '7'], ...)
```

Mude para:
```python
parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5', '6', '7', '8_quick', '8_comprehensive', '8_all_in', '8_full'], ...)
```

Ou apenas para a opção que deseja. Exemplo, se quer apenas QUICK:
```python
parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5', '6', '7', '8_quick'], ...)
```

### 3. Actualizar cleanup vars

Na função `run_script()`, procure:
```python
_exp_vars = [
    'VAE_LATENT_DIM', 'VAE_BETA', 'VAE_LR', 'VAE_EPOCHS', 'VAE_COSINE_LR', 'VAE_KL_ANNEALING_EPOCHS',
    ...
]
```

Adicione (if using advanced tests):
```python
_exp_vars = [
    'VAE_LATENT_DIM', 'VAE_BETA', 'VAE_LR', 'VAE_EPOCHS', 
    'VAE_COSINE_LR', 'VAE_KL_ANNEALING_EPOCHS', 
    'VAE_PERCEPTUAL_LOSS', 'VAE_FREE_BITS', 'VAE_CYCLICAL_KL',  # ← ADD THESE
    'DCGAN_LATENT', 'DCGAN_NGF', 'DCGAN_NDF', 'DCGAN_BETA1', 'DCGAN_LR',
    'DIFF_CHANNELS', 'DIFF_T_STEPS', 'DIFF_LR', 'DIFF_BETA_START', 'DIFF_BETA_END',
]
```

---

## COMO EXECUTAR

Depois de copiar-colar, execute no terminal:

### QUICK START:
```bash
cd /Users/duartepereira/IAG
python3 scripts/run_experiments.py --pc 8_quick
```

### COMPREHENSIVE:
```bash
python3 scripts/run_experiments.py --pc 8_comprehensive
```

### ALL IN:
```bash
python3 scripts/run_experiments.py --pc 8_all_in
```

### FULL:
```bash
python3 scripts/run_experiments.py --pc 8_full
```

---

## VERIFICAÇÃO PÓS-EXECUÇÃO

Após completar os testes, verifique:

```bash
# Listar todos os testes executados
ls -1 /Users/duartepereira/IAG/results/ | grep ronda5

# Comparar FID de todos
for dir in /Users/duartepereira/IAG/results/ronda5_*; do
  if [ -f "$dir/results.csv" ]; then
    fid=$(grep FID "$dir/results.csv" | awk -F',' '{print $2}')
    echo "$(basename $dir): FID=$fid"
  fi
done
```

Ou use Python:
```python
import pandas as pd
from pathlib import Path

results = {}
for exp_dir in Path('/Users/duartepereira/IAG/results').glob('ronda5_*'):
    csv_file = exp_dir / 'results.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        fid = df[df['Metric'] == 'FID']['Value'].values[0] if 'FID' in df['Metric'].values else None
        if fid:
            results[exp_dir.name] = fid

# Sort by FID
for name, fid in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name}: FID = {fid:.1f}")
```

---

## CHECKLIST FINAL

Antes de executar:

```
☐ Leia RONDA5_QUICK_DECISION.md para decidir qual opção
☐ Copie o código correspondente de cima para run_experiments.py
☐ Actualize argparse choices
☐ Actualize cleanup _exp_vars
☐ Verifique que tem as implementations prontas:
  ☐ KL normalization fix (se T2-T8)
  ☐ Cosine scheduler (se T2,T4,T8)
  ☐ Perceptual loss (se T5,T8)
  ☐ Free bits (se T6,T8)
  ☐ Cyclical KL (se T7,T8)
☐ Corra: python3 scripts/run_experiments.py --pc 8_quick (ou outra escolha)
☐ Espera... (4-20+ horas dependendo opção)
☐ Analisa resultados
```

---

## PRÓXIMO PASSO

1. **Copie uma das 4 opções acima** para `EXPERIMENTS` em run_experiments.py
2. **Execute:** `python3 scripts/run_experiments.py --pc 8_quick` (ou opção escolhida)
3. **Espera**, **recolhe resultados**, **celebra vitória**

Boa sorte! 🚀

