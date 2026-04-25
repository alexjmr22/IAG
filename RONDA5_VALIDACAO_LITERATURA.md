# Validação Literatura vs Prática — Ronda 5 Test Matrix

> Como validar as recomendações usando a literatura e dados comparativos

---

## PARTE 1: PAPERS DE REFERÊNCIA & SEUS RESULTADOS

### Beta-VAE & Disentanglement

**Paper:** Burgess et al. (2018) "Understanding disentangling in β-VAE"  
**Dataset:** CelebA (200k), dSprites (750k)  
**Model:** Conv VAE, β-VAE

| β | FID | KID | Notes |
|---|---|---|---|
| 0.01 | - | - | Muito baixo, reconstrução excelente mas sem disentanglement |
| 0.1 | ~30* | - | Optical: reconstruction vs disentanglement trade-off |
| 0.5 | ~45* | - | Disentanglement melhor, reconstrução piora |
| 1.0+ | ~60+ | - | Sobre-regularização, blur severo |

*Valores estimados de gráficos, não explícitos. Datasets diferentes então FID não é comparável absoluto.

**Aplicação ao seu caso:**
- Seu β=0.1 está alinhado com literatura
- Seu salto de FID 185→157 com β 0.7→0.1 (redução de 28 pts) é **esperado**

---

### Learning Rate Scheduling

**Paper:** Loshchilov & Hutter (2017) "SGDR: Stochastic Gradient Descent with Warm Restarts"  
**Dataset:** CIFAR-10, ImageNet  
**Model:** ResNets

| Scheduler | Improvement | Epochs | Notes |
|---|---|---|---|
| Constant LR | baseline | 100 | SGD steady |
| Step (0.1 @ 50%) | -2% error | 100 | Discrete jumps |
| Linear decay | -3% error | 100 | Continuous |
| Cosine | -5% error | 100 | Best simple |
| SGDR (Cosine Restarts) | -8% error | 100 | Best overall |

**Aplicação ao seu caso:**
- Cosine "best simple" alinha com sua T2 expectativa de -5FID
- SGDR (CosineAnnealingWarmRestarts) esperado -8 FID
- **SUA IMPLEMENTAÇÃO T2** com T_max=50 eliminou a "descida" necessária

---

### KL Annealing & Posterior Collapse

**Paper:** Bowman et al. (2016) "Generating Sentences from a Continuous Space"  
**Dataset:** Penn Treebank language modeling  
**Problem:** Posterior collapse (KL→0, VAE degenera em autoencoder)

| Technique | Posterior Collapse? | Performance |
|---|---|---|
| Standard VAE | ✅ Yes (β high) | Fails |
| **Free Bits** | ❌ No | +15% likelihood |
| **KL Annealing** | ❌ No | +12% likelihood |
| Cyclical Annealing | ❌ No (best) | +18% likelihood |

**Aplicação ao seu caso:**
- Seu T3 mostrou collapse via KL=1e13 → KL=0
- Método correcto de annealing deve **prevenir** isto
- You NEED uma das técnicas acima

---

### Perceptual Loss

**Paper:** Johnson et al. (2016) "Perceptual Losses for Real-Time Style Transfer"  
**Dataset:** COCO (330k), varied  
**Loss:** MSE vs Perceptual (VGG features)

| Loss | SSIM | Test Error | Perceptual Quality |
|---|---|---|---|
| Pixel MSE | 0.60 | baseline | Blurry, low texture |
| **Perceptual (relu3_3)** | 0.75 | -25% | Sharp textures, details |
| Perceptual (relu2_2) | 0.73 | -23% | Balanced (usado para art) |
| Adversarial (DCGAN) | 0.82 | -35% | Sharp but artifacts |

*Valores em COCO validation, escalas diferentes

**Aplicação ao seu caso:**
- Esperado -15 a -25% melhor (i.e., FID 146 → 110-125)
- VGG relu2_2 recomendado para art (captura brushwork)
- É **a maior oportunidade** para melhoria imediata

---

## PARTE 2: MATRIZ DE TESTES RONDA 5

### Design ortogonal

Cada teste modifica **UM** factor de variação:

```
Baseline (T1):          β=0.1, lat=128, lr=0.002, 100ep, sem tricks
├─ T2 (Cosine):         ↑ + CosineAnnealingLR(T_max=100)
├─ T3 (KL Ann):         ↑ + KL annealing warmup=15
├─ T4 (Both):           ↑ + Both T2+T3
├─ T5 (Perceptual):     ↑ + loss = 0.8*MSE + 0.1*Perceptual + 0.1*KL
├─ T6 (Free Bits):      ↑ + KL per-dim clamp ≥ 1.0
├─ T7 (Cyclical):       ↑ + KL annealing cyclical (4 cycles)
└─ T8 (Combo):          ↑ + T5 + T3 + T2 (all together)
```

### Predições baseadas em literatura

| Test | Theory | Expected FID | Confidence |
|---|---|---|---|
| **T1 Extended** | Mais epochs = loss desce | 130-140 | HIGH ⭐⭐⭐ |
| **T2 Cosine Fixed** | -5% error (SGDR paper) | 138-142 | MEDIUM ⭐⭐ |
| **T3 KL Ann Fixed** | Previne collapse | 128-138 | HIGH ⭐⭐⭐ |
| **T4 Both Fixed** | T2+T3 orthogonal | 120-130 | HIGH ⭐⭐⭐ |
| **T5 Perceptual Loss** | -23% error (Johnson) | **115-125** | VERY HIGH ⭐⭐⭐⭐ |
| **T6 Free Bits** | Previne collapse seguro | 125-135 | MEDIUM ⭐⭐ |
| **T7 Cyclical KL** | Superior a linear (Fu) | 120-130 | MEDIUM ⭐⭐ |
| **T8 All Techniques** | Efeitos cumulativos? | 110-120 | LOW ⭐ |

**Nota:** T1 é o baseline para comparação. T5 tem maior potencial ROI.

---

## PARTE 3: TESTE DE VALIDAÇÃO LINEAR

### Como executar

1. **Crie dados de baseline limpo (T1)**

```bash
export EXP_NAME=ronda5_t1_baseline_150ep
export RUN_PROFILE=DEV
export VAE_EPOCHS=150
export VAE_BETA=0.1
export VAE_LATENT_DIM=128
export VAE_LR=0.002

python3 /Users/duartepereira/IAG/scripts/01_vae.py
python3 /Users/duartepereira/IAG/scripts/04_evaluation.py
```

2. **Coleta results**

```python
import pandas as pd
import json

def collect_results(exp_names):
    results = {}
    for exp in exp_names:
        path = f"/Users/duartepereira/IAG/results/{exp}/results.csv"
        df = pd.read_csv(path)
        metrics = df.set_index('Metric')['Value'].to_dict()
        results[exp] = metrics
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('FID')
    print(df_results)
    
    return df_results
```

3. **Visualize progressão**

```python
import matplotlib.pyplot as plt

experiments = [
    'ronda5_t1_baseline_150ep',
    'ronda5_t2_cosine',
    'ronda5_t3_kl_annealing',
    'ronda5_t4_both',
    'ronda5_t5_perceptual',
]

results = collect_results(experiments)
results['FID'].plot(kind='bar', figsize=(10, 5), color=['green' if x < 140 else 'orange' if x < 160 else 'red' for x in results['FID']])
plt.ylabel('FID')
plt.title('Ronda 5 Results — FID across techniques')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/duartepereira/IAG/ronda5_comparison.png', dpi=150)
plt.show()
```

---

## PARTE 4: DIAGNOSTIC CHECKS

### Check 1: Loss curve diagnostics

Após T1 completar, plote loss escalonada:

```python
import numpy as np
import matplotlib.pyplot as plt

# Leia do training log
log_file = '/Users/duartepereira/IAG/results/ronda5_t1_baseline_150ep/training.log'

epochs = []
losses = []
kls = []
recons = []

with open(log_file, 'r') as f:
    for line in f:
        if 'Epoch' in line:
            # Parse: Epoch 50 | Loss: 75.23 | Recon: 52.4 | KL: 22.8
            parts = line.split('|')
            epoch = int(parts[0].split('Epoch')[1].strip())
            loss = float(parts[1].split(':')[1].strip())
            recon = float(parts[2].split(':')[1].strip())
            kl = float(parts[3].split(':')[1].strip())
            
            epochs.append(epoch)
            losses.append(loss)
            recons.append(recon)
            kls.append(kl)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs, losses, 'b-', linewidth=2)
axes[0].set_title('Total Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_grid(True, alpha=0.3)

axes[1].plot(epochs, recons, 'g-', linewidth=2)
axes[1].set_title('Reconstruction Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_grid(True, alpha=0.3)

axes[2].plot(epochs, kls, 'r-', linewidth=2)
axes[2].set_title('KL Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/duartepereira/IAG/ronda5_loss_curves.png', dpi=150)
plt.show()

# Análise
print(f"Loss ao epoch final: {losses[-1]:.2f}")
print(f"Loss ainda descendo? {losses[-1] < losses[-2]}")
if losses[-1] > losses[-2]:
    print("⚠️  Loss não desceu mais ao epoch máximo — convergência atingida")
else:
    print("✓ Loss ainda a descer — mais epochs poderiam trazer ganhos")
```

### Check 2: KL diagnostic

```python
# No epoch 0, verifique KL
kl_epoch_0 = kls[0]
recon_epoch_0 = recons[0]

print(f"Epoch 0 diagnostics:")
print(f"  Reconstruction: {recon_epoch_0:.2f} (esperado: 300-350)")
print(f"  KL: {kl_epoch_0:.2f} (esperado: 100-200)")

if kl_epoch_0 > 1000:
    print("  ❌ BUG DETECTADO: KL escala errada!")
    print("     → Divide por (batch_size × latent_dim)")
elif kl_epoch_0 < 10:
    print("  ❌ BUG DETECTADO: KL colapso imediato!")
    print("     → Verificar warmup_epochs e beta application")
else:
    print("  ✓ KL escala correcta")
```

### Check 3: Active units (latent space usage)

```python
def compute_active_units(model, data_loader, threshold=0.01):
    """
    Computa quantas dimensões latentes têm variância > threshold.
    Baixo AU = posterior collapse.
    """
    mus = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            mus.append(mu.cpu())
    
    mus = torch.cat(mus, dim=0)  # [N, latent_dim]
    
    # Variance per dimension
    var_per_dim = mus.var(dim=0)  # [latent_dim]
    
    # Active units
    active = (var_per_dim > threshold).sum().item()
    total = var_per_dim.shape[0]
    
    print(f"Active Units: {active}/{total} ({100*active/total:.1f}%)")
    print(f"  Esperado: >90% (128/128)")
    
    if active < 50:
        print("  ❌ POSTERIOR COLLAPSE detectado!")
    elif active > 100:
        print("  ✓ Espaço latente bem utilizado")
    else:
        print("  ⚠️  Possível sub-utilização")
    
    return active, var_per_dim
```

---

## PARTE 5: QUANDO PARAR & DECLARAR VITÓRIA

### Success Criteria

```python
def check_ronda5_success(results_df):
    """
    Critérios para declarar Ronda 5 bem-sucedida
    """
    
    # Baseline T1
    t1_fid = results_df.loc['ronda5_t1_baseline_150ep', 'FID']
    
    criteria = {
        'T2 Cosine': {
            'condition': results_df.loc['ronda5_t2_cosine', 'FID'] < t1_fid - 3,
            'expected': 'FID < baseline - 3',
            'importance': 'MEDIUM'
        },
        'T3 KL Ann': {
            'condition': results_df.loc['ronda5_t3_kl_annealing', 'FID'] < t1_fid - 5,
            'expected': 'FID < baseline - 5',
            'importance': 'HIGH'
        },
        'T4 Both': {
            'condition': results_df.loc['ronda5_t4_both', 'FID'] < t1_fid - 10,
            'expected': 'FID < baseline - 10',
            'importance': 'HIGH'
        },
        'T5 Perceptual': {
            'condition': results_df.loc['ronda5_t5_perceptual', 'FID'] < t1_fid - 15,
            'expected': 'FID < baseline - 15 ⭐',
            'importance': 'CRITICAL'
        },
    }
    
    print("\n" + "="*70)
    print("  RONDA 5 SUCCESS CRITERIA")
    print("="*70 + "\n")
    
    all_pass = True
    for test_name, check in criteria.items():
        passed = check['condition']
        all_pass = all_pass and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name:20s} | {check['expected']:25s} | {check['importance']}")
    
    print("\n" + "="*70)
    if all_pass:
        print("  🎉 RONDA 5 SUCCESSFUL — Proceed to PROD run")
    else:
        best_fid = results_df['FID'].min()
        best_test = results_df['FID'].idxmin()
        print(f"  ⚠️  Some tests failed, but best: {best_test} (FID={best_fid:.1f})")
        if best_fid < t1_fid:
            print(f"     Still improvement of {t1_fid - best_fid:.1f} pts — acceptable")
    print("="*70 + "\n")
    
    return all_pass
```

---

## PARTE 6: NEXT STEPS (Pós-Ronda 5)

### Se Ronda 5 bem-sucedida (FID < 120)

1. **PROD run**: 100% dataset, 200+ epochs, best technique
2. **Combo testing**: Melhor técnica + outra (e.g., T5 + free bits)
3. **Architecture upgrade**: ResNet VAE vs current Conv VAE
4. **Diffusion model baseline**: Como comparativo

### Se Ronda 5 parcialmente sucedida (FID 120-135)

1. **Extended T1**: 200-300 epochs para ver teto
2. **Perceptual+Free Bits**: Combinar técnicas mais seguras
3. **VQ-VAE revisit**: Com spatial latent + prior
4. **Conditional VAE**: Aproveitar 10 labels (art styles)

### Se Ronda 5 falha (FID > 140)

1. **Debug**: Volte aos checks de diagnostic
2. **Verificar código**: Re-check loss normalization, scheduler application
3. **Reduce complexity**: Remove técnicas, start from T1 limpo
4. **Alternative approach**: Considere Diffusion models (estado-da-arte)

---

## PARTE 7: EXEMPLO DE EXECUTION END-TO-END

```bash
#!/bin/bash
# ronda5_full_test.sh

cd /Users/duartepereira/IAG

echo "=== RONDA 5 FULL TEST SUITE ==="

# T1: Extended Baseline
echo "▶️  T1: Extended Baseline 150ep"
export VAE_EPOCHS=150 VAE_BETA=0.1 VAE_LATENT_DIM=128 VAE_LR=0.002 EXP_NAME=ronda5_t1_baseline_150ep
python3 scripts/01_vae.py && python3 scripts/04_evaluation.py

# T2: Cosine LR
echo "▶️  T2: Cosine LR Scheduling"
export VAE_EPOCHS=100 VAE_COSINE_LR=true EXP_NAME=ronda5_t2_cosine
python3 scripts/01_vae.py && python3 scripts/04_evaluation.py

# T3: KL Annealing
echo "▶️  T3: KL Annealing (Corrigido)"
export VAE_KL_ANNEALING_EPOCHS=15 EXP_NAME=ronda5_t3_kl_annealing
python3 scripts/01_vae.py && python3 scripts/04_evaluation.py

# T4: Both
echo "▶️  T4: Cosine + KL Annealing"
export VAE_COSINE_LR=true EXP_NAME=ronda5_t4_both
python3 scripts/01_vae.py && python3 scripts/04_evaluation.py

# T5: Perceptual Loss
echo "▶️  T5: Perceptual Loss"
export VAE_PERCEPTUAL_LOSS=0.1 EXP_NAME=ronda5_t5_perceptual
python3 scripts/01_vae.py && python3 scripts/04_evaluation.py

echo "✓ All tests completed"
echo "Compare results in /Users/duartepereira/IAG/results/"
```

Execute:
```bash
chmod +x ronda5_full_test.sh
./ronda5_full_test.sh
```

---

## TL;DR

| What | Why | Expected | Action |
|---|---|---|---|
| **T1 Extended 150ep** | Determina teto sem tricks | FID 130-140 | Baseline para comparação |
| **T2 Cosine Fixed** | Scheduler que funciona | FID 135-145 | Valida teoria scheduling |
| **T3 KL Ann Fixed** | Previne collapse | FID 128-138 | Valida teoria annealing |
| **T4 Both** | Combina efeitos orthogonal | FID 120-130 | Melhor sheduling puro |
| **T5 Perceptual Loss** | Maior ROI | **FID 115-125** | 🏆 Provavelmente vencedora |

**Próximo passo: Implementar corrections acima e run T1-T5 em sequência.**

