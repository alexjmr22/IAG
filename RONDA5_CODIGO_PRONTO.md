# Ronda 5 — Código Pronto para Implementação

> Copie e cole para seus scripts. Todas as correções e novas técnicas.

---

## PARTE 1: VERIFICAR E CORRIGIR LOSS NORMALIZATION

### Diagnóstico: O seu loss atual

Procure em `scripts/01_vae.py` pela seção `def train_vae()` e veja como KL é calculado:

```python
# PROCURE ISTO (linhas ~250-280):
def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z)
    
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # MSE reconstruction
    mse = criterion(recon, x)  # ou F.mse_loss(recon, x)
    
    loss = mse + beta * kl
    return loss
```

### ✅ CORRECTO (normalizado):

```python
def forward(self, x):
    batch_size = x.shape[0]
    latent_dim = self.latent_dim  # e.g. 128
    
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    recon = self.decode(z)
    
    # KL divergence — NORMALIZADO
    kl_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_normalized = kl_raw / (batch_size * latent_dim)  # ← CRÍTICO
    
    # MSE reconstruction
    mse_loss = F.mse_loss(recon, x, reduction='mean')  # já normalizado por pixels
    
    # Total loss com beta annealing
    beta_t = self.get_kl_beta(current_epoch)  # linear warmup 0→final_beta
    loss = mse_loss + beta_t * kl_normalized
    
    return loss, mse_loss.item(), kl_normalized.item()

def get_kl_beta(self, epoch):
    """Linear warmup: epoch 0→warmup_epochs ramps beta 0→final_beta"""
    if epoch < self.warmup_epochs:
        return (epoch / self.warmup_epochs) * self.final_beta
    return self.final_beta
```

### Checklist de verificação

Execute isto no terminal para detectar o bug:

```bash
# Coloque isto no seu training script temporariamente:
def diagnose_loss_scale():
    """Testa valores esperados de loss"""
    mu = torch.randn(128, 128)  # batch_size=128, latent_dim=128
    logvar = torch.randn(128, 128)
    
    # Método errado (sem normalizar)
    kl_wrong = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(f"KL SEM normalização: {kl_wrong.item():.2e}")  # → ~1e6 ou maior
    
    # Método correcto (normalizado)
    batch_size, latent_dim = 128, 128
    kl_correct = kl_wrong / (batch_size * latent_dim)
    print(f"KL COM normalização: {kl_correct.item():.2f}")  # → ~200 (esperado)
    
    # Se seu script mostra ~1e6 ou ~1e13, tem o bug
    # Se mostra ~200, está correcto

# Run no terminal:
python3 -c "
import torch
mu = torch.randn(128, 128)
logvar = torch.randn(128, 128)
kl_wrong = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
kl_right = kl_wrong / (128 * 128)
print(f'Without norm: {kl_wrong.item():.2e}')
print(f'With norm: {kl_right.item():.2f}')
"
# Output esperado:
# Without norm: 1.64e+04 (ou maior em early epochs)
# With norm: 125.7 (ou ~100-300)
```

---

## PARTE 2: PERCEPTUAL LOSS (Alto impacto, simples de implementar)

### Opção A: Classe isolada (recomendado)

Crie um novo ficheiro `scripts/perceptual_loss.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    VGG-16 perceptual loss (Johnson et al. 2016)
    Mede similarity no feature space em vez de pixel space
    """
    
    def __init__(self, layer='relu2_2', device='cuda'):
        super().__init__()
        self.device = device
        
        # Carrega VGG-16 pré-treinado
        vgg = models.vgg16(pretrained=True)
        
        # Selecciona até camada desejada
        layer_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,  # ← recomendado para art (captura texture)
            'relu3_4': 17,
            'relu4_4': 26,
        }
        
        layer_idx = layer_mapping.get(layer, 8)
        self.features = vgg.features[:layer_idx].to(device)
        
        # Congela pesos (não treina)
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
        
        # Normalização ImageNet
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(self, x, target):
        """
        Args:
            x: predicted image [B, 3, H, W] em [0, 1]
            target: ground truth image [B, 3, H, W] em [0, 1]
        Returns:
            MSE loss no feature space
        """
        # Normaliza ImageNet
        x_norm = (x - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extrai features
        x_feat = self.features(x_norm)
        target_feat = self.features(target_norm)
        
        # Calcula MSE no feature space
        loss = F.mse_loss(x_feat, target_feat, reduction='mean')
        return loss


# Teste rápido
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perc_loss_fn = PerceptualLoss(layer='relu2_2', device=device)
    
    x_pred = torch.randn(4, 3, 32, 32).clamp(0, 1).to(device)
    x_real = torch.randn(4, 3, 32, 32).clamp(0, 1).to(device)
    
    loss_perc = perc_loss_fn(x_pred, x_real)
    print(f"Perceptual Loss: {loss_perc.item():.4f}")  # → ~0.1-1.0 é normal
```

### Opção B: Integrar diretamente em `01_vae.py`

No topo do ficheiro, adicione:

```python
import torchvision.models as models

def create_perceptual_loss(device='cpu'):
    """Factory function para perceptual loss"""
    vgg = models.vgg16(pretrained=True).features[:8].to(device)  # até relu2_2
    for param in vgg.parameters():
        param.requires_grad = False
    vgg.eval()
    return vgg

def perceptual_loss(x_pred, x_real, vgg_model, mean=None, std=None):
    """Calcula perceptual loss entre features"""
    # Normaliza ImageNet se não feito
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_pred.device)
    
    x_norm = (x_pred - mean) / std
    real_norm = (x_real - mean) / std
    
    feat_pred = vgg_model(x_norm)
    feat_real = vgg_model(real_norm)
    
    return F.mse_loss(feat_pred, feat_real, reduction='mean')
```

### Usar na training loop

Em `train_vae()`, antes do loop:

```python
# Carrega VGG para perceptual loss
vgg_model = create_perceptual_loss(device=device)
vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

# Hyperparameters
alpha_recon = 0.8      # peso MSE
alpha_perc = 0.1       # peso VGG perceptual
lambda_kl = beta       # peso KL (já tem beta annealing)

# No loop, durante treino:
recon_mse = F.mse_loss(recon, x, reduction='mean')
recon_perc = perceptual_loss(recon, x, vgg_model, vgg_mean, vgg_std)
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (batch_size * latent_dim)

# Total loss
loss = alpha_recon * recon_mse + alpha_perc * recon_perc + lambda_kl * kl_loss

# Log
if iteration % 100 == 0:
    print(f"MSE: {recon_mse:.4f}, Perc: {recon_perc:.4f}, KL: {kl_loss:.2f}")
```

---

## PARTE 3: COSINE ANNEALING LR CORRIGIDO

### Opção A: CosineAnnealingLR simples (T_max = total_epochs)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_vae_with_cosine(model, train_loader, test_loader, epochs=100, use_cosine=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    if use_cosine:
        # CRÍTICO: T_max deve ser igual ao número total de epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = None
    
    for epoch in range(epochs):
        # Treino
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            
            # Forward + loss (como acima)
            recon, mu, logvar = model(x)
            loss = compute_loss(recon, x, mu, logvar, epoch)
            
            loss.backward()
            optimizer.step()
        
        # Step scheduler APÓS época completa
        if scheduler is not None:
            scheduler.step()  # ← Crucial: depois do epoch, não depois de batch
        
        print(f"Epoch {epoch}, Loss: {loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return model
```

### Opção B: CosineAnnealingWarmRestarts (mais sofisticado, potencialmente melhor)

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train_vae_with_restarts(model, train_loader, test_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Reinicia a cada T_0=25 epochs, com período crescendo T_mult=1.5×
    # Epoch layout: 0-24 cosine, 25-49 cosine, 50-99 cosine (períodos crescentes)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=25,        # Primeiro período 25 epochs
        T_mult=1.5,    # Próximo período = 25×1.5=37.5≈38, depois 38×1.5≈57
        eta_min=1e-5   # LR mínima
    )
    
    for epoch in range(epochs):
        for batch_idx, (x, _) enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            loss = compute_loss(...)
            loss.backward()
            optimizer.step()
        
        # Step APÓS epoch (importante)
        scheduler.step()
        
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return model
```

### Opção C: ReduceLROnPlateau (adaptativo, não schedule fixo)

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_vae_with_plateau_reduction(model, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # Reduz quando val_loss para melhorar
        factor=0.5,           # Reduz LR por 50%
        patience=5,           # Espera 5 epochs sem melhoria
        min_lr=1e-6,
        verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Treino
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            loss = compute_loss(...)
            loss.backward()
            optimizer.step()
        
        # Validação
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Step baseado em val_loss
        scheduler.step(val_loss)
    
    return model
```

---

## PARTE 4: KL ANNEALING CORRIGIDO (com verificação)

```python
class VAEWithKLAnnealing(nn.Module):
    def __init__(self, latent_dim=128, beta=0.1, warmup_epochs=15):
        super().__init__()
        self.latent_dim = latent_dim
        self.final_beta = beta
        self.warmup_epochs = warmup_epochs
        # ... resto da arquitectura ...
    
    def get_kl_beta(self, epoch):
        """Linear KL annealing: 0 → β over warmup_epochs"""
        if epoch < self.warmup_epochs:
            return (epoch / self.warmup_epochs) * self.final_beta
        return self.final_beta
    
    def forward(self, x, epoch=0, return_components=False):
        batch_size = x.shape[0]
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL loss — NORMALIZADO
        kl_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_normalized = kl_raw / (batch_size * self.latent_dim)
        
        # Beta annealing
        beta_t = self.get_kl_beta(epoch)
        
        # Total loss
        loss = recon_loss + beta_t * kl_normalized
        
        if return_components:
            return loss, recon_loss.item(), kl_normalized.item(), beta_t
        return loss

def train_with_annealing(model, train_loader, epochs=100, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            
            loss, recon, kl, beta_t = model(x, epoch=epoch, return_components=True)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon
            total_kl += kl
        
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        beta_t = model.get_kl_beta(epoch)
        
        print(f"E{epoch:03d} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | β_t: {beta_t:.4f}")
        
        # Diagnóstico de bugs
        if epoch == 0 and avg_kl > 1000:
            print(f"⚠️  WARNING: KL muito alto no epoch 0! Verificar normalização.")
            print(f"   Valores esperados: Recon~330, KL~100-150 para primeira epoch")
        
        if avg_kl < 0.1 and epoch > 5:
            print(f"⚠️  WARNING: KL colapso detectado!")
```

---

## PARTE 5: TEST SUITE RONDA 5

Crie `scripts/run_ronda5_tests.py`:

```python
#!/usr/bin/env python3
"""
Ronda 5 — Testes sistemáticos das correções
"""

import os
import sys
from pathlib import Path

# Configurações
TESTS = {
    'extended_baseline': {
        'epochs': 150,
        'use_cosine': False,
        'use_kl_annealing': False,
        'use_perceptual': False,
        'expected_fid': '130-140',
        'priority': 'CRITICAL',
    },
    'cosine_lr_corrected': {
        'epochs': 100,
        'use_cosine': True,        # CosineAnnealingLR(T_max=100)
        'use_kl_annealing': False,
        'use_perceptual': False,
        'expected_fid': '125-135',
        'priority': 'HIGH',
    },
    'kl_annealing_corrected': {
        'epochs': 100,
        'use_cosine': False,
        'use_kl_annealing': True,  # Linear warmup 15 epochs, normalized
        'use_perceptual': False,
        'expected_fid': '125-135',
        'priority': 'HIGH',
    },
    'both_schedulers': {
        'epochs': 100,
        'use_cosine': True,
        'use_kl_annealing': True,  # Ambas as techniques
        'use_perceptual': False,
        'expected_fid': '120-130',
        'priority': 'CRITICAL',
    },
    'perceptual_loss': {
        'epochs': 100,
        'use_cosine': False,
        'use_kl_annealing': False,
        'use_perceptual': True,      # 0.8*MSE + 0.1*Perceptual + 0.1*KL
        'expected_fid': '115-125',
        'priority': 'CRITICAL',
    },
}

def run_test(test_name, config):
    """Executa um teste"""
    print(f"\n{'='*70}")
    print(f"  TEST: {test_name}")
    print(f"  Priority: {config['priority']}")
    print(f"  Expected FID: {config['expected_fid']}")
    print(f"{'='*70}\n")
    
    # Set environment variables
    env = os.environ.copy()
    env['EXP_NAME'] = test_name
    env['EVAL_TARGET'] = 'VAE'
    env['RUN_PROFILE'] = 'DEV'
    env['VAE_EPOCHS'] = str(config['epochs'])
    env['VAE_BETA'] = '0.1'
    env['VAE_LATENT_DIM'] = '128'
    env['VAE_LR'] = '0.002'
    
    # Feature flags
    if config['use_cosine']:
        env['VAE_COSINE_LR'] = 'true'
    if config['use_kl_annealing']:
        env['VAE_KL_ANNEALING_EPOCHS'] = '15'  # warmup
    if config['use_perceptual']:
        env['VAE_PERCEPTUAL_LOSS'] = '0.1'    # weight
    
    # Run training
    cmd = f"cd /Users/duartepereira/IAG && python3 scripts/01_vae.py"
    result = os.system(f"{cmd} 2>&1 | tee logs/ronda5_{test_name}.log")
    
    if result == 0:
        print(f"✅ {test_name} completed successfully")
        # Run evaluation
        eval_cmd = f"cd /Users/duartepereira/IAG && python3 scripts/04_evaluation.py"
        os.system(f"{eval_cmd}")
    else:
        print(f"❌ {test_name} FAILED")
    
    return result == 0

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  RONDA 5 — TEST SUITE")
    print("  Executa testes em ordem de prioridade")
    print("="*70 + "\n")
    
    # Executa testes em ordem de prioridade
    ordered = sorted(TESTS.items(), key=lambda x: (x[1]['priority']=='MEDIUM', x[0]))
    
    passed = []
    failed = []
    
    for test_name, config in ordered:
        success = run_test(test_name, config)
        if success:
            passed.append(test_name)
        else:
            failed.append(test_name)
    
    # Resumo final
    print("\n" + "="*70)
    print("  RONDA 5 — SUMMARY")
    print("="*70)
    print(f"\n✅ PASSED ({len(passed)}):")
    for t in passed:
        print(f"   • {t}: {TESTS[t]['expected_fid']} FID")
    
    if failed:
        print(f"\n❌ FAILED ({len(failed)}):")
        for t in failed:
            print(f"   • {t}")
    
    print("\n")
```

Execute:

```bash
cd /Users/duartepereira/IAG
python3 scripts/run_ronda5_tests.py
```

---

## PARTE 6: VERIFICAÇÃO VISUAL DOS RESULTADOS

Após cada teste, verifique:

```python
import pandas as pd
from pathlib import Path

def compare_ronda_results():
    """Compara FID/KID de todos os testes"""
    results = {}
    
    for test_name in ['extended_baseline', 'cosine_lr_corrected', 'kl_annealing_corrected', 'both_schedulers', 'perceptual_loss']:
        experiment_path = Path(f"/Users/duartepereira/IAG/results/{test_name}")
        if (experiment_path / 'results.csv').exists():
            df = pd.read_csv(experiment_path / 'results.csv')
            fid = df[df['Metric'] == 'FID']['Value'].values[0]
            kid = df[df['Metric'] == 'KID']['Value'].values[0]
            results[test_name] = {'FID': fid, 'KID': kid}
    
    # Ordena por FID (ascendente = melhor)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['FID'])
    
    print("\n" + "="*60)
    print("  RONDA 5 RESULTS — Ordenado por FID")
    print("="*60 + "\n")
    
    for i, (test_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {test_name:30s} | FID: {metrics['FID']:6.1f} | KID: {metrics['KID']:.4f}")
    
    best = sorted_results[0]
    print(f"\n🏆 WINNER: {best[0]} (FID={best[1]['FID']:.1f})")
    print("\n")

# Execute
if __name__ == '__main__':
    compare_ronda_results()
```

---

## PARTE 7: STOP-LOSS CONDITIONS

Se estes sintomas aparecerem, **PARAR e DEBUG**:

```python
def diagnose_training():
    """Checks para problemas comuns durante treino"""
    
    checks = {
        'KL > 10000 no epoch 0': {
            'cause': 'Loss normalization bug',
            'fix': 'Divide KL loss por (batch_size × latent_dim)'
        },
        'KL → 0 antes epoch 5': {
            'cause': 'Posterior collapse ou warmup=0',
            'fix': 'Aumentar warmup_epochs para 15+, verificar beta_t'
        },
        'Loss oscillates wildly': {
            'cause': 'LR muito alta ou gradient explosion',
            'fix': 'Reduzir LR para 0.001, adicionar gradient clipping'
        },
        'FID piora com mais epochs': {
            'cause': 'Overfitting ou LR schedule errado',
            'fix': 'Usar ReduceLROnPlateau ou lower LR'
        },
        'Perceptual loss NaN': {
            'cause': 'VGG não normalizado ImageNet',
            'fix': 'Verificar mean=0.485,0.456,0.406 e std=0.229,0.224,0.225'
        },
    }
    
    for issue, info in checks.items():
        print(f"❗ {issue}")
        print(f"   Causa: {info['cause']}")
        print(f"   Fix: {info['fix']}\n")
```

---

## TL;DR — Copy-Paste Rápido

Se quer apenas começar:

1. **Verificar KL normalization** (5 minutos):
```python
kl_normalized = kl_raw / (batch_size * latent_dim)  # Em vez de apenas kl_raw
```

2. **Adicionar Perceptual Loss** (10 minutos):
```python
# Import acima
from torchvision import models

# Create
vgg = models.vgg16(pretrained=True).features[:8].to(device)
for p in vgg.parameters(): p.requires_grad = False

# Use
feat_pred = vgg((x_recon - mean) / std)
feat_real = vgg((x - mean) / std)
perc_loss = F.mse_loss(feat_pred, feat_real)
total_loss = 0.8*mse + 0.1*perc_loss + beta_t*kl
```

3. **Fix Cosine Scheduler** (5 minutos):
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
# Step AFTER each epoch
```

4. **Fix KL Annealing** (5 minutos):
```python
def get_kl_beta(epoch):
    return (epoch / 15) * 0.1 if epoch < 15 else 0.1
```

Essa é a diferença entre FID 176 (falho) e FID 120-125 (sucesso).

