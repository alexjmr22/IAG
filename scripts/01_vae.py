#!/usr/bin/env python
# coding: utf-8

# # 01 — Variational Autoencoder (VAE)
# 
# Baseado na `ConvVAE` do notebook **3 - Autoencoders and Variational Autoencoders**, adaptada para:
# - 3 canais RGB (em vez de 1 canal MNIST)
# - imagens 32×32 do ArtBench-10
# 
# ### Checklist
# - [ ] Encoder CNN → (μ, log σ²)
# - [ ] Reparameterisation trick
# - [ ] Decoder ConvTranspose2d → imagem
# - [ ] Loss = reconstrução (BCE/MSE) + β·KL
# - [ ] Treinar no subset 20% → depois no dataset completo
# - [ ] Guardar checkpoint para avaliação FID/KID
# 
# ### Extensões (bónus)
# - β-VAE (β > 1)
# - CVAE condicionado ao estilo artístico
# - Interpolação no espaço latente

# In[7]:


from __future__ import annotations
import sys, random, csv
from pathlib import Path
import os, multiprocessing, warnings, sys
warnings.filterwarnings('ignore')
if multiprocessing.current_process().name != 'MainProcess':
    sys.stdout = open(os.devnull, 'w')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / 'TP' / 'TP1-alunos-src-only' / 'scripts'
SUBSET_CSV  = REPO_ROOT / 'TP' / 'TP1-alunos-src-only' / 'student_start_pack' / 'training_20_percent.csv'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── reproducibility & device ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print('Device:', device)


# ## 1. Hiperparâmetros

# In[8]:


import os
IMAGE_SIZE  = 32
LATENT_DIM  = int(os.environ.get('VAE_LATENT_DIM', 128))
BETA        = float(os.environ.get('VAE_BETA', 0.7))    # peso KL (aumentar para β-VAE)
BATCH_SIZE  = 128    # conforme Notebooks 3, 4, 5
LR          = float(os.environ.get('VAE_LR', 1e-3))

from config import cfg
EPOCHS      = cfg.vae_epochs
USE_SUBSET  = cfg.use_subset   # False → dataset completo (avaliação final)

EXP_NAME = os.environ.get('EXP_NAME', 'vae')
OUT_DIR = REPO_ROOT / 'results' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOG DE PARÂMETROS (.md) ──────────────────────────────────────────────────
params_md = f"""# VAE Experiment: {EXP_NAME}
- **Date**: {pd.Timestamp.now() if 'pd' in globals() else 'N/A'}
- **Latent Dim**: {LATENT_DIM}
- **Beta (KL)**: {BETA}
- **Learning Rate**: {LR}
- **Epochs**: {EPOCHS}
- **Batch Size**: {BATCH_SIZE}
- **Dataset**: {"20% Subset" if USE_SUBSET else "Full ArtBench10"}
- **Profile**: {cfg.__class__.__name__ if hasattr(cfg, '__class__') else 'N/A'}
"""
with open(OUT_DIR / "experiment_params.md", "w", encoding="utf-8") as f:
    f.write(params_md)
print(f"Parâmetros guardados em {OUT_DIR / 'experiment_params.md'}")


# ## 2. Dataset — reutiliza `HFDatasetTorch` do notebook 00

# In[9]:


def safe_num_workers():
    return 4
class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds      = hf_split
        self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        x  = self.transform(ex['image']) if self.transform else ex['image']
        return x, int(ex['label'])

def load_ids_from_csv(csv_path, column='train_id_original'):
    ids = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            v = str(row.get(column, '')).strip()
            if v: ids.append(int(v))
    print(f'Loaded {len(ids)} ids')
    return ids

transform = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

from datasets import load_dataset, load_from_disk

DATA_CACHE = REPO_ROOT / 'data' / 'artbench10_hf'
if DATA_CACHE.exists():
    print(f'A carregar dataset do disco: {DATA_CACHE}')
    train_hf = load_from_disk(str(DATA_CACHE))['train']
else:
    print('A fazer download do dataset (primeira vez)...')
    ds = load_dataset('zguo0525/ArtBench')
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATA_CACHE))
    print(f'Dataset guardado em: {DATA_CACHE}')
    train_hf = ds['train']

subset_ids   = load_ids_from_csv(SUBSET_CSV) if USE_SUBSET else None
train_ds     = HFDatasetTorch(train_hf, transform, indices=subset_ids)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=safe_num_workers(), pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)
print(f'Amostras: {len(train_ds)} | Batches: {len(train_loader)}')


# ## 3. Arquitectura — adaptada de `ConvVAE` (notebook 3)
# 
# Diferenças face ao original MNIST:
# - `in_channels=3` (RGB)
# - encoder stride ajustado para 32×32 → bottleneck 4×4
# - decoder espelha o encoder

# In[10]:


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder (32×32 → 4×4) ────────────────────────────────────────────
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 32→16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8→4
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # ── Decoder (latent → 32×32) ──────────────────────────────────────────
        self.dec_fc   = nn.Linear(latent_dim, 128 * 4 * 4)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4→8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 8→16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 16→32
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 4, 4)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = ConvVAE(LATENT_DIM).to(device)
print(f'Parâmetros: {sum(p.numel() for p in model.parameters()):,}')


# ## 4. Loss e optimizador — baseados em `vae_loss` / `train_vae` (notebook 3)

# In[11]:


def vae_loss(xhat, x, mu, logvar, beta=0.7):
    """Reconstrução (MSE) + β·KL  —  mesma formulação do notebook 3."""
    recon = F.mse_loss(xhat, x, reduction='sum') / x.size(0)
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl


def train_vae(model, loader, optimizer, epochs=50, beta=0.7):
    """Loop de treino do VAE — adaptado de train_vae (notebook 3)."""
    model.train()
    history = []
    for ep in range(epochs):
        tl = tr = tk = 0.0
        for x, _ in tqdm(loader, desc=f'Epoch {ep+1}/{epochs}', leave=False):
            x = x.to(device)
            
            # ANTES: optimizer.zero_grad()
            # DEPOIS: set_to_none=True
            # MOTIVO: Evita gerir memória com matrizes completas de float 0.0, apontando apenas as memórias dos gradientes como nulas. Em GPUs tipo MPS e M1/M4, acelera a alocação do retro-passo.
            optimizer.zero_grad(set_to_none=True)
            
            xhat, mu, logvar = model(x)
            loss, recon, kl  = vae_loss(xhat, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            # ANTES: tl += loss.item() * x.size(0)
            #        tr += recon.item() * x.size(0)
            #        tk += kl.item()   * x.size(0)
            # DEPOIS: Em vez de .item(), usamos .detach() a cada iteração individual.
            # MOTIVO: O `.item()` obriga a Placa Gráfica a suspender-se e a parar iterativamente para dar o valor solto ao CPU. Com o `.detach()` o processamento ocorre ininterruptamente no GPU para maior it/s!
            tl += loss.detach() * x.size(0)
            tr += recon.detach() * x.size(0)
            tk += kl.detach() * x.size(0)
            
        n = len(loader.dataset)
        
        # Só chamamos e exigimos o update do .item() ao CPU apenas no Final de casa Epoch estatístico!
        epoch_loss  = (tl/n).item()
        epoch_recon = (tr/n).item()
        epoch_kl    = (tk/n).item()
        
        history.append({'loss': epoch_loss, 'recon': epoch_recon, 'kl': epoch_kl})
        print(f'Epoch {ep+1:03d}/{epochs} | loss={epoch_loss:.4f}  recon={epoch_recon:.4f}  kl={epoch_kl:.4f}')
        
        # amostras intermédias
        if cfg.save_samples and (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                z  = torch.randn(64, LATENT_DIM, device=device)
                imgs = (model.decode(z) * 0.5 + 0.5).clamp(0, 1)
            save_image(imgs, OUT_DIR / f'samples_epoch{ep+1:03d}.png', nrow=8)
            model.train()
    return history


optimizer = torch.optim.Adam(model.parameters(), lr=LR)


if __name__ == '__main__':
    # ## 5. Treino
    history = train_vae(model, train_loader, optimizer, epochs=EPOCHS, beta=BETA)
    torch.save(model.state_dict(), OUT_DIR / 'vae_checkpoint.pth')
    print('Checkpoint guardado.')

    # ## 6. Curvas de treino
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ['loss', 'recon', 'kl']):
        ax.plot([h[key] for h in history])
        ax.set_title(key.capitalize())
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'training_curves.png')
    # plt.show()

    # ## 7. Amostras geradas
    model.eval()
    with torch.no_grad():
        z     = torch.randn(64, LATENT_DIM, device=device)
        imgs_ = (model.decode(z) * 0.5 + 0.5).clamp(0, 1).cpu()

    grid = make_grid(imgs_, nrow=8).permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.title('VAE — amostras geradas')
    plt.axis('off')
    plt.savefig(OUT_DIR / 'generated_samples.png', bbox_inches='tight')
    # plt.show()

    # ## 8. Reconstruções (real vs gerado)
    model.eval()
    x_real, _ = next(iter(train_loader))
    x_real = x_real[:8].to(device)
    with torch.no_grad():
        x_recon, _, _ = model(x_real)

    def denorm(t): return (t * 0.5 + 0.5).clamp(0, 1).cpu()

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i in range(8):
        axes[0, i].imshow(denorm(x_real[i]).permute(1,2,0))
        axes[1, i].imshow(denorm(x_recon[i]).permute(1,2,0))
        axes[0, i].axis('off'); axes[1, i].axis('off')
    axes[0, 0].set_ylabel('real', fontsize=9)
    axes[1, 0].set_ylabel('recon', fontsize=9)
    plt.suptitle('VAE — reconstruções')
    plt.tight_layout()
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'reconstructions.png', bbox_inches='tight')
    # plt.show()

    # ## 9. Interpolação no espaço latente
    model.eval()
    with torch.no_grad():
        z0     = torch.randn(1, LATENT_DIM, device=device)
        z1     = torch.randn(1, LATENT_DIM, device=device)
        alphas = torch.linspace(0, 1, 10, device=device).unsqueeze(1)
        zs     = (1 - alphas) * z0 + alphas * z1
        imgs_  = (model.decode(zs) * 0.5 + 0.5).clamp(0, 1).cpu()

    grid = make_grid(imgs_, nrow=10).permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 2))
    plt.imshow(grid)
    plt.title('VAE — interpolação no espaço latente')
    plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'latent_interpolation.png', bbox_inches='tight')
    # plt.show()

