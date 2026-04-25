#!/usr/bin/env python
# coding: utf-8

# # 03 — Conditional Variational Autoencoder (CVAE)
# 
# Extensão do VAE padrão para geração condicionada aos 10 estilos artísticos de ArtBench.
# 
# **Diferenças vs VAE:**
# - Encoder: recebe x AND classe c → (μ, log σ²)
# - Decoder: recebe z AND classe c → x reconstruído
# - Loss: idêntico (MSE + β·KL)
# 
# Permite:
# - Geração controlada por estilo: decoder(z, c="Impressionism") → imagem Impressionista
# - Reduz mode collapse
# - Melhor qualidade FID/KID

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

import os
IMAGE_SIZE  = 32
LATENT_DIM  = int(os.environ.get('VAE_LATENT_DIM', 128))
BETA        = float(os.environ.get('VAE_BETA', 0.1))
BATCH_SIZE  = 128
LR          = float(os.environ.get('VAE_LR', 2e-3))
NUM_CLASSES = 10  # 10 estilos em ArtBench

from config import cfg
EPOCHS      = cfg.vae_epochs
USE_SUBSET  = cfg.use_subset

# ── Schedulers ────────────────────────────────────────────────────────────────
USE_COSINE_LR = os.environ.get('VAE_COSINE_LR', 'false').lower() == 'true'
KL_ANNEALING_EPOCHS = int(os.environ.get('VAE_KL_ANNEALING_EPOCHS', 0))

EXP_NAME = os.environ.get('EXP_NAME', 'cvae')
OUT_DIR = REPO_ROOT / 'results' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOG DE PARÂMETROS (.md) ──────────────────────────────────────────────────
params_md = f"""# CVAE Experiment: {EXP_NAME}
- **Architecture**: Conditional VAE
- **Latent Dim**: {LATENT_DIM}
- **Beta (KL)**: {BETA}
- **Learning Rate**: {LR}
- **Epochs**: {EPOCHS}
- **Batch Size**: {BATCH_SIZE}
- **Num Classes**: {NUM_CLASSES}
- **Dataset**: {"20% Subset" if USE_SUBSET else "Full ArtBench10"}
- **Cosine Annealing LR**: {USE_COSINE_LR}
- **KL Annealing Epochs**: {KL_ANNEALING_EPOCHS}
"""
with open(OUT_DIR / "experiment_params.md", "w", encoding="utf-8") as f:
    f.write(params_md)
print(f"Parâmetros guardados em {OUT_DIR / 'experiment_params.md'}")


# ## 2. Dataset

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


# ## 3. Arquitectura CVAE

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10, class_embedding_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim

        # ── Class Embedding ───────────────────────────────────────────────────
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        # ── Encoder (32×32 + class → 4×4) ────────────────────────────────────
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 32→16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8→4
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        # 128*4*4 + class_embedding_dim → latent_dim
        self.fc_mu     = nn.Linear(128 * 4 * 4 + class_embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 + class_embedding_dim, latent_dim)

        # ── Decoder (latent + class → 32×32) ──────────────────────────────────
        # latent_dim + class_embedding_dim → 128*4*4
        self.dec_fc   = nn.Linear(latent_dim + class_embedding_dim, 128 * 4 * 4)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4→8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 8→16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 16→32
            nn.Tanh(),
        )

    def encode(self, x, c):
        """Encode com informação de classe."""
        h = self.enc_conv(x).view(x.size(0), -1)
        c_emb = self.class_embedding(c)
        h = torch.cat([h, c_emb], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """Decode com informação de classe."""
        c_emb = self.class_embedding(c)
        z_c = torch.cat([z, c_emb], dim=1)
        h = self.dec_fc(z_c).view(-1, 128, 4, 4)
        return self.dec_conv(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


model = ConditionalVAE(LATENT_DIM, NUM_CLASSES).to(device)
print(f'Parâmetros: {sum(p.numel() for p in model.parameters()):,}')


# ## 4. Loss e otimizador

def cvae_loss(xhat, x, mu, logvar, beta=0.1):
    """CVAE Loss — idêntico ao VAE, só muda o forward."""
    recon = F.mse_loss(xhat, x, reduction='sum') / x.size(0)
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl


def get_kl_beta(epoch, warmup_epochs, final_beta=0.1):
    """Calcula beta com annealing linear nos primeiros warmup_epochs."""
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return final_beta
    return final_beta * (epoch / warmup_epochs)


def train_cvae(model, loader, optimizer, epochs=50, beta=0.1, use_cosine_lr=False, kl_warmup_epochs=0):
    """Loop de treino do CVAE."""
    model.train()
    history = []
    
    # ── Scheduler (opcional) ──────────────────────────────────────────────────
    if use_cosine_lr:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = None
    
    for ep in range(epochs):
        tl = tr = tk = 0.0
        current_beta = get_kl_beta(ep, kl_warmup_epochs, beta)
        
        for x, c in tqdm(loader, desc=f'Epoch {ep+1}/{epochs} (β={current_beta:.4f})', leave=False):
            x, c = x.to(device), c.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            xhat, mu, logvar = model(x, c)
            loss, recon, kl  = cvae_loss(xhat, x, mu, logvar, current_beta)
            loss.backward()
            optimizer.step()
            
            tl += loss.detach() * x.size(0)
            tr += recon.detach() * x.size(0)
            tk += kl.detach() * x.size(0)
            
        n = len(loader.dataset)
        
        epoch_loss  = (tl/n).item()
        epoch_recon = (tr/n).item()
        epoch_kl    = (tk/n).item()
        
        history.append({'loss': epoch_loss, 'recon': epoch_recon, 'kl': epoch_kl})
        print(f'Epoch {ep+1:03d}/{epochs} | loss={epoch_loss:.4f}  recon={epoch_recon:.4f}  kl={epoch_kl:.4f}')
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Amostras intermédias
        if cfg.save_samples and (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for class_id in range(min(5, NUM_CLASSES)):  # 5 primeiras classes
                    z = torch.randn(8, LATENT_DIM, device=device)
                    c = torch.full((8,), class_id, dtype=torch.long, device=device)
                    imgs = (model.decode(z, c) * 0.5 + 0.5).clamp(0, 1)
                    save_image(imgs, OUT_DIR / f'samples_epoch{ep+1:03d}_class{class_id}.png', nrow=8)
            model.train()
    return history


optimizer = torch.optim.Adam(model.parameters(), lr=LR)


if __name__ == '__main__':
    # ## 5. Treino
    history = train_cvae(model, train_loader, optimizer, epochs=EPOCHS, beta=BETA,
                        use_cosine_lr=USE_COSINE_LR, kl_warmup_epochs=KL_ANNEALING_EPOCHS)
    torch.save(model.state_dict(), OUT_DIR / 'cvae_checkpoint.pth')
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

    # ## 7. Amostras geradas (por classe)
    model.eval()
    fig, axes = plt.subplots(NUM_CLASSES, 8, figsize=(12, 2*NUM_CLASSES))
    with torch.no_grad():
        for class_id in range(NUM_CLASSES):
            z = torch.randn(8, LATENT_DIM, device=device)
            c = torch.full((8,), class_id, dtype=torch.long, device=device)
            imgs = (model.decode(z, c) * 0.5 + 0.5).clamp(0, 1).cpu()
            for i in range(8):
                axes[class_id, i].imshow(imgs[i].permute(1,2,0))
                axes[class_id, i].axis('off')
            axes[class_id, 0].set_ylabel(f'Class {class_id}', fontsize=9)
    plt.suptitle('CVAE — amostras geradas por classe')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'generated_samples_by_class.png', bbox_inches='tight')

    # ## 8. Reconstruções
    model.eval()
    x_real, c_real = next(iter(train_loader))
    x_real, c_real = x_real[:8].to(device), c_real[:8].to(device)
    with torch.no_grad():
        x_recon, _, _ = model(x_real, c_real)

    def denorm(t): return (t * 0.5 + 0.5).clamp(0, 1).cpu()

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i in range(8):
        axes[0, i].imshow(denorm(x_real[i]).permute(1,2,0))
        axes[1, i].imshow(denorm(x_recon[i]).permute(1,2,0))
        axes[0, i].axis('off'); axes[1, i].axis('off')
    axes[0, 0].set_ylabel('real', fontsize=9)
    axes[1, 0].set_ylabel('recon', fontsize=9)
    plt.suptitle('CVAE — reconstruções')
    plt.tight_layout()
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'reconstructions.png', bbox_inches='tight')

    print('CVAE training complete!')
