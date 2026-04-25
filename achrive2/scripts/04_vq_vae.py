#!/usr/bin/env python
# coding: utf-8

# # 04 — Vector Quantized Variational Autoencoder (VQ-VAE)
# 
# Variante do VAE com latent space DISCRETO em vez de contínuo.
# 
# **Diferenças vs VAE:**
# - Encoder: x → e (encoded)
# - Quantization: e → nearest codebook vector (discreto!)
# - Decoder: z (quantized) → x reconstruído
# - Loss: reconstrução + commitment loss + codebook loss
# 
# **Vantagens:**
# - Sem KL collapse → maior liberdade do encoder
# - Latent discreto = melhor para estruturas artísticas
# - Tipicamente 15-20% FID improvement vs β-VAE

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
NUM_EMBEDDINGS = int(os.environ.get('VQ_NUM_EMBEDDINGS', 256))  # tamanho do codebook
BATCH_SIZE  = 128
LR          = float(os.environ.get('VAE_LR', 2e-3))
COMMITMENT_LOSS_WEIGHT = 0.25  # peso do commitment loss na perda total

from config import cfg
EPOCHS      = cfg.vae_epochs
USE_SUBSET  = cfg.use_subset

# ── Schedulers ────────────────────────────────────────────────────────────────
USE_COSINE_LR = os.environ.get('VAE_COSINE_LR', 'false').lower() == 'true'
KL_ANNEALING_EPOCHS = int(os.environ.get('VAE_KL_ANNEALING_EPOCHS', 0))

EXP_NAME = os.environ.get('EXP_NAME', 'vq_vae')
OUT_DIR = REPO_ROOT / 'results' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOG DE PARÂMETROS (.md) ──────────────────────────────────────────────────
params_md = f"""# VQ-VAE Experiment: {EXP_NAME}
- **Architecture**: Vector Quantized VAE
- **Latent Dim**: {LATENT_DIM}
- **Num Embeddings (Codebook)**: {NUM_EMBEDDINGS}
- **Commitment Loss Weight**: {COMMITMENT_LOSS_WEIGHT}
- **Learning Rate**: {LR}
- **Epochs**: {EPOCHS}
- **Batch Size**: {BATCH_SIZE}
- **Dataset**: {"20% Subset" if USE_SUBSET else "Full ArtBench10"}
- **Cosine Annealing LR**: {USE_COSINE_LR}
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


# ## 3. Arquitectura VQ-VAE

class VectorQuantizer(nn.Module):
    """Quantização Vetorial — mapeia contínuo para discreto."""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Codebook: (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        """
        inputs: (B, C, ...)
        outputs: (B, C, ...), indices, loss
        """
        input_shape = inputs.shape
        flat = inputs.view(-1, self.embedding_dim)  # (N, C)
        
        # Distância euclidiana até cada embedding
        # ||x - e_i||^2 = ||x||^2 + ||e_i||^2 - 2*<x, e_i>
        distances = (torch.sum(flat**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat, self.embedding.weight.t()))
        
        # Índice do embedding mais próximo
        indices = torch.argmin(distances, dim=1)  # (N,)
        quantized = self.embedding(indices).view(input_shape)  # (B, C, ...)
        
        # Perda: commitment loss (encoder aproxima-se do codebook)
        # e_commitment loss (codebook aproxima-se do encoder)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + COMMITMENT_LOSS_WEIGHT * e_latent_loss
        
        # Straight-through estimator: durante backprop, deixa passar gradientes
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, indices


class VQVAE(nn.Module):
    def __init__(self, latent_dim=128, num_embeddings=256):
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
        self.enc_to_latent = nn.Conv2d(128, latent_dim, 1, 1, 0)
        
        # ── Vector Quantizer ──────────────────────────────────────────────────
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        
        # ── Decoder (4×4 → 32×32) ─────────────────────────────────────────────
        self.dec_from_latent = nn.Conv2d(latent_dim, 128, 1, 1, 0)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4→8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 8→16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 16→32
            nn.Tanh(),
        )
    
    def encode(self, x):
        """Encode: x → z (contínuo antes da VQ)."""
        h = self.enc_conv(x)
        z = self.enc_to_latent(h)
        return z
    
    def decode(self, z_q):
        """Decode: z_q (quantizado) → x."""
        h = self.dec_from_latent(z_q)
        return self.dec_conv(h)
    
    def forward(self, x):
        """Forward: VQ-VAE completo."""
        z = self.encode(x)
        z_q, vq_loss, indices = self.vq(z)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices


model = VQVAE(LATENT_DIM, NUM_EMBEDDINGS).to(device)
print(f'Parâmetros: {sum(p.numel() for p in model.parameters()):,}')


# ## 4. Loss e otimizador

def vqvae_loss(xhat, x, vq_loss):
    """VQ-VAE Loss = reconstrução + vq_loss."""
    recon = F.mse_loss(xhat, x)
    return recon + vq_loss, recon, vq_loss


def train_vqvae(model, loader, optimizer, epochs=50, use_cosine_lr=False):
    """Loop de treino do VQ-VAE."""
    model.train()
    history = []
    
    # ── Scheduler (opcional) ──────────────────────────────────────────────────
    if use_cosine_lr:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        scheduler = None
    
    for ep in range(epochs):
        tl = tr = tvq = 0.0
        
        for x, _ in tqdm(loader, desc=f'Epoch {ep+1}/{epochs}', leave=False):
            x = x.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            xhat, vq_loss, _ = model(x)
            loss, recon, vq = vqvae_loss(xhat, x, vq_loss)
            loss.backward()
            optimizer.step()
            
            tl += loss.detach() * x.size(0)
            tr += recon.detach() * x.size(0)
            tvq += vq.detach() * x.size(0)
            
        n = len(loader.dataset)
        
        epoch_loss  = (tl/n).item()
        epoch_recon = (tr/n).item()
        epoch_vq    = (tvq/n).item()
        
        history.append({'loss': epoch_loss, 'recon': epoch_recon, 'vq': epoch_vq})
        print(f'Epoch {ep+1:03d}/{epochs} | loss={epoch_loss:.4f}  recon={epoch_recon:.4f}  vq={epoch_vq:.6f}')
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Amostras intermédias
        if cfg.save_samples and (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                z = torch.randn(64, LATENT_DIM, 4, 4, device=device)
                imgs = (model.decode(z) * 0.5 + 0.5).clamp(0, 1)
            save_image(imgs, OUT_DIR / f'samples_epoch{ep+1:03d}.png', nrow=8)
            model.train()
    return history


optimizer = torch.optim.Adam(model.parameters(), lr=LR)


if __name__ == '__main__':
    # ## 5. Treino
    history = train_vqvae(model, train_loader, optimizer, epochs=EPOCHS, use_cosine_lr=USE_COSINE_LR)
    torch.save(model.state_dict(), OUT_DIR / 'vq_vae_checkpoint.pth')
    print('Checkpoint guardado.')

    # ## 6. Curvas de treino
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ['loss', 'recon', 'vq']):
        ax.plot([h[key] for h in history])
        ax.set_title(key.capitalize())
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'training_curves.png')

    # ## 7. Amostras geradas (aleatórias)
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, LATENT_DIM, 4, 4, device=device)
        imgs = (model.decode(z) * 0.5 + 0.5).clamp(0, 1).cpu()

    grid = make_grid(imgs, nrow=8).permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.title('VQ-VAE — amostras geradas (latent aleatório)')
    plt.axis('off')
    plt.savefig(OUT_DIR / 'generated_samples.png', bbox_inches='tight')

    # ## 8. Reconstruções
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
    plt.suptitle('VQ-VAE — reconstruções')
    plt.tight_layout()
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'reconstructions.png', bbox_inches='tight')

    print('VQ-VAE training complete!')
