#!/usr/bin/env python
# coding: utf-8
# 06 — cDCGAN (Conditional DCGAN)
# Baseado em 02_dcgan.py. O gerador e discriminador recebem label como condição.
# Spectral Norm + Cosine LR sempre activos.

from __future__ import annotations
import sys, random, csv
from pathlib import Path
import os, multiprocessing, warnings
warnings.filterwarnings('ignore')
if multiprocessing.current_process().name != 'MainProcess':
    sys.stdout = open(os.devnull, 'w')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / 'TP' / 'TP1-alunos-src-only' / 'scripts'
SUBSET_CSV  = REPO_ROOT / 'TP' / 'TP1-alunos-src-only' / 'student_start_pack' / 'training_20_percent.csv'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print('Device:', device)


# ── Hiperparâmetros ──────────────────────────────────────────────────────────

IMAGE_SIZE = 32
N_CLASSES  = 10   # ArtBench-10
LATENT_DIM = int(os.environ.get('DCGAN_LATENT', 100))
EMBED_DIM  = int(os.environ.get('CDCGAN_EMBED', 32))   # dimensão do label embedding em G
NGF        = int(os.environ.get('DCGAN_NGF', 64))
NDF        = int(os.environ.get('DCGAN_NDF', 64))
BATCH_SIZE = 128
LR_G       = float(os.environ.get('DCGAN_LR_G', os.environ.get('DCGAN_LR', 2e-4)))
LR_D       = float(os.environ.get('DCGAN_LR_D', os.environ.get('DCGAN_LR', 2e-4)))
BETA1      = float(os.environ.get('DCGAN_BETA1', 0.5))

from config import cfg
EPOCHS     = int(os.environ.get('DCGAN_EPOCHS', cfg.dcgan_epochs))
USE_SUBSET = cfg.use_subset

EXP_NAME = os.environ.get('EXP_NAME', 'cdcgan')
OUT_DIR  = REPO_ROOT / 'results' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_NOISE  = torch.randn(64, LATENT_DIM, device=device)
FIXED_LABELS = torch.tensor([i % N_CLASSES for i in range(64)], device=device)

print(f'cDCGAN | latent={LATENT_DIM} embed={EMBED_DIM} ngf={NGF} ndf={NDF} classes={N_CLASSES}')
print(f'Spectral Norm (D) + Cosine LR: sempre activos')


# ── Dataset ───────────────────────────────────────────────────────────────────

def safe_num_workers(): return 4

class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds = hf_split; self.transform = transform
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
    T.CenterCrop(IMAGE_SIZE), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
])

from datasets import load_dataset, load_from_disk
DATA_CACHE = REPO_ROOT / 'data' / 'artbench10_hf'
if DATA_CACHE.exists():
    train_hf = load_from_disk(str(DATA_CACHE))['train']
else:
    ds = load_dataset('zguo0525/ArtBench')
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATA_CACHE))
    train_hf = ds['train']

subset_ids   = load_ids_from_csv(SUBSET_CSV) if USE_SUBSET else None
train_ds     = HFDatasetTorch(train_hf, transform, indices=subset_ids)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=safe_num_workers(), pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)
print(f'Amostras: {len(train_ds)} | Batches: {len(train_loader)}')


# ── Arquitectura ─────────────────────────────────────────────────────────────

class cDCGenerator(nn.Module):
    """G(z, label) → imagem. Label embedding concatenado com z antes das ConvTranspose."""
    def __init__(self, latent_dim=100, n_classes=10, embed_dim=32, image_channels=3, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb  = nn.Embedding(n_classes, embed_dim)
        in_ch = latent_dim + embed_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch,  ngf*4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4,  ngf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2,  ngf,   4, 2, 1, bias=False), nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)                          # (B, embed_dim)
        inp = torch.cat([z, emb], dim=1)                      # (B, latent_dim + embed_dim)
        return self.net(inp.view(inp.size(0), -1, 1, 1))


class cDCDiscriminator(nn.Module):
    """D(image, label) → score. Label projetado como canal extra (spatial map 32x32)."""
    def __init__(self, n_classes=10, image_channels=3, ndf=64):
        super().__init__()
        # label → mapa espacial de 1 canal (aplicado antes da primeira conv)
        self.label_emb = nn.Sequential(
            nn.Embedding(n_classes, ndf),
            nn.Linear(ndf, IMAGE_SIZE * IMAGE_SIZE),
        )
        # imagem (3ch) + label map (1ch) = 4 canais de entrada; SN aplicado após init
        self.net = nn.Sequential(
            nn.Conv2d(image_channels + 1, ndf,   4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,   ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1,     4, 1, 0, bias=False), nn.Sigmoid(),
        )

    def forward(self, x, labels):
        label_map = self.label_emb(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        return self.net(torch.cat([x, label_map], dim=1)).view(-1, 1)


def init_dcgan_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def _apply_sn(m):
    if isinstance(m, nn.Conv2d):
        nn.utils.spectral_norm(m)


generator     = cDCGenerator(LATENT_DIM, N_CLASSES, EMBED_DIM, 3, NGF).to(device).apply(init_dcgan_weights)
discriminator = cDCDiscriminator(N_CLASSES, 3, NDF).to(device).apply(init_dcgan_weights)
discriminator.apply(_apply_sn)

print('Generator params    :', sum(p.numel() for p in generator.parameters()))
print('Discriminator params:', sum(p.numel() for p in discriminator.parameters()))


# ── Loop de treino ────────────────────────────────────────────────────────────

def train_cgan(generator, discriminator, loader, latent_dim, epochs=100):
    criterion = nn.BCELoss()
    opt_g   = torch.optim.Adam(generator.parameters(),     lr=LR_G, betas=(BETA1, 0.999))
    opt_d   = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    history = {'g_loss': [], 'd_loss': []}
    generator.train(); discriminator.train()

    for epoch in range(epochs):
        g_run = d_run = n_batches = 0

        for real, labels in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            real   = real.to(device)
            labels = labels.to(device)
            bs     = real.size(0)
            real_t = torch.ones(bs,  1, device=device)
            fake_t = torch.zeros(bs, 1, device=device)

            # ── Discriminator ─────────────────────────────────────────────────
            opt_d.zero_grad(set_to_none=True)
            d_loss_real = criterion(discriminator(real, labels), real_t)
            z           = torch.randn(bs, latent_dim, device=device)
            fake_labels = torch.randint(0, N_CLASSES, (bs,), device=device)
            fake        = generator(z, fake_labels).detach()
            d_loss_fake = criterion(discriminator(fake, fake_labels), fake_t)
            d_loss      = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # ── Generator ─────────────────────────────────────────────────────
            opt_g.zero_grad(set_to_none=True)
            z           = torch.randn(bs, latent_dim, device=device)
            fake_labels = torch.randint(0, N_CLASSES, (bs,), device=device)
            fake        = generator(z, fake_labels)
            g_loss      = criterion(discriminator(fake, fake_labels), real_t)
            g_loss.backward()
            opt_g.step()

            g_run += g_loss.detach(); d_run += d_loss.detach(); n_batches += 1

        nb    = max(n_batches, 1)
        to_f  = lambda t: t.item() if hasattr(t, 'item') else float(t)
        history['g_loss'].append(to_f(g_run / nb))
        history['d_loss'].append(to_f(d_run / nb))
        print(f'Epoch {epoch+1:03d}/{epochs} | D: {history["d_loss"][-1]:.4f} | G: {history["g_loss"][-1]:.4f}')
        sched_g.step(); sched_d.step()

        if cfg.save_samples and (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                imgs = (generator(FIXED_NOISE, FIXED_LABELS) * 0.5 + 0.5).clamp(0, 1)
            save_image(imgs, OUT_DIR / f'samples_epoch{epoch+1:03d}.png', nrow=8)
            generator.train()

    return history


# ── Treino ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    history = train_cgan(generator, discriminator, train_loader, LATENT_DIM, epochs=EPOCHS)

    torch.save({
        'generator'    : generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'history'      : history,
        'config'       : {
            'latent_dim': LATENT_DIM, 'n_classes': N_CLASSES, 'embed_dim': EMBED_DIM,
            'channels': 3, 'image_size': IMAGE_SIZE, 'model': 'cdcgan',
        },
    }, OUT_DIR / 'cdcgan_checkpoint.pt')
    print('Checkpoint guardado.')


# ── Curvas de treino ──────────────────────────────────────────────────────────

def plot_losses(history, title='cDCGAN losses'):
    plt.figure(figsize=(7, 4))
    plt.plot(history['d_loss'], label='Discriminator')
    plt.plot(history['g_loss'], label='Generator')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'training_curves.png', bbox_inches='tight')

if __name__ == '__main__':
    plot_losses(history, title='ArtBench cDCGAN losses')


# ── Inferência ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(generator, latent_dim, n_samples=64, seed=123, title='cDCGAN inference'):
    """Gera amostras com labels cíclicos (6-7 por classe)."""
    torch.manual_seed(seed)
    generator.eval()
    z      = torch.randn(n_samples, latent_dim, device=device)
    labels = torch.tensor([i % N_CLASSES for i in range(n_samples)], device=device)
    fake   = (generator(z, labels) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid   = make_grid(fake, nrow=8).permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 12)); plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'generated_samples.png', bbox_inches='tight')


@torch.no_grad()
def latent_walk(generator, latent_dim, steps=10, label=0, title='cDCGAN latent interpolation'):
    """Interpola no espaço latente mantendo a classe fixa."""
    generator.eval()
    z0     = torch.randn(1, latent_dim, device=device)
    z1     = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    zs     = (1 - alphas) * z0 + alphas * z1
    labels = torch.full((steps,), label, dtype=torch.long, device=device)
    imgs   = (generator(zs, labels) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid   = make_grid(imgs, nrow=steps).permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 2)); plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'latent_interpolation.png', bbox_inches='tight')


if __name__ == '__main__':
    run_inference(generator, LATENT_DIM, n_samples=64, seed=123, title='ArtBench cDCGAN inference')
    latent_walk(generator, LATENT_DIM, steps=10, label=0, title='ArtBench cDCGAN latent interpolation (class 0)')
