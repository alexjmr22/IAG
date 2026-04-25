#!/usr/bin/env python
# coding: utf-8
# 05 — WGAN-GP
# Wasserstein GAN + Gradient Penalty (Gulrajani et al., 2017)
# Baseia-se em 02_dcgan.py: mesmo DCGenerator, mesmo dataset.
# Diferenças: WGANCritic (sem Sigmoid, sem BN), loss Wasserstein, n_critic,
#             Spectral Norm + Cosine LR sempre activos.

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
LATENT_DIM = int(os.environ.get('DCGAN_LATENT', 100))
NGF        = int(os.environ.get('DCGAN_NGF', 64))
NDF        = int(os.environ.get('DCGAN_NDF', 64))
BATCH_SIZE = 128
# WGAN-GP paper: lr=1e-4, Adam(0.0, 0.9)
LR_G       = float(os.environ.get('DCGAN_LR_G', os.environ.get('DCGAN_LR', 1e-4)))
LR_D       = float(os.environ.get('DCGAN_LR_D', os.environ.get('DCGAN_LR', 1e-4)))
BETA1      = float(os.environ.get('DCGAN_BETA1', 0.0))
BETA2      = float(os.environ.get('DCGAN_BETA2', 0.9))
LAMBDA_GP  = float(os.environ.get('WGAN_LAMBDA', 10.0))
N_CRITIC   = int(os.environ.get('WGAN_N_CRITIC', 5))
USE_COSINE = os.environ.get('WGAN_COSINE', '1') == '1'

from config import cfg
EPOCHS     = int(os.environ.get('DCGAN_EPOCHS', cfg.dcgan_epochs))
USE_SUBSET = cfg.use_subset

EXP_NAME = os.environ.get('EXP_NAME', 'wgan_gp')
OUT_DIR  = REPO_ROOT / 'results' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_NOISE = torch.randn(64, LATENT_DIM, device=device)

print(f'WGAN-GP | latent={LATENT_DIM} ngf={NGF} ndf={NDF} λ={LAMBDA_GP} n_critic={N_CRITIC}')
print(f'lr_G={LR_G} lr_D={LR_D} β=({BETA1},{BETA2}) | Spectral Norm: ON | Cosine LR: {USE_COSINE}')


# ── Dataset (idêntico ao 02_dcgan.py) ────────────────────────────────────────

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

class DCGenerator(nn.Module):
    """Idêntico ao 02_dcgan.py — gerador sem alterações."""
    def __init__(self, latent_dim=100, image_channels=3, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class WGANCritic(nn.Module):
    """Critic Wasserstein: sem Sigmoid, sem BN (incompatível com GP), SN aplicado após init."""
    def __init__(self, image_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, ndf,   4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,   ndf*2, 4, 2, 1, bias=False),          nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),          nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1,     4, 1, 0, bias=False),
        )
    def forward(self, x):
        return self.net(x).view(-1, 1)


def init_weights(m):
    if 'Conv' in m.__class__.__name__:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in m.__class__.__name__:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def _apply_sn(m):
    if isinstance(m, nn.Conv2d):
        nn.utils.spectral_norm(m)


generator = DCGenerator(LATENT_DIM, image_channels=3, ngf=NGF).to(device).apply(init_weights)
critic     = WGANCritic(image_channels=3, ndf=NDF).to(device).apply(init_weights)
critic.apply(_apply_sn)

print('Generator params:', sum(p.numel() for p in generator.parameters()))
print('Critic params   :', sum(p.numel() for p in critic.parameters()))


# ── Gradient Penalty ──────────────────────────────────────────────────────────

def gradient_penalty(critic, real, fake):
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    d_interp = critic(interp)
    grads = torch.autograd.grad(
        d_interp, interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()


# ── Loop de treino ────────────────────────────────────────────────────────────

def train_wgan(generator, critic, loader, latent_dim, epochs=100):
    opt_g   = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_d   = torch.optim.Adam(critic.parameters(),    lr=LR_D, betas=(BETA1, BETA2))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs) if USE_COSINE else None
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs) if USE_COSINE else None

    history = {'g_loss': [], 'd_loss': [], 'w_dist': []}
    generator.train(); critic.train()

    for epoch in range(epochs):
        g_run = d_run = w_run = n_batches = 0

        for real, _ in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            real = real.to(device)
            bs   = real.size(0)

            # ── Critic: N_CRITIC passos por passo de G ────────────────────────
            for _ in range(N_CRITIC):
                z    = torch.randn(bs, latent_dim, device=device)
                fake = generator(z).detach()

                opt_d.zero_grad(set_to_none=True)
                d_real = critic(real).mean()
                d_fake = critic(fake).mean()
                gp     = gradient_penalty(critic, real, fake)
                d_loss = d_fake - d_real + LAMBDA_GP * gp
                d_loss.backward()
                opt_d.step()

            # ── Generator: 1 passo ────────────────────────────────────────────
            opt_g.zero_grad(set_to_none=True)
            z    = torch.randn(bs, latent_dim, device=device)
            fake = generator(z)
            g_loss = -critic(fake).mean()
            g_loss.backward()
            opt_g.step()

            w_run   += (d_real - d_fake).detach()
            g_run   += g_loss.detach()
            d_run   += d_loss.detach()
            n_batches += 1

        nb = max(n_batches, 1)
        to_f = lambda t: t.item() if hasattr(t, 'item') else float(t)
        history['g_loss'].append(to_f(g_run / nb))
        history['d_loss'].append(to_f(d_run / nb))
        history['w_dist'].append(to_f(w_run / nb))

        print(f'Epoch {epoch+1:03d}/{epochs} | W-dist: {history["w_dist"][-1]:.4f} | G: {history["g_loss"][-1]:.4f}')
        if USE_COSINE:
            sched_g.step(); sched_d.step()

        if cfg.save_samples and (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                imgs = (generator(FIXED_NOISE) * 0.5 + 0.5).clamp(0, 1)
            save_image(imgs, OUT_DIR / f'samples_epoch{epoch+1:03d}.png', nrow=8)
            generator.train()

    return history


# ── Treino ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    history = train_wgan(generator, critic, train_loader, LATENT_DIM, epochs=EPOCHS)

    # checkpoint compatível com 04_evaluation.py (EVAL_TARGET='DCGAN')
    torch.save({
        'generator'    : generator.state_dict(),
        'discriminator': critic.state_dict(),
        'history'      : history,
        'config'       : {
            'latent_dim': LATENT_DIM, 'channels': 3, 'image_size': IMAGE_SIZE,
            'model': 'wgan_gp', 'lambda_gp': LAMBDA_GP, 'n_critic': N_CRITIC,
        },
    }, OUT_DIR / 'dcgan_checkpoint.pt')
    print('Checkpoint guardado.')


# ── Curvas de treino ──────────────────────────────────────────────────────────

def plot_wgan_losses(history, title='WGAN-GP losses'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['w_dist'], label='W-distance'); ax1.set_title('Wasserstein Distance (↑)'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['g_loss'], label='G loss');     ax2.set_title('Generator Loss (↓)');       ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(title)
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'training_curves.png', bbox_inches='tight')

if __name__ == '__main__':
    plot_wgan_losses(history, title='ArtBench WGAN-GP losses')


# ── Inferência ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(generator, latent_dim, n_samples=64, seed=123, title='WGAN-GP inference'):
    torch.manual_seed(seed)
    generator.eval()
    z    = torch.randn(n_samples, latent_dim, device=device)
    fake = (generator(z) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid = make_grid(fake, nrow=8).permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 12)); plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'generated_samples.png', bbox_inches='tight')

@torch.no_grad()
def latent_walk(generator, latent_dim, steps=10, title='WGAN-GP latent interpolation'):
    generator.eval()
    z0 = torch.randn(1, latent_dim, device=device)
    z1 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    zs   = (1 - alphas) * z0 + alphas * z1
    imgs = (generator(zs) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid = make_grid(imgs, nrow=steps).permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 2)); plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'latent_interpolation.png', bbox_inches='tight')

if __name__ == '__main__':
    run_inference(generator, LATENT_DIM, n_samples=64, seed=123, title='ArtBench WGAN-GP inference')
    latent_walk(generator, LATENT_DIM, steps=10, title='ArtBench WGAN-GP latent interpolation')
