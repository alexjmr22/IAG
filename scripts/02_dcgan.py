#!/usr/bin/env python
# coding: utf-8

# # 02 — DCGAN
# 
# Baseado nas classes `DCGenerator` / `DCDiscriminator` e na função `train_gan` do notebook  
# **4 - Vanilla GANs (MNIST and CIFAR-10)**, adaptadas para o ArtBench-10 (3 canais, 32×32).
# 
# ### Checklist
# - [ ] Generator: z (100×1×1) → imagem (3×32×32)
# - [ ] Discriminator: imagem → score real/fake
# - [ ] Loop adversarial BCE
# - [ ] Guardar checkpoint para avaliação FID/KID
# 
# ### Extensões (bónus)
# - WGAN-GP (Wasserstein + gradient penalty)
# - cDCGAN condicionado ao estilo artístico
# - Mais épocas no dataset completo

# In[ ]:


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


def _apply_sn(m):
    if isinstance(m, nn.Conv2d):
        nn.utils.spectral_norm(m)


# ## 1. Hiperparâmetros

# In[ ]:


import os
IMAGE_SIZE  = 32
LATENT_DIM  = int(os.environ.get('DCGAN_LATENT', 100))
NGF         = int(os.environ.get('DCGAN_NGF', 64))     # feature maps Generator
NDF         = int(os.environ.get('DCGAN_NDF', 64))     # feature maps Discriminator
BATCH_SIZE  = 128    # conforme Notebooks 3, 4, 5
LR_G        = float(os.environ.get('DCGAN_LR_G', os.environ.get('DCGAN_LR', 2e-4)))
LR_D        = float(os.environ.get('DCGAN_LR_D', os.environ.get('DCGAN_LR', 2e-4)))
BETA1        = float(os.environ.get('DCGAN_BETA1', 0.5))
USE_SPECTRAL = os.environ.get('DCGAN_SPECTRAL', '0') == '1'
USE_COSINE   = os.environ.get('DCGAN_COSINE',   '0') == '1'

from config import cfg
EPOCHS      = int(os.environ.get('DCGAN_EPOCHS', cfg.dcgan_epochs))
USE_SUBSET  = cfg.use_subset

EXP_NAME = os.environ.get('EXP_NAME', 'dcgan')
OUT_DIR = REPO_ROOT / 'results' / 'gan' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ruído fixo para visualização consistente entre épocas
FIXED_NOISE = torch.randn(64, LATENT_DIM, device=device)


# ## 2. Dataset

# In[ ]:


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


# ## 3. Arquitectura — `DCGenerator` / `DCDiscriminator` (notebook 4)
# 
# Cópia directa das classes do notebook 4, com `image_channels=3` para ArtBench.

# In[ ]:


class DCGenerator(nn.Module):
    """z (latent_dim,) → imagem (3, 32, 32) — notebook 4."""
    def __init__(self, latent_dim=100, image_channels=3, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf,     4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),     nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class DCDiscriminator(nn.Module):
    """imagem (3, 32, 32) → score — notebook 4."""
    def __init__(self, image_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, ndf,     4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,     ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def init_dcgan_weights(m):
    """Inicialização de pesos DCGAN — notebook 4."""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


generator     = DCGenerator(LATENT_DIM, image_channels=3, ngf=NGF).to(device).apply(init_dcgan_weights)
discriminator = DCDiscriminator(image_channels=3, ndf=NDF).to(device).apply(init_dcgan_weights)
if USE_SPECTRAL:
    discriminator.apply(_apply_sn)

print('Generator params    :', sum(p.numel() for p in generator.parameters()))
print('Discriminator params:', sum(p.numel() for p in discriminator.parameters()))
print(f'Spectral Norm (D): {USE_SPECTRAL} | Cosine LR: {USE_COSINE}')


# ## 4. Loop de treino adversarial — `train_gan` (notebook 4)

# In[ ]:


def train_gan(generator, discriminator, loader, latent_dim, epochs=100):
    """Loop adversarial BCE — retirado do notebook 4."""
    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(),     lr=LR_G, betas=(BETA1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs) if USE_COSINE else None
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs) if USE_COSINE else None

    history = {'g_loss': [], 'd_loss': []}
    generator.train(); discriminator.train()

    for epoch in range(epochs):
        g_run = d_run = n_batches = 0

        for real, _ in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            real = real.to(device)
            bs   = real.size(0)
            real_targets = torch.ones(bs,  1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            # ── Discriminator update ──────────────────────────────────────────
            opt_d.zero_grad(set_to_none=True)
            d_loss_real = criterion(discriminator(real), real_targets)
            z           = torch.randn(bs, latent_dim, device=device)
            fake        = generator(z)
            d_loss_fake = criterion(discriminator(fake.detach()), fake_targets)
            d_loss      = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # ── Generator update ──────────────────────────────────────────────
            opt_g.zero_grad(set_to_none=True)
            z      = torch.randn(bs, latent_dim, device=device)
            fake   = generator(z)
            g_loss = criterion(discriminator(fake), real_targets)
            g_loss.backward()
            opt_g.step()

            g_run += g_loss.detach()
            d_run += d_loss.detach()
            n_batches += 1

        history['g_loss'].append((g_run / max(n_batches, 1)).item() if hasattr(g_run, 'item') else g_run / max(n_batches, 1))
        history['d_loss'].append((d_run / max(n_batches, 1)).item() if hasattr(d_run, 'item') else d_run / max(n_batches, 1))
        print(f'Epoch {epoch+1:03d}/{epochs} | D loss: {history["d_loss"][-1]:.4f} | G loss: {history["g_loss"][-1]:.4f}')
        if USE_COSINE:
            sched_g.step(); sched_d.step()

        # amostras intermédias
        if cfg.save_samples and (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                imgs = (generator(FIXED_NOISE) * 0.5 + 0.5).clamp(0, 1)
            save_image(imgs, OUT_DIR / f'samples_epoch{epoch+1:03d}.png', nrow=8)
            generator.train()

    return history


# ## 5. Treino

# In[ ]:


if __name__ == '__main__':
    history = train_gan(generator, discriminator, train_loader, LATENT_DIM, epochs=EPOCHS)

    # guardar checkpoint (mesmo formato do notebook 4)
    torch.save({
        'generator'    : generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'history'      : history,
        'config'       : {'latent_dim': LATENT_DIM, 'channels': 3, 'image_size': IMAGE_SIZE},
    }, OUT_DIR / 'dcgan_checkpoint.pt')
    print('Checkpoint guardado.')


# ## 6. Curvas de treino

# In[ ]:


def plot_gan_losses(history, title='GAN losses'):
    plt.figure(figsize=(7, 4))
    plt.plot(history['d_loss'], label='Discriminator')
    plt.plot(history['g_loss'], label='Generator')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'training_curves.png', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    plot_gan_losses(history, title='ArtBench DCGAN losses')


# ## 7. Inferência — `run_inference` / `latent_walk` (notebook 4)

# In[ ]:


@torch.no_grad()
def run_inference(generator, latent_dim, n_samples=64, seed=123, title='Generator inference'):
    """Gera amostras aleatórias — notebook 4."""
    torch.manual_seed(seed)
    generator.eval()
    z    = torch.randn(n_samples, latent_dim, device=device)
    fake = (generator(z) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid = make_grid(fake, nrow=8).permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'generated_samples.png', bbox_inches='tight')
    # plt.show()


@torch.no_grad()
def latent_walk(generator, latent_dim, steps=10, title='Latent interpolation'):
    """Interpolação entre dois pontos latentes — notebook 4."""
    generator.eval()
    z0     = torch.randn(1, latent_dim, device=device)
    z1     = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    zs     = (1 - alphas) * z0 + alphas * z1
    imgs_  = (generator(zs) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid   = make_grid(imgs_, nrow=steps).permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 2))
    plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'latent_interpolation.png', bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    run_inference(generator, LATENT_DIM, n_samples=64, seed=123, title='ArtBench DCGAN inference')
    latent_walk(generator, LATENT_DIM, steps=10, title='ArtBench DCGAN latent interpolation')

