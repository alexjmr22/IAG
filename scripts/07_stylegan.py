#!/usr/bin/env python
# coding: utf-8
# 07 — StyleGAN (adaptado para 32×32)
# Karras et al. 2019 — "A Style-Based Generator Architecture for GANs"
#
# Diferenças face ao DCGAN:
#   - Mapping network: z → w (espaço latente intermédio)
#   - Synthesis network: começa de constante aprendida (não de z)
#   - AdaIN: estilo aplicado por camada em vez de uma vez na entrada
#   - Noise injection: ruído estocástico por resolução
#   - R1 regularization (Mescheder 2018) em vez de GP
#   - Discriminador: DCDiscriminator + SN + MiniBatch StdDev (ProGAN)
#
# Simplificações face ao paper original (1024px):
#   - w_dim = 128 (vs 512 no paper)
#   - Mapping: 4 camadas FC (vs 8)
#   - Resoluções: 4→8→16→32 (vs 4→1024)
#   - Sem progressive growing (desnecessário a 32×32)
#   - PixelNorm em z (como no paper)

from __future__ import annotations
import sys, random, csv, math
from pathlib import Path
import os, multiprocessing, warnings
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

IMAGE_SIZE   = 32
LATENT_DIM   = int(os.environ.get('DCGAN_LATENT', 100))
W_DIM        = int(os.environ.get('STYLEGAN_WDIM', 128))     # dimensão do espaço w
MAP_LAYERS   = int(os.environ.get('STYLEGAN_MAP_LAYERS', 4)) # camadas da mapping network
NGF          = int(os.environ.get('DCGAN_NGF', 64))          # base de canais do generator
NDF          = int(os.environ.get('DCGAN_NDF', 64))          # base de canais do discriminador
BATCH_SIZE   = 128
LR_G         = float(os.environ.get('DCGAN_LR_G', os.environ.get('DCGAN_LR', 2e-4)))
LR_D         = float(os.environ.get('DCGAN_LR_D', os.environ.get('DCGAN_LR', 2e-4)))
BETA1        = float(os.environ.get('DCGAN_BETA1', 0.0))     # paper: β₁=0 para R1
BETA2        = float(os.environ.get('DCGAN_BETA2', 0.99))
R1_GAMMA     = float(os.environ.get('STYLEGAN_R1_GAMMA', 10.0))
R1_EVERY     = int(os.environ.get('STYLEGAN_R1_EVERY', 16))  # R1 a cada N batches
STYLE_MIX    = float(os.environ.get('STYLEGAN_MIX_PROB', 0.9))  # prob. style mixing

from config import cfg
EPOCHS     = int(os.environ.get('DCGAN_EPOCHS', cfg.dcgan_epochs))
USE_SUBSET = cfg.use_subset

EXP_NAME = os.environ.get('EXP_NAME', 'stylegan')
OUT_DIR  = REPO_ROOT / 'results' / 'gan' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_NOISE = torch.randn(64, LATENT_DIM, device=device)

print(f'StyleGAN | latent={LATENT_DIM} w_dim={W_DIM} map_layers={MAP_LAYERS} ngf={NGF}')
print(f'lr_G={LR_G} lr_D={LR_D} β=({BETA1},{BETA2}) R1γ={R1_GAMMA} mix_prob={STYLE_MIX}')


# ── Dataset (idêntico aos outros scripts) ────────────────────────────────────

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


# ── Arquitectura — StyleGAN Generator ────────────────────────────────────────

class PixelNorm(nn.Module):
    """Normalização por pixel: x / sqrt(mean(x²) + ε). Aplica-se a z antes da mapping net."""
    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True) + 1e-8).sqrt()


class MappingNetwork(nn.Module):
    """
    z ∈ Z → w ∈ W via MLP com PixelNorm.
    Paper: 8 camadas FC, dim=512. Aqui: MAP_LAYERS camadas, dim=W_DIM.
    """
    def __init__(self, latent_dim, w_dim, n_layers):
        super().__init__()
        layers = [PixelNorm(), nn.Linear(latent_dim, w_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(w_dim, w_dim), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (Huang & Belongie 2017).
    Normaliza cada feature map para (0, 1) e aplica estilo (γ, β) derivado de w.
    """
    def forward(self, x, style):
        B, C, H, W = x.shape
        gamma = style[:, :C].view(B, C, 1, 1)
        beta  = style[:, C:].view(B, C, 1, 1)
        x_norm = F.instance_norm(x)
        return gamma * x_norm + beta


class StyleBlock(nn.Module):
    """
    Um bloco da synthesis network:
      Conv3×3 → Noise (B) → AdaIN(A, w) → LeakyReLU
    """
    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.conv        = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.adain       = AdaIN()
        self.style       = nn.Linear(w_dim, 2 * out_ch)   # afine A: w → (γ, β)
        self.noise_scale = nn.Parameter(torch.zeros(out_ch, 1, 1))  # escala B por canal
        self.act         = nn.LeakyReLU(0.2, inplace=True)
        nn.init.normal_(self.conv.weight, 0, 1)

    def forward(self, x, w):
        x     = self.conv(x)
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        x     = x + self.noise_scale * noise
        style = self.style(w)
        x     = self.adain(x, style)
        return self.act(x)


class StyleGenerator(nn.Module):
    """
    Synthesis network: constante aprendida → blocos de estilo → imagem.
    Resoluções: 4 → 8 → 16 → 32 (3 upsamplings, 2 blocos por resolução).

    Canais (ngf=64): 256 → 256 → 128 → 64 → 32
    Canais (ngf=128): 512 → 512 → 256 → 128 → 64
    """
    def __init__(self, latent_dim=100, w_dim=128, ngf=64, n_map_layers=4, image_channels=3):
        super().__init__()
        c = [min(ngf * 4, 512), min(ngf * 4, 512), min(ngf * 2, 256),
             min(ngf, 128), max(ngf // 2, 16)]

        self.mapping = MappingNetwork(latent_dim, w_dim, n_map_layers)

        # Constante aprendida 4×4 (paper: "learned constant input")
        self.const = nn.Parameter(torch.randn(1, c[0], 4, 4))

        # Blocos de estilo por resolução (2 por nível)
        self.blocks = nn.ModuleList([
            # 4×4
            StyleBlock(c[0], c[0], w_dim),
            StyleBlock(c[0], c[1], w_dim),
            # 8×8
            StyleBlock(c[1], c[2], w_dim),
            StyleBlock(c[2], c[2], w_dim),
            # 16×16
            StyleBlock(c[2], c[3], w_dim),
            StyleBlock(c[3], c[3], w_dim),
            # 32×32
            StyleBlock(c[3], c[4], w_dim),
            StyleBlock(c[4], c[4], w_dim),
        ])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(c[4], image_channels, 1),
            nn.Tanh(),
        )

    def forward(self, z, z2=None, mix_layer=None):
        """
        z:         latent principal
        z2:        segundo latent para style mixing (opcional)
        mix_layer: índice do bloco a partir do qual usar w2
        """
        w  = self.mapping(z)
        w2 = self.mapping(z2) if z2 is not None else None

        x = self.const.expand(z.size(0), -1, -1, -1)

        for i, block in enumerate(self.blocks):
            # style mixing: a partir de mix_layer usa w2
            w_cur = w2 if (w2 is not None and mix_layer is not None and i >= mix_layer) else w
            x = block(x, w_cur)
            # upsample após cada par de blocos (exceto o último)
            if i % 2 == 1 and i < len(self.blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.to_rgb(x)


# ── Discriminador — DCDiscriminator + SN + MiniBatch StdDev ─────────────────

class MiniBatchStdDev(nn.Module):
    """
    ProGAN / StyleGAN: adiciona canal extra com std entre amostras do batch.
    Melhora a diversidade ao penalizar amostras com baixa variância.
    """
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        B, C, H, W = x.shape
        G = min(self.group_size, B)
        # (G, B//G, C, H, W) → std por slot dentro de cada grupo
        y = x.view(G, -1, C, H, W)
        y = y - y.mean(0, keepdim=True)
        y = (y.pow(2).mean(0) + 1e-8).sqrt()   # (B//G, C, H, W)
        y = y.mean([1, 2, 3], keepdim=True)     # (B//G, 1, 1, 1)
        y = y.expand(-1, 1, H, W)               # (B//G, 1, H, W)
        y = y.repeat(G, 1, 1, 1)               # (B,    1, H, W)
        return torch.cat([x, y], dim=1)


class StyleDiscriminator(nn.Module):
    """DCDiscriminator + Spectral Norm + MiniBatch StdDev."""
    def __init__(self, image_channels=3, ndf=64):
        super().__init__()
        self.mbstd = MiniBatchStdDev(group_size=4)
        # +1 canal do MiniBatch StdDev na última camada
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_channels, ndf,     4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf,     ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # MiniBatch StdDev adiciona 1 canal → ndf*4 + 1
        self.final = nn.utils.spectral_norm(nn.Conv2d(ndf * 4 + 1, 1, 4, 1, 0, bias=False))

    def forward(self, x):
        x = self.net(x)
        x = self.mbstd(x)
        return self.final(x).view(-1, 1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


generator     = StyleGenerator(LATENT_DIM, W_DIM, NGF, MAP_LAYERS).to(device)
discriminator = StyleDiscriminator(image_channels=3, ndf=NDF).to(device).apply(init_weights)

print('Generator params    :', sum(p.numel() for p in generator.parameters()))
print('Discriminator params:', sum(p.numel() for p in discriminator.parameters()))


# ── R1 Regularization (Mescheder et al. 2018) ─────────────────────────────

def r1_penalty(discriminator, real):
    """
    R1 = γ/2 * E[||∇D(x)||²], calculado só em imagens reais.
    Mais simples que GP: não precisa de amostras interpoladas.
    """
    real = real.requires_grad_(True)
    d_real = discriminator(real)
    grads  = torch.autograd.grad(
        d_real.sum(), real, create_graph=True
    )[0]
    return (grads.view(real.size(0), -1).norm(2, dim=1) ** 2).mean()


# ── Style Mixing Regularization ───────────────────────────────────────────

def sample_with_mixing(generator, z, mix_prob, n_blocks=8):
    """
    Com probabilidade mix_prob, gera um segundo z e mistura os estilos.
    mix_layer é amostrado uniformemente entre os blocos.
    """
    if random.random() < mix_prob:
        z2         = torch.randn_like(z)
        mix_layer  = random.randint(1, n_blocks - 1)
        return generator(z, z2=z2, mix_layer=mix_layer)
    return generator(z)


# ── Loop de treino ────────────────────────────────────────────────────────────

def train_stylegan(generator, discriminator, loader, latent_dim, epochs):
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(),     lr=LR_G, betas=(BETA1, BETA2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    history   = {'g_loss': [], 'd_loss': [], 'r1': []}
    n_blocks  = len(generator.blocks)
    batch_idx = 0

    generator.train(); discriminator.train()

    for epoch in range(epochs):
        g_run = d_run = r1_run = n_batches = 0

        for real, _ in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            real = real.to(device)
            bs   = real.size(0)
            real_t = torch.ones(bs,  1, device=device)
            fake_t = torch.zeros(bs, 1, device=device)

            # ── Discriminador ─────────────────────────────────────────────────
            opt_d.zero_grad(set_to_none=True)

            with torch.no_grad():
                z    = torch.randn(bs, latent_dim, device=device)
                fake = sample_with_mixing(generator, z, STYLE_MIX, n_blocks)

            d_loss = criterion(discriminator(real), real_t) \
                   + criterion(discriminator(fake.detach()), fake_t)

            # R1 regularization a cada R1_EVERY batches
            r1_val = torch.tensor(0.0, device=device)
            if batch_idx % R1_EVERY == 0:
                r1_val  = r1_penalty(discriminator, real)
                d_loss  = d_loss + (R1_GAMMA / 2) * r1_val * R1_EVERY

            d_loss.backward()
            opt_d.step()

            # ── Generator ─────────────────────────────────────────────────────
            opt_g.zero_grad(set_to_none=True)
            z    = torch.randn(bs, latent_dim, device=device)
            fake = sample_with_mixing(generator, z, STYLE_MIX, n_blocks)
            g_loss = criterion(discriminator(fake), real_t)
            g_loss.backward()
            opt_g.step()

            g_run   += g_loss.detach()
            d_run   += d_loss.detach()
            r1_run  += r1_val.detach()
            n_batches += 1
            batch_idx += 1

        nb   = max(n_batches, 1)
        to_f = lambda t: t.item() if hasattr(t, 'item') else float(t)
        history['g_loss'].append(to_f(g_run / nb))
        history['d_loss'].append(to_f(d_run / nb))
        history['r1'].append(to_f(r1_run / nb))
        print(f'Epoch {epoch+1:03d}/{epochs} | D: {history["d_loss"][-1]:.4f} | '
              f'G: {history["g_loss"][-1]:.4f} | R1: {history["r1"][-1]:.4f}')

        if cfg.save_samples and (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                imgs = (generator(FIXED_NOISE) * 0.5 + 0.5).clamp(0, 1)
            save_image(imgs, OUT_DIR / f'samples_epoch{epoch+1:03d}.png', nrow=8)
            generator.train()

    return history


# ── Treino ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    history = train_stylegan(generator, discriminator, train_loader, LATENT_DIM, EPOCHS)

    torch.save({
        'generator'    : generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'history'      : history,
        'config'       : {
            'latent_dim': LATENT_DIM, 'w_dim': W_DIM, 'channels': 3,
            'image_size': IMAGE_SIZE, 'model': 'stylegan',
        },
    }, OUT_DIR / 'dcgan_checkpoint.pt')
    print('Checkpoint guardado.')


# ── Curvas de treino ──────────────────────────────────────────────────────────

def plot_losses(history, title='StyleGAN losses'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['d_loss'], label='D'); ax1.plot(history['g_loss'], label='G')
    ax1.set_title('Adversarial losses'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['r1'], label='R1 penalty', color='tab:red')
    ax2.set_title('R1 regularization'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(title)
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'training_curves.png', bbox_inches='tight')

if __name__ == '__main__':
    plot_losses(history, title='ArtBench StyleGAN losses')


# ── Inferência ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(generator, latent_dim, n_samples=64, seed=123, title='StyleGAN inference'):
    torch.manual_seed(seed)
    generator.eval()
    z    = torch.randn(n_samples, latent_dim, device=device)
    fake = (generator(z) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid = make_grid(fake, nrow=8).permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 12)); plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'generated_samples.png', bbox_inches='tight')


@torch.no_grad()
def latent_walk(generator, latent_dim, steps=10, title='StyleGAN latent interpolation'):
    """Interpolação em z (passa pela mapping network — trajectória suave em W)."""
    generator.eval()
    z0     = torch.randn(1, latent_dim, device=device)
    z1     = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    zs     = (1 - alphas) * z0 + alphas * z1
    imgs   = (generator(zs) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid   = make_grid(imgs, nrow=steps).permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 2)); plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'latent_interpolation.png', bbox_inches='tight')


@torch.no_grad()
def style_mixing_demo(generator, latent_dim, steps=8, title='StyleGAN style mixing'):
    """
    Demonstra style mixing: linha = estilo base, coluna = estilo de detalhe.
    Mostra a separação entre atributos globais e locais.
    """
    generator.eval()
    n_base   = 4
    n_detail = 8
    z_base   = torch.randn(n_base,   latent_dim, device=device)
    z_detail = torch.randn(n_detail, latent_dim, device=device)
    n_blocks = len(generator.blocks)
    mix_at   = n_blocks // 2  # mix a partir de metade dos blocos

    rows = []
    for zb in z_base:
        row = []
        for zd in z_detail:
            img = generator(zb.unsqueeze(0), z2=zd.unsqueeze(0), mix_layer=mix_at)
            row.append(img)
        rows.append(torch.cat(row, dim=0))
    grid_imgs = (torch.cat(rows, dim=0) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid = make_grid(grid_imgs, nrow=n_detail).permute(1, 2, 0).numpy()
    plt.figure(figsize=(16, 8)); plt.imshow(grid); plt.title(title); plt.axis('off')
    if cfg.save_samples:
        plt.savefig(OUT_DIR / 'style_mixing.png', bbox_inches='tight')


if __name__ == '__main__':
    run_inference(generator, LATENT_DIM, n_samples=64, seed=123, title='ArtBench StyleGAN')
    latent_walk(generator, LATENT_DIM, steps=10, title='ArtBench StyleGAN latent walk')
    style_mixing_demo(generator, LATENT_DIM, title='ArtBench StyleGAN style mixing')
