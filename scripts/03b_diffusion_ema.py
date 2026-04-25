#!/usr/bin/env python
# coding: utf-8

# # 03b — Diffusion Model com EMA (Exponential Moving Average)
#
# Idêntico ao 03_diffusion.py mas com EMAModel a manter uma cópia suavizada
# dos pesos. O sampling final usa os pesos EMA em vez dos pesos do optimizador.
#
# Único parâmetro novo: DIFF_EMA_DECAY (default 0.9999)

from __future__ import annotations
import sys, random, csv, math, copy
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

import os
IMAGE_SIZE   = 32
BATCH_SIZE   = 128
LR           = float(os.environ.get('DIFF_LR', 4e-4))

from config import cfg
EPOCHS       = int(os.environ.get('DIFF_EPOCHS', 100))
USE_SUBSET   = cfg.use_subset

T_STEPS       = int(os.environ.get('DIFF_T_STEPS', 1000))
BETA_START    = float(os.environ.get('DIFF_BETA_START', 1e-4))
BETA_END      = float(os.environ.get('DIFF_BETA_END', 0.02))
WARMUP_EPOCHS = int(os.environ.get('DIFF_WARMUP_EPOCHS', 5))
EMA_DECAY     = float(os.environ.get('DIFF_EMA_DECAY', 0.9999))

EXP_NAME = os.environ.get('EXP_NAME', 'diffusion_ema')
OUT_DIR  = REPO_ROOT / 'results' / 'diff' / EXP_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLER    = os.environ.get('DIFF_SAMPLER', 'ddim').lower()
DDIM_STEPS = int(os.environ.get('DIFF_DDIM_STEPS', 100))

def save_params():
    params_md = f"""# Diffusion EMA Experiment: {EXP_NAME}
- **T Steps**: {T_STEPS}
- **Learning Rate**: {LR}
- **LR Scheduler**: cosine annealing with {WARMUP_EPOCHS}-epoch linear warmup
- **Epochs**: {EPOCHS}
- **Batch Size**: {BATCH_SIZE}
- **Beta Start**: {BETA_START}
- **Beta End**: {BETA_END}
- **Channels**: {int(os.environ.get('DIFF_CHANNELS', 96))}
- **Sampler**: {SAMPLER.upper()}{f' ({DDIM_STEPS} steps)' if SAMPLER == 'ddim' else ''}
- **EMA Decay**: {EMA_DECAY}
- **Dataset**: {"20% Subset" if USE_SUBSET else "Full ArtBench10"}
"""
    with open(OUT_DIR / "experiment_params.md", "w", encoding="utf-8") as f:
        f.write(params_md)
    print(f"Parâmetros guardados em {OUT_DIR / 'experiment_params.md'}")

save_params()


# ── Dataset ──────────────────────────────────────────────────────────────────

def safe_num_workers():
    return 4

class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds        = hf_split
        self.transform = transform
        self.indices   = list(range(len(hf_split))) if indices is None else list(indices)
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


# ── GaussianDiffusion ────────────────────────────────────────────────────────

class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device        = device

        self.betas               = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas              = 1. - self.betas
        self.alphas_cumprod      = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod           = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance   = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        a  = self._get_index(self.sqrt_alphas_cumprod,           t, x_0.shape)
        sa = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return a * x_0 + sa * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t    = self._get_index(self.betas, t, x.shape)
        sqrt_1mab  = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip = 1. / torch.sqrt(self._get_index(self.alphas, t, x.shape))

        pred_noise = model(x, t)
        mean       = sqrt_recip * (x - betas_t * pred_noise / sqrt_1mab)

        if t_index == 0:
            return mean
        var   = self._get_index(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        model.eval()
        x = torch.randn(shape).to(self.device)
        for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps,
                      desc='Sampling', leave=False):
            t = torch.full((shape[0],), i, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, t, i)
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, ddim_steps=100, eta=0.0):
        model.eval()
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.num_timesteps, step_size)))[:ddim_steps]

        x = torch.randn(shape).to(self.device)
        for i, t_cur in enumerate(tqdm(timesteps, desc='DDIM Sampling', leave=False)):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            t_batch    = torch.full((shape[0],), t_cur, dtype=torch.long, device=self.device)
            pred_noise = model(x, t_batch)

            a_t  = self.alphas_cumprod[t_cur]
            a_tp = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            x0_pred = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = eta * ((1 - a_tp) / (1 - a_t)).sqrt() * (1 - a_t / a_tp).sqrt()
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = a_tp.sqrt() * x0_pred + (1 - a_tp - sigma**2).clamp(min=0).sqrt() * pred_noise + sigma * noise

        return x

    def _get_index(self, tensor, t, x_shape):
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


schedule = GaussianDiffusion(T_STEPS, BETA_START, BETA_END, device=str(device))

def _sample(model, shape):
    if SAMPLER == 'ddim':
        return schedule.ddim_sample_loop(model, shape, ddim_steps=DDIM_STEPS)
    return schedule.p_sample_loop(model, shape)


# ── PixelUNet ────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, out_dim=None):
        super().__init__()
        self.out_dim  = out_dim or dim
        self.mlp      = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, self.out_dim))
        self.conv1    = nn.Conv2d(dim,          self.out_dim, 3, padding=1)
        self.conv2    = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        self.norm1    = nn.GroupNorm(4, dim)
        self.norm2    = nn.GroupNorm(4, self.out_dim)
        self.act      = nn.SiLU()
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h) + self.mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return self.shortcut(x) + h


class PixelUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=int(os.environ.get('DIFF_CHANNELS', 96))):
        super().__init__()
        C, t = model_channels, model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(C),
            nn.Linear(C, t), nn.SiLU(), nn.Linear(t, t),
        )
        self.init_conv  = nn.Conv2d(in_channels, C, 3, padding=1)

        self.down1_res  = ResnetBlock(C,   t)
        self.down1_pool = nn.Conv2d(C,   C,   3, stride=2, padding=1)
        self.down2_res  = ResnetBlock(C,   t, out_dim=C*2)
        self.down2_pool = nn.Conv2d(C*2, C*2, 3, stride=2, padding=1)

        self.mid_res1 = ResnetBlock(C*2, t)
        self.mid_res2 = ResnetBlock(C*2, t)

        self.up2_conv = nn.ConvTranspose2d(C*2, C,   4, stride=2, padding=1)
        self.up2_res  = ResnetBlock(C*3, t, out_dim=C)
        self.up1_conv = nn.ConvTranspose2d(C,   C,   4, stride=2, padding=1)
        self.up1_res  = ResnetBlock(C*2, t, out_dim=C)

        self.out_conv = nn.Conv2d(C, in_channels, 3, padding=1)

    def forward(self, x, t):
        te  = self.time_embed(t)
        h   = self.init_conv(x)
        h1  = self.down1_res(h,   te)
        h1p = self.down1_pool(h1)
        h2  = self.down2_res(h1p, te)
        h2p = self.down2_pool(h2)
        hm  = self.mid_res2(self.mid_res1(h2p, te), te)
        d2  = self.up2_res(torch.cat([self.up2_conv(hm), h2], 1), te)
        d1  = self.up1_res(torch.cat([self.up1_conv(d2), h1], 1), te)
        return self.out_conv(d1)


# ── EMAModel ─────────────────────────────────────────────────────────────────

class EMAModel:
    """
    Mantém uma cópia suavizada dos pesos do modelo via média exponencial.
    Usar ema.apply() antes do sampling e ema.restore() depois para não
    interferir com o treino.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay      = decay
        self.shadow     = {k: v.clone().float() for k, v in model.state_dict().items()}
        self._backup    = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply(self, model: nn.Module):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        ema_state = {k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()}
        model.load_state_dict(ema_state)

    def restore(self, model: nn.Module):
        model.load_state_dict(self._backup)
        self._backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.float() for k, v in state.items()}


# ── Treino com EMA ───────────────────────────────────────────────────────────

def train_diffusion_ema(model, loader, schedule, epochs=100, lr=4e-4, warmup_epochs=5, ema_decay=0.9999):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ema = EMAModel(model, decay=ema_decay)
    history = []
    model.train()

    for epoch in range(epochs):
        running = n_batches = 0
        for x, _ in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            x  = x.to(device)
            bs = x.shape[0]
            t  = torch.randint(0, schedule.num_timesteps, (bs,), device=device).long()

            noise = torch.randn_like(x)
            x_t   = schedule.q_sample(x, t, noise=noise)
            pred  = model(x_t, t)
            loss  = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ema.update(model)

            running   += loss.detach()
            n_batches += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg = (running / max(n_batches, 1)).item() if hasattr(running, 'item') else running / max(n_batches, 1)
        history.append({'loss': avg, 'lr': current_lr})
        print(f'Epoch {epoch+1:03d}/{epochs} | loss: {avg:.5f} | lr: {current_lr:.2e}')

        if cfg.save_samples and (epoch + 1) % 10 == 0:
            ema.apply(model)
            samples = _sample(model, shape=(16, 3, IMAGE_SIZE, IMAGE_SIZE))
            ema.restore(model)
            imgs = (samples * 0.5 + 0.5).clamp(0, 1)
            save_image(imgs, OUT_DIR / f'samples_epoch{epoch+1:03d}.png', nrow=4)
            model.train()

    return model, ema, history


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _diff_ch = int(os.environ.get('DIFF_CHANNELS', 96))
    model = PixelUNet(in_channels=3, model_channels=_diff_ch).to(device)
    print(f'Parâmetros: {sum(p.numel() for p in model.parameters()):,}')

    model, ema, history = train_diffusion_ema(
        model, train_loader, schedule,
        epochs=EPOCHS, lr=LR, warmup_epochs=WARMUP_EPOCHS, ema_decay=EMA_DECAY
    )

    # Guardar pesos raw e pesos EMA separados
    torch.save(model.state_dict(), OUT_DIR / 'diffusion_checkpoint.pth')
    torch.save(ema.state_dict(),   OUT_DIR / 'diffusion_ema_checkpoint.pth')
    print('Checkpoints guardados (raw + EMA).')

    # Curva de treino
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([h['loss'] for h in history])
    axes[0].set_title('Diffusion EMA training loss (MSE)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    axes[1].plot([h['lr'] for h in history])
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('LR')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'training_curves.png')
    plt.close()

    # Amostras finais com pesos EMA
    ema.apply(model)
    samples = _sample(model, shape=(16, 3, IMAGE_SIZE, IMAGE_SIZE))
    ema.restore(model)

    imgs_ = (samples * 0.5 + 0.5).clamp(0, 1).cpu()
    grid  = make_grid(imgs_, nrow=4).permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.title(f'{SAMPLER.upper()} EMA — amostras geradas')
    plt.axis('off')
    plt.savefig(OUT_DIR / 'generated_samples.png', bbox_inches='tight')
    plt.close()
    print('Amostras geradas com pesos EMA.')

    # Reconstruções — forward noise + denoising parcial com pesos EMA
    model.eval()
    x_real, _ = next(iter(train_loader))
    x_real = x_real[:8].to(device)

    t_mid = schedule.num_timesteps // 2
    with torch.no_grad():
        t_batch = torch.full((8,), t_mid, dtype=torch.long, device=device)
        x_noisy = schedule.q_sample(x_real, t_batch)
        ema.apply(model)
        x_denoised = x_noisy.clone()
        for i in reversed(range(t_mid)):
            t_i = torch.full((8,), i, dtype=torch.long, device=device)
            x_denoised = schedule.p_sample(model, x_denoised, t_i, i)
        ema.restore(model)

    def denorm(t): return (t * 0.5 + 0.5).clamp(0, 1).cpu()

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for i in range(8):
        axes[0, i].imshow(denorm(x_real[i]).permute(1, 2, 0))
        axes[1, i].imshow(denorm(x_noisy[i]).permute(1, 2, 0))
        axes[2, i].imshow(denorm(x_denoised[i]).permute(1, 2, 0))
        for row in range(3): axes[row, i].axis('off')
    axes[0, 0].set_ylabel('original', fontsize=9)
    axes[1, 0].set_ylabel(f'noisy (t={t_mid})', fontsize=9)
    axes[2, 0].set_ylabel('denoised (EMA)', fontsize=9)
    plt.suptitle(f'Diffusion EMA — reconstruções (forward t={t_mid} + reverse)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'reconstructions.png', bbox_inches='tight')
    plt.close()
    print('Reconstruções guardadas.')
