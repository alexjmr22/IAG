#!/usr/bin/env python
# coding: utf-8
"""
Percorre pastas de resultados já treinados e gera:
  - reconstructions.png  : original | noisy/latent | denoised/gerado
  - generated_samples.png: 16 amostras geradas (se não existir)

Suporta: Diffusion (DDPM/DDIM), Diffusion EMA, DCGAN

Uso:
  python scripts/generate_diff_plots.py                        # todas as pastas diff_* e dcgan_*
  python scripts/generate_diff_plots.py --dirs diff_ema_e200 dcgan_spectral_200ep
  python scripts/generate_diff_plots.py --force               # regenera mesmo que já exista
"""

from __future__ import annotations
import argparse, csv, math, re, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS     = REPO_ROOT / 'results'
SCRIPTS_DIR = REPO_ROOT / 'TP' / 'TP1-alunos-src-only' / 'scripts'
SUBSET_CSV  = REPO_ROOT / 'TP' / 'TP1-alunos-src-only' / 'student_start_pack' / 'training_20_percent.csv'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

IMAGE_SIZE = 32

# ── Device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print('Device:', device)

# ── Dataset ───────────────────────────────────────────────────────────────────

transform = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

def load_ids_from_csv(csv_path, column='train_id_original'):
    ids = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            v = str(row.get(column, '')).strip()
            if v: ids.append(int(v))
    return ids

class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds = hf_split; self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        x  = self.transform(ex['image']) if self.transform else ex['image']
        return x, int(ex['label'])

_loader_cache: dict = {}

def get_loader(use_subset: bool):
    key = 'subset' if use_subset else 'full'
    if key in _loader_cache:
        return _loader_cache[key]
    from datasets import load_from_disk, load_dataset
    DATA_CACHE = REPO_ROOT / 'data' / 'artbench10_hf'
    if DATA_CACHE.exists():
        train_hf = load_from_disk(str(DATA_CACHE))['train']
    else:
        ds = load_dataset('zguo0525/ArtBench')
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(DATA_CACHE))
        train_hf = ds['train']
    indices = load_ids_from_csv(SUBSET_CSV) if use_subset else None
    ds_torch = HFDatasetTorch(train_hf, transform, indices=indices)
    loader = DataLoader(ds_torch, batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    _loader_cache[key] = loader
    return loader

def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1).cpu()

# ═══════════════════════════════════════════════════════════════════════════════
#  DIFFUSION
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
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
    def __init__(self, in_channels=3, model_channels=64):
        super().__init__()
        C, t = model_channels, model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(C), nn.Linear(C, t), nn.SiLU(), nn.Linear(t, t))
        self.init_conv  = nn.Conv2d(in_channels, C, 3, padding=1)
        self.down1_res  = ResnetBlock(C,   t)
        self.down1_pool = nn.Conv2d(C,   C,   3, stride=2, padding=1)
        self.down2_res  = ResnetBlock(C,   t, out_dim=C*2)
        self.down2_pool = nn.Conv2d(C*2, C*2, 3, stride=2, padding=1)
        self.mid_res1   = ResnetBlock(C*2, t)
        self.mid_res2   = ResnetBlock(C*2, t)
        self.up2_conv   = nn.ConvTranspose2d(C*2, C,   4, stride=2, padding=1)
        self.up2_res    = ResnetBlock(C*3, t, out_dim=C)
        self.up1_conv   = nn.ConvTranspose2d(C,   C,   4, stride=2, padding=1)
        self.up1_res    = ResnetBlock(C*2, t, out_dim=C)
        self.out_conv   = nn.Conv2d(C, in_channels, 3, padding=1)
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
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        a  = self._get(self.sqrt_alphas_cumprod,           t, x_0.shape)
        sa = self._get(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return a * x_0 + sa * noise

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, ddim_steps=100):
        model.eval()
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.num_timesteps, step_size)))[:ddim_steps]
        x = torch.randn(shape).to(self.device)
        for i, t_cur in enumerate(tqdm(timesteps, desc='DDIM generate', leave=False)):
            t_prev     = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch    = torch.full((shape[0],), t_cur, dtype=torch.long, device=self.device)
            pred_noise = model(x, t_batch)
            a_t  = self.alphas_cumprod[t_cur]
            a_tp = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
            x0_pred = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            x = a_tp.sqrt() * x0_pred + (1 - a_tp).clamp(min=0).sqrt() * pred_noise
        return x

    @torch.no_grad()
    def ddim_reconstruct(self, model, x_real, t_noise, ddim_steps=50):
        """Forward noise até t_noise, depois DDIM reverse t_noise → 0."""
        model.eval()
        bs = x_real.shape[0]
        t_batch = torch.full((bs,), t_noise, dtype=torch.long, device=self.device)
        x_noisy = self.q_sample(x_real, t_batch)

        step_size = max(1, t_noise // ddim_steps)
        timesteps = list(reversed(range(0, t_noise, step_size)))[:ddim_steps]

        x = x_noisy.clone()
        for i, t_cur in enumerate(tqdm(timesteps, desc='DDIM recon', leave=False)):
            t_prev      = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch_cur = torch.full((bs,), t_cur, dtype=torch.long, device=self.device)
            pred_noise  = model(x, t_batch_cur)
            a_t  = self.alphas_cumprod[t_cur]
            a_tp = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
            x0_pred = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            x = a_tp.sqrt() * x0_pred + (1 - a_tp).clamp(min=0).sqrt() * pred_noise
        return x_noisy, x

    def _get(self, tensor, t, x_shape):
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

# ── parse experiment_params.md ───────────────────────────────────────────────

def parse_params(params_md: Path) -> dict:
    defaults = {'T_STEPS': 1000, 'BETA_START': 1e-4, 'BETA_END': 0.02,
                'CHANNELS': 64, 'DDIM_STEPS': 100, 'IS_EMA': False,
                'LATENT_DIM': 100, 'NGF': 64, 'NDF': 64,
                'USE_SUBSET': False}
    if not params_md.exists():
        return defaults
    text = params_md.read_text(encoding='utf-8')
    def find(pattern, cast=float, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default
    defaults['T_STEPS']    = find(r'T Steps[^\d]+(\d+)', int,   1000)
    defaults['BETA_START'] = find(r'Beta Start[^\d]+([\d.e+-]+)', float, 1e-4)
    defaults['BETA_END']   = find(r'Beta End[^\d]+([\d.e+-]+)',   float, 0.02)
    defaults['CHANNELS']   = find(r'Channels[^\d]+(\d+)', int,   64)
    defaults['DDIM_STEPS'] = find(r'DDIM \((\d+) steps\)', int,  100) or 100
    defaults['LATENT_DIM'] = find(r'Latent Dim[^\d]+(\d+)', int, 100)
    defaults['NGF']        = find(r'NGF[^\d]+(\d+)', int,        64)
    defaults['IS_EMA']     = 'EMA' in text or 'ema' in params_md.parent.name.lower()
    defaults['USE_SUBSET'] = '20%' in text
    return defaults

# ── Diffusion: gerar plots ───────────────────────────────────────────────────

def process_diffusion(out_dir: Path, force: bool):
    params    = parse_params(out_dir / 'experiment_params.md')
    is_ema    = params['IS_EMA']
    ckpt_path = out_dir / ('diffusion_ema_checkpoint.pth' if is_ema else 'diffusion_checkpoint.pth')
    if not ckpt_path.exists():
        ckpt_path = out_dir / 'diffusion_checkpoint.pth'
    if not ckpt_path.exists():
        print(f'  SKIP — checkpoint não encontrado'); return

    recon_out = out_dir / 'reconstructions.png'
    gen_out   = out_dir / 'generated_samples.png'
    if not force and recon_out.exists() and gen_out.exists():
        print(f'  SKIP — ficheiros já existem (usa --force para regenerar)'); return

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}

    # inferir channels directamente do checkpoint (init_conv.weight shape: [C, 3, 3, 3])
    channels = state['init_conv.weight'].shape[0]

    model = PixelUNet(in_channels=3, model_channels=channels).to(device)
    state = {k: v.to(next(model.parameters()).dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'  channels={channels}  T={params["T_STEPS"]}  EMA={is_ema}  ckpt={ckpt_path.name}')

    schedule = GaussianDiffusion(params['T_STEPS'], params['BETA_START'], params['BETA_END'], device=str(device))
    ema_tag  = ' EMA' if is_ema else ''
    loader   = get_loader(params['USE_SUBSET'])

    # ── Reconstruções ────────────────────────────────────────────────────────
    if force or not recon_out.exists():
        x_real, _ = next(iter(loader))
        x_real = x_real[:8].to(device)

        # t_noise = 15% do schedule — ainda há estrutura visível no noisy
        t_noise = max(50, params['T_STEPS'] // 7)
        ddim_recon_steps = min(50, t_noise)

        with torch.no_grad():
            x_noisy, x_denoised = schedule.ddim_reconstruct(model, x_real, t_noise, ddim_steps=ddim_recon_steps)

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        for i in range(8):
            axes[0, i].imshow(denorm(x_real[i]).permute(1, 2, 0))
            axes[1, i].imshow(denorm(x_noisy[i]).permute(1, 2, 0))    # clamp já dentro de denorm
            axes[2, i].imshow(denorm(x_denoised[i]).permute(1, 2, 0))
            for row in range(3): axes[row, i].axis('off')
        axes[0, 0].set_ylabel('original',           fontsize=9)
        axes[1, 0].set_ylabel(f'noisy (t={t_noise})', fontsize=9)
        axes[2, 0].set_ylabel('denoised (DDIM)',     fontsize=9)
        plt.suptitle(f'Diffusion{ema_tag} [{out_dir.name}] — reconstruções (t={t_noise}, {ddim_recon_steps} DDIM steps)')
        plt.tight_layout()
        plt.savefig(recon_out, bbox_inches='tight', dpi=120)
        plt.close()
        print(f'  -> reconstructions.png')

    # ── Amostras geradas ─────────────────────────────────────────────────────
    if force or not gen_out.exists():
        with torch.no_grad():
            samples = schedule.ddim_sample_loop(model, shape=(16, 3, IMAGE_SIZE, IMAGE_SIZE),
                                                ddim_steps=params['DDIM_STEPS'])
        imgs_ = denorm(samples)
        grid  = make_grid(imgs_, nrow=4).permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.title(f'DDIM{ema_tag} [{out_dir.name}] — amostras geradas')
        plt.axis('off')
        plt.savefig(gen_out, bbox_inches='tight', dpi=120)
        plt.close()
        print(f'  -> generated_samples.png')

    # ── Progressão do denoising ──────────────────────────────────────────────
    progress_out = out_dir / 'denoising_progress.png'
    if force or not progress_out.exists():
        T       = params['T_STEPS']
        d_steps = params['DDIM_STEPS']
        # 4 imagens de partida, mostrar 6 instantes do processo
        n_imgs    = 4
        step_size = T // d_steps
        timesteps = list(reversed(range(0, T, step_size)))[:d_steps]
        snapshots_at = [0, len(timesteps)//5, 2*len(timesteps)//5,
                        3*len(timesteps)//5, 4*len(timesteps)//5, len(timesteps)-1]

        with torch.no_grad():
            model.eval()
            x = torch.randn(n_imgs, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            frames = []
            for step_i, (i, t_cur) in enumerate(enumerate(tqdm(timesteps, desc='DDIM progress', leave=False))):
                if step_i in snapshots_at:
                    frames.append(x.clone().cpu())
                t_prev     = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t_batch    = torch.full((n_imgs,), t_cur, dtype=torch.long, device=device)
                pred_noise = model(x, t_batch)
                a_t  = schedule.alphas_cumprod[t_cur]
                a_tp = schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
                x0_pred = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
                x0_pred = x0_pred.clamp(-1, 1)
                x = a_tp.sqrt() * x0_pred + (1 - a_tp).clamp(min=0).sqrt() * pred_noise
            frames.append(x.clone().cpu())  # frame final

        n_frames = len(frames)
        fig, axes = plt.subplots(n_imgs, n_frames, figsize=(n_frames * 2.5, n_imgs * 2.5))
        step_labels = [f't={timesteps[s]}' for s in snapshots_at] + ['t=0 (final)']
        for col, (frame, label) in enumerate(zip(frames, step_labels)):
            for row in range(n_imgs):
                axes[row, col].imshow(denorm(frame[row]).permute(1, 2, 0))
                axes[row, col].axis('off')
            axes[0, col].set_title(label, fontsize=8)
        plt.suptitle(f'Diffusion{ema_tag} [{out_dir.name}] — progressão do denoising')
        plt.tight_layout()
        plt.savefig(progress_out, bbox_inches='tight', dpi=120)
        plt.close()
        print(f'  -> denoising_progress.png')

# ═══════════════════════════════════════════════════════════════════════════════
#  DCGAN
# ═══════════════════════════════════════════════════════════════════════════════

class DCGenerator(nn.Module):
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

def gan_inversion(generator, x_real, latent_dim, steps=500, lr=0.05):
    """Encontra z tal que G(z) ≈ x_real por gradient descent no espaço latente."""
    bs = x_real.shape[0]
    z  = torch.randn(bs, latent_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x_gen = generator(z)
        loss  = F.mse_loss(x_gen, x_real) + 0.01 * z.pow(2).mean()  # MSE + L2 reg no z
        loss.backward()
        opt.step()
    with torch.no_grad():
        x_recon = generator(z)
    return x_recon.detach()


def process_dcgan(out_dir: Path, force: bool):
    ckpt_path = out_dir / 'dcgan_checkpoint.pt'
    if not ckpt_path.exists():
        print(f'  SKIP — dcgan_checkpoint.pt não encontrado'); return

    recon_out = out_dir / 'reconstructions.png'
    gen_out   = out_dir / 'generated_samples.png'
    if not force and recon_out.exists() and gen_out.exists():
        print(f'  SKIP — ficheiros já existem (usa --force para regenerar)'); return

    ckpt = torch.load(ckpt_path, map_location=device)

    cfg_saved  = ckpt.get('config', {})
    latent_dim = cfg_saved.get('latent_dim', 100)
    gen_state  = ckpt['generator']
    first_key  = next(k for k in gen_state if 'net.0.weight' in k)
    # ConvTranspose2d weight: [in_channels, out_channels, kH, kW] → shape[1] = ngf*4
    ngf_x4     = gen_state[first_key].shape[1]
    ngf        = ngf_x4 // 4

    generator = DCGenerator(latent_dim=latent_dim, image_channels=3, ngf=ngf).to(device)
    generator.load_state_dict(gen_state)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    print(f'  latent_dim={latent_dim}  ngf={ngf}')

    loader = get_loader(use_subset=False)

    # ── Reconstrução DCGAN: GAN Inversion ────────────────────────────────────
    # Otimiza z por gradient descent até G(z) ≈ x_real
    if force or not recon_out.exists():
        x_real, _ = next(iter(loader))
        x_real = x_real[:8].to(device)

        print('  GAN inversion (500 steps)...')
        x_recon = gan_inversion(generator, x_real, latent_dim, steps=500, lr=0.05)

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(denorm(x_real[i].cpu()).permute(1, 2, 0))
            axes[1, i].imshow(denorm(x_recon[i].cpu()).permute(1, 2, 0))
            axes[0, i].axis('off'); axes[1, i].axis('off')
        axes[0, 0].set_ylabel('original', fontsize=9)
        axes[1, 0].set_ylabel('recon\n(GAN inv)', fontsize=9)
        plt.suptitle(f'DCGAN [{out_dir.name}] — reconstruções via GAN Inversion')
        plt.tight_layout()
        plt.savefig(recon_out, bbox_inches='tight', dpi=120)
        plt.close()
        print(f'  -> reconstructions.png')

    # ── Amostras geradas (grid 4×4) ──────────────────────────────────────────
    if force or not gen_out.exists():
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            samples = generator(z).cpu()
        imgs_ = denorm(samples)
        grid  = make_grid(imgs_, nrow=4).permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.title(f'DCGAN [{out_dir.name}] — amostras geradas')
        plt.axis('off')
        plt.savefig(gen_out, bbox_inches='tight', dpi=120)
        plt.close()
        print(f'  -> generated_samples.png')

# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def detect_type(out_dir: Path) -> str:
    if (out_dir / 'dcgan_checkpoint.pt').exists():
        return 'dcgan'
    if (out_dir / 'diffusion_ema_checkpoint.pth').exists():
        return 'diffusion_ema'
    if (out_dir / 'diffusion_checkpoint.pth').exists():
        return 'diffusion'
    return 'unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='*', default=None,
                        help='Nomes exactos das pastas em results/')
    parser.add_argument('--filter', nargs='*', default=None,
                        help='Filtrar pastas que contenham estas strings (ex: --filter diff ema)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerar mesmo que os ficheiros já existam')
    args = parser.parse_args()

    if args.dirs:
        targets = [RESULTS / d for d in args.dirs]
    elif args.filter:
        targets = sorted(
            p for p in RESULTS.iterdir()
            if p.is_dir() and any(tag.lower() in p.name.lower() for tag in args.filter)
        )
    else:
        targets = sorted(
            p for p in RESULTS.iterdir()
            if p.is_dir() and any(tag in p.name.lower() for tag in ('diff', 'dcgan'))
        )

    if not targets:
        print('Nenhuma pasta encontrada em results/'); return

    print(f'Pastas a processar ({len(targets)}): {[p.name for p in targets]}')
    print('A carregar dataset...')
    get_loader(use_subset=False)   # pré-carregar

    for target in targets:
        if not target.exists():
            print(f'\nAVISO: {target.name} não existe, a saltar.'); continue

        kind = detect_type(target)
        print(f'\n{"="*60}\n  {target.name}  [{kind}]\n{"="*60}')

        if kind in ('diffusion', 'diffusion_ema'):
            process_diffusion(target, args.force)
        elif kind == 'dcgan':
            process_dcgan(target, args.force)
        else:
            print(f'  SKIP — tipo desconhecido (sem checkpoint reconhecível)')

    print('\nConcluído.')

if __name__ == '__main__':
    main()
