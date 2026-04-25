#!/usr/bin/env python
# coding: utf-8

# # 04 — Avaliação: FID & KID
# 
# Implementa o **protocolo obrigatório** do enunciado.
# 
# ### Protocolo
# 1. Por cada modelo: gerar **5 000 amostras**
# 2. Amostrar **5 000 imagens reais** do ArtBench-10
# 3. Calcular **FID** (5k vs 5k, conjunto completo)
# 4. Calcular **KID** em **50 subsets de tamanho 100** → média ± desvio-padrão
# 5. Repetir tudo com **10 seeds diferentes**
# 6. Reportar **FID: média ± std** e **KID: média ± std** por modelo
# 
# > Pré-processamento, contagem de amostras e código de avaliação devem ser **idênticos** entre modelos.

# In[1]:


from __future__ import annotations
import sys, random, csv, math
from pathlib import Path
import os, multiprocessing, warnings, sys
warnings.filterwarnings('ignore')
if multiprocessing.current_process().name != 'MainProcess':
    sys.stdout = open(os.devnull, 'w')

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import traceback

# métricas
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

# CONFIGURAÇÃO DE DEBUG: Mostrar parâmetros globais
print("="*60)
print(f"DEBUG: INICIALIZANDO TESTE")
print(f"  TARGET:        {os.environ.get('EVAL_TARGET', 'ALL')}")
print(f"  EXP_NAME:      {os.environ.get('EXP_NAME', 'ALL')}")
print(f"  VAE_LATENT:    {os.environ.get('VAE_LATENT_DIM', '128 (default)')}")
print(f"  DCGAN_LATENT:  {os.environ.get('DCGAN_LATENT', '100 (default)')}")
print(f"  DCGAN_NGF:     {os.environ.get('DCGAN_NGF', '64 (default)')}")
print(f"  DEVICE:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*60)

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


# ## 1. Configuração do protocolo

# In[2]:


KID_SUBSETS   = 50
KID_SUBSET_SZ = 100
BATCH_SIZE    = 128    # conforme Notebooks 3, 4, 5
IMAGE_SIZE    = 32
METRIC_BATCH  = 64     # tamanho do batch para atualizações das métricas (evita picos de memória)

from config import cfg
N_SAMPLES     = cfg.eval_samples
N_SEEDS       = cfg.eval_seeds

import os
EXP_NAME = os.environ.get('EXP_NAME', 'ALL')
EVAL_TARGET = os.environ.get('EVAL_TARGET', 'ALL')

CHECKPOINTS = {
    'VAE'      : REPO_ROOT / 'results' / (EXP_NAME if EVAL_TARGET in ['ALL', 'VAE'] and EXP_NAME != 'ALL' else 'vae') / 'vae_checkpoint.pth',
    'CVAE'     : REPO_ROOT / 'results' / (EXP_NAME if EVAL_TARGET in ['ALL', 'CVAE'] and EXP_NAME != 'ALL' else 'cvae') / 'cvae_checkpoint.pth',
    'VQ_VAE'   : REPO_ROOT / 'results' / (EXP_NAME if EVAL_TARGET in ['ALL', 'VQ_VAE'] and EXP_NAME != 'ALL' else 'vq_vae') / 'vq_vae_checkpoint.pth',
    'DCGAN'    : REPO_ROOT / 'results' / (EXP_NAME if EVAL_TARGET in ['ALL', 'DCGAN'] and EXP_NAME != 'ALL' else 'dcgan') / 'dcgan_checkpoint.pt',
    'Diffusion': REPO_ROOT / 'results' / (EXP_NAME if EVAL_TARGET in ['ALL', 'Diffusion'] and EXP_NAME != 'ALL' else 'diffusion') / 'diffusion_checkpoint.pth',
}

OUT_DIR = REPO_ROOT / 'results' / (EXP_NAME if EXP_NAME != 'ALL' else 'evaluation')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ## 2. Pool de imagens reais

# In[3]:


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

# [0,1] para FID/KID
transform_eval = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
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

real_pool = HFDatasetTorch(train_hf, transform_eval)
print(f'Pool de imagens reais: {len(real_pool)}')


def sample_real_images(n: int, seed: int) -> torch.Tensor:
    """Retorna (N, 3, H, W) uint8 [0,255]."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(real_pool), size=n, replace=False)
    imgs = torch.stack([real_pool[int(i)][0] for i in idx])
    return (imgs * 255).to(torch.uint8)


# ## 3. Definições dos modelos
# 
# Copia aqui as classes dos notebooks 01/02/03 para poder carregar os checkpoints.

# In[4]:


# ── ConvVAE (notebook 01) ─────────────────────────────────────────────────────
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)
        self.dec_fc    = nn.Linear(latent_dim, 128*4*4)
        self.dec_conv  = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1),  nn.Tanh(),
        )
    def encode(self, x):
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def decode(self, z):
        return self.dec_conv(self.dec_fc(z).view(-1, 128, 4, 4))
    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(self.reparameterize(mu, lv)), mu, lv


# ── ConditionalVAE (CVAE com suporte a classe) ────────────────────────────────
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10, class_embedding_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(128*4*4 + class_embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4 + class_embedding_dim, latent_dim)
        self.dec_fc    = nn.Linear(latent_dim + class_embedding_dim, 128*4*4)
        self.dec_conv  = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1),  nn.Tanh(),
        )
    
    def encode(self, x, c):
        h = self.enc_conv(x).view(x.size(0), -1)
        c_emb = self.class_embedding(c)
        h = torch.cat([h, c_emb], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    
    def decode(self, z, c):
        c_emb = self.class_embedding(c)
        z_c = torch.cat([z, c_emb], dim=1)
        return self.dec_conv(self.dec_fc(z_c).view(-1, 128, 4, 4))
    
    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        return self.decode(self.reparameterize(mu, lv), c), mu, lv


# ── VectorQuantizer (para VQ-VAE) ─────────────────────────────────────────────
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        input_shape = inputs.shape
        flat = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat, self.embedding.weight.t()))
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()
        return quantized, indices


# ── VQVAE (Vector Quantized VAE) ──────────────────────────────────────────────
class VQVAE(nn.Module):
    def __init__(self, latent_dim=128, num_embeddings=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.enc_to_latent = nn.Conv2d(128, latent_dim, 1, 1, 0)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.dec_from_latent = nn.Conv2d(latent_dim, 128, 1, 1, 0)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1),  nn.Tanh(),
        )
    
    def encode(self, x):
        h = self.enc_conv(x)
        z = self.enc_to_latent(h)
        return z
    
    def decode(self, z_q):
        h = self.dec_from_latent(z_q)
        return self.dec_conv(h)
    
    def forward(self, x):
        z = self.encode(x)
        z_q, indices = self.vq(z)
        return self.decode(z_q), z_q, indices


# ── DCGenerator (notebook 02) ─────────────────────────────────────────────────
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, image_channels=3, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),      nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),      nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False), nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


# ── PixelUNet helpers (notebook 03) ───────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb  = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResnetBlock(nn.Module):
    def __init__(self, dim, t_dim, out_dim=None):
        super().__init__()
        od = out_dim or dim
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, od))
        self.conv1 = nn.Conv2d(dim, od, 3, padding=1); self.conv2 = nn.Conv2d(od, od, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, dim); self.norm2 = nn.GroupNorm(4, od)
        self.act = nn.SiLU(); self.shortcut = nn.Conv2d(dim, od, 1) if dim != od else nn.Identity()
    def forward(self, x, te):
        h = self.conv1(self.act(self.norm1(x))) + self.mlp(te)[:, :, None, None]
        return self.shortcut(x) + self.conv2(self.act(self.norm2(h)))

class PixelUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=64):
        super().__init__()
        C, t = model_channels, model_channels*4
        self.time_embed = nn.Sequential(SinusoidalPosEmb(C), nn.Linear(C, t), nn.SiLU(), nn.Linear(t, t))
        self.init_conv  = nn.Conv2d(in_channels, C, 3, padding=1)
        self.down1_res  = ResnetBlock(C, t); self.down1_pool = nn.Conv2d(C, C, 3, stride=2, padding=1)
        self.down2_res  = ResnetBlock(C, t, C*2); self.down2_pool = nn.Conv2d(C*2, C*2, 3, stride=2, padding=1)
        self.mid_res1   = ResnetBlock(C*2, t); self.mid_res2 = ResnetBlock(C*2, t)
        self.up2_conv   = nn.ConvTranspose2d(C*2, C, 4, stride=2, padding=1); self.up2_res = ResnetBlock(C*3, t, C)
        self.up1_conv   = nn.ConvTranspose2d(C, C, 4, stride=2, padding=1);   self.up1_res = ResnetBlock(C*2, t, C)
        self.out_conv   = nn.Conv2d(C, in_channels, 3, padding=1)
    def forward(self, x, t):
        te  = self.time_embed(t); h = self.init_conv(x)
        h1  = self.down1_res(h, te);  h1p = self.down1_pool(h1)
        h2  = self.down2_res(h1p, te); h2p = self.down2_pool(h2)
        hm  = self.mid_res2(self.mid_res1(h2p, te), te)
        d2  = self.up2_res(torch.cat([self.up2_conv(hm), h2], 1), te)
        d1  = self.up1_res(torch.cat([self.up1_conv(d2), h1], 1), te)
        return self.out_conv(d1)


# ## 4. Carregar checkpoints

# In[5]:


# ── VAE ──────────────────────────────────────────────────────────────────────
lat_dim = int(os.environ.get('VAE_LATENT_DIM', 128))
vae_model = ConvVAE(latent_dim=lat_dim).to(device)
if EVAL_TARGET in ['ALL', 'VAE'] and CHECKPOINTS['VAE'].exists():
    vae_model.load_state_dict(torch.load(CHECKPOINTS['VAE'], map_location=device))
    vae_model.eval()
    print(f'VAE carregado (latent_dim={lat_dim})')

# ── CVAE (Conditional VAE) ───────────────────────────────────────────────────
cvae_model = ConditionalVAE(latent_dim=lat_dim, num_classes=10).to(device)
if EVAL_TARGET in ['ALL', 'CVAE'] and CHECKPOINTS['CVAE'].exists():
    cvae_model.load_state_dict(torch.load(CHECKPOINTS['CVAE'], map_location=device))
    cvae_model.eval()
    print(f'CVAE carregado (latent_dim={lat_dim})')

# ── VQ-VAE (Vector Quantized VAE) ────────────────────────────────────────────
vqvae_model = VQVAE(latent_dim=lat_dim, num_embeddings=256).to(device)
if EVAL_TARGET in ['ALL', 'VQ_VAE'] and CHECKPOINTS['VQ_VAE'].exists():
    vqvae_model.load_state_dict(torch.load(CHECKPOINTS['VQ_VAE'], map_location=device))
    vqvae_model.eval()
    print(f'VQ-VAE carregado (latent_dim={lat_dim})')

# ── DCGAN Generator ───────────────────────────────────────────────────────────
lat_gan = int(os.environ.get('DCGAN_LATENT', 100))
ngf_gan = int(os.environ.get('DCGAN_NGF', 64))
G_model  = DCGenerator(latent_dim=lat_gan, image_channels=3, ngf=ngf_gan).to(device)
if EVAL_TARGET in ['ALL', 'DCGAN'] and CHECKPOINTS['DCGAN'].exists():
    ckpt_gan = torch.load(CHECKPOINTS['DCGAN'], map_location=device)
    G_model.load_state_dict(ckpt_gan['generator'])
    G_model.eval()
    print(f'DCGAN Generator carregado (latent_dim={lat_gan}, ngf={ngf_gan})')

# ── Diffusion ─────────────────────────────────────────────────────────────────
# Prioridade: 1. Nome da EXPN_NAME | 2. Variável de ambiente | 3. Default 64
diff_ch = 64
if 'ch32' in EXP_NAME: diff_ch = 32
elif 'ch128' in EXP_NAME: diff_ch = 128
else: diff_ch = int(os.environ.get('DIFF_CHANNELS', 64))

diff_t = 1000
if 'T100' in EXP_NAME: diff_t = 100
elif 'T250' in EXP_NAME: diff_t = 250
elif 'T500' in EXP_NAME: diff_t = 500
else: diff_t = int(os.environ.get('DIFF_T_STEPS', 1000))

diff_model = PixelUNet(in_channels=3, model_channels=diff_ch).to(device)
if EVAL_TARGET in ['ALL', 'Diffusion'] and CHECKPOINTS['Diffusion'].exists():
    diff_model.load_state_dict(torch.load(CHECKPOINTS['Diffusion'], map_location=device))
    diff_model.eval()
    print(f'Diffusion model carregado (channels={diff_ch}, T={diff_t})')

# ── GaussianDiffusion schedule (necessário para sampling) ─────────────────────
class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps; self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        acp_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod           = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - acp_prev) / (1. - self.alphas_cumprod)
    def _get_index(self, tensor, t, shape):
        return tensor.gather(-1, t).view(t.shape[0], *((1,)*(len(shape)-1)))
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        x = torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps, desc='Sampling', leave=False):
            t    = torch.full((shape[0],), i, dtype=torch.long, device=self.device)
            b    = self._get_index(self.betas, t, x.shape)
            s1m  = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sr   = 1. / torch.sqrt(self._get_index(self.alphas, t, x.shape))
            eps  = model(x, t)
            mean = sr * (x - b * eps / s1m)
            if i > 0:
                x = mean + torch.sqrt(self._get_index(self.posterior_variance, t, x.shape)) * torch.randn_like(x)
            else:
                x = mean
        return x

schedule = GaussianDiffusion(diff_t, 1e-4, 0.02, device=str(device))


# ## 5. Funções de geração de amostras

# In[ ]:


@torch.no_grad()
def generate_vae(n: int, seed: int) -> torch.Tensor:
    """(N, 3, 32, 32) uint8."""
    torch.manual_seed(seed)
    batches = []
    for start in range(0, n, BATCH_SIZE):
        bs   = min(BATCH_SIZE, n - start)
        z    = torch.randn(bs, lat_dim, device=device)
        imgs = (vae_model.decode(z) * 0.5 + 0.5).clamp(0, 1)
        batches.append((imgs.cpu() * 255).to(torch.uint8))
    return torch.cat(batches)


@torch.no_grad()
def generate_dcgan(n: int, seed: int) -> torch.Tensor:
    """(N, 3, 32, 32) uint8."""
    torch.manual_seed(seed)
    batches = []
    for start in range(0, n, BATCH_SIZE):
        bs   = min(BATCH_SIZE, n - start)
        z    = torch.randn(bs, lat_gan, device=device)
        imgs = (G_model(z) * 0.5 + 0.5).clamp(0, 1)
        batches.append((imgs.cpu() * 255).to(torch.uint8))
    return torch.cat(batches)


@torch.no_grad()
def generate_diffusion(n: int, seed: int) -> torch.Tensor:
    """(N, 3, 32, 32) uint8 — sampling em batches para não esgotar VRAM."""
    torch.manual_seed(seed)
    batches = []
    for start in range(0, n, BATCH_SIZE):
        bs   = min(BATCH_SIZE, n - start)
        imgs = schedule.p_sample_loop(diff_model, shape=(bs, 3, IMAGE_SIZE, IMAGE_SIZE))
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        batches.append((imgs.cpu() * 255).to(torch.uint8))
    return torch.cat(batches)


@torch.no_grad()
def generate_cvae(n: int, seed: int) -> torch.Tensor:
    """CVAE: gera amostras com classe aleatória. (N, 3, 32, 32) uint8."""
    torch.manual_seed(seed)
    batches = []
    for start in range(0, n, BATCH_SIZE):
        bs   = min(BATCH_SIZE, n - start)
        z    = torch.randn(bs, lat_dim, device=device)
        c    = torch.randint(0, 10, (bs,), device=device)  # classe aleatória [0, 9]
        imgs = (cvae_model.decode(z, c) * 0.5 + 0.5).clamp(0, 1)
        batches.append((imgs.cpu() * 255).to(torch.uint8))
    return torch.cat(batches)


@torch.no_grad()
def generate_vqvae(n: int, seed: int) -> torch.Tensor:
    """VQ-VAE: gera amostras com latent aleatório. (N, 3, 32, 32) uint8."""
    torch.manual_seed(seed)
    batches = []
    for start in range(0, n, BATCH_SIZE):
        bs   = min(BATCH_SIZE, n - start)
        # VQ-VAE usa latent 4×4 (spatial), não 1×1
        z = torch.randn(bs, lat_dim, 4, 4, device=device)
        imgs = (vqvae_model.decode(z) * 0.5 + 0.5).clamp(0, 1)
        batches.append((imgs.cpu() * 255).to(torch.uint8))
    return torch.cat(batches)


# ## 6. Helpers de métricas

# In[ ]:


def compute_fid(real_u8: torch.Tensor, fake_u8: torch.Tensor) -> float:
    """FID no conjunto completo 5k vs 5k."""
    # If running on Apple MPS, compute metrics on CPU to avoid float64 MPS issues
    run_on_cpu = (device.type == 'mps')

    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    if run_on_cpu:
        fid = fid.cpu()
    else:
        fid = fid.to(device)

    # torchmetrics expects uint8 [0,255]; keep as-is
    real = real_u8
    fake = fake_u8

    # se estivermos no MPS, já definimos run_on_cpu True e o objeto da métrica está em CPU
    # processar updates em batches para reduzir uso de memória
    n = real.shape[0]
    try:
        try:
            fid_dev = next(fid.parameters()).device
        except StopIteration:
            fid_dev = 'cpu'
        print(f"[DEBUG] compute_fid: real dtype={real.dtype} device={real.device} shape={tuple(real.shape)}")
        print(f"[DEBUG] compute_fid: fid device={fid_dev}")

        for start in range(0, n, METRIC_BATCH):
            end = min(n, start + METRIC_BATCH)
            r_batch = real[start:end]
            f_batch = fake[start:end]
            if not run_on_cpu:
                r_batch = r_batch.to(device)
                f_batch = f_batch.to(device)
            else:
                # garantir que estejam em CPU
                r_batch = r_batch.cpu()
                f_batch = f_batch.cpu()
            fid.update(r_batch, real=True)
            fid.update(f_batch, real=False)

        return fid.compute().item()
    except Exception:
        print("[ERROR] Exception in compute_fid:")
        traceback.print_exc()
        raise


def compute_kid(real_u8: torch.Tensor, fake_u8: torch.Tensor,
                n_subsets=KID_SUBSETS, subset_size=KID_SUBSET_SZ) -> tuple[float, float]:
    """KID: média e std em 50 subsets de tamanho 100."""
    # If running on Apple MPS, compute metrics on CPU to avoid float64 MPS issues
    run_on_cpu = (device.type == 'mps')

    kid = KernelInceptionDistance(feature=2048, subset_size=subset_size, normalize=False)
    if run_on_cpu:
        kid = kid.cpu()
    else:
        kid = kid.to(device)

    # torchmetrics expects uint8 [0,255]; keep as-is
    real = real_u8
    fake = fake_u8

    n = real.shape[0]
    try:
        try:
            kid_dev = next(kid.parameters()).device
        except StopIteration:
            kid_dev = 'cpu'
        print(f"[DEBUG] compute_kid: real dtype={real.dtype} device={real.device} shape={tuple(real.shape)}")
        print(f"[DEBUG] compute_kid: kid device={kid_dev}")

        for start in range(0, n, METRIC_BATCH):
            end = min(n, start + METRIC_BATCH)
            r_batch = real[start:end]
            f_batch = fake[start:end]
            if not run_on_cpu:
                r_batch = r_batch.to(device)
                f_batch = f_batch.to(device)
            else:
                r_batch = r_batch.cpu()
                f_batch = f_batch.cpu()
            kid.update(r_batch, real=True)
            kid.update(f_batch, real=False)

        mean, std = kid.compute()
        return mean.item(), std.item()
    except Exception:
        print("[ERROR] Exception in compute_kid:")
        traceback.print_exc()
        raise


def evaluate_model(name: str, generate_fn, seeds=None) -> dict:
    """
    Protocolo completo: 10 seeds × (FID + KID).
    Reporta mean ± std de FID e mean ± std de KID.
    """
    if seeds is None:
        seeds = list(range(42, 42 + N_SEEDS))

    fid_scores, kid_means, kid_stds = [], [], []

    for seed in tqdm(seeds, desc=f'Evaluating {name}'):
        try:
            real = sample_real_images(N_SAMPLES, seed)
            fake = generate_fn(N_SAMPLES, seed)

            fid_val = compute_fid(real, fake)
            km, ks = compute_kid(real, fake)

            fid_scores.append(fid_val)
            kid_means.append(km)
            kid_stds.append(ks)
        except Exception:
            print(f"[WARNING] Error evaluating seed {seed} for model {name}, skipping this seed.")
            traceback.print_exc()
            fid_scores.append(float('nan'))
            kid_means.append(float('nan'))
            kid_stds.append(float('nan'))

    return {
        'model'   : name,
        'fid_mean': float(np.mean(fid_scores)),
        'fid_std' : float(np.std(fid_scores)),
        'kid_mean': float(np.mean(kid_means)),
        'kid_std' : float(np.mean(kid_stds)),
    }


# ## 7. Executar avaliação
# 
# > Nota: o Diffusion é lento (1000 passos por batch). Para testes rápidos reduz `N_SEEDS` ou `N_SAMPLES`.

# In[ ]:


if __name__ == '__main__':
    results = []
    if EVAL_TARGET in ['ALL', 'VAE'] and CHECKPOINTS['VAE'].exists():
        results.append(evaluate_model('VAE',       generate_vae))
    if EVAL_TARGET in ['ALL', 'CVAE'] and CHECKPOINTS['CVAE'].exists():
        results.append(evaluate_model('CVAE',      generate_cvae))
    if EVAL_TARGET in ['ALL', 'VQ_VAE'] and CHECKPOINTS['VQ_VAE'].exists():
        results.append(evaluate_model('VQ-VAE',    generate_vqvae))
    if EVAL_TARGET in ['ALL', 'DCGAN'] and CHECKPOINTS['DCGAN'].exists():
        results.append(evaluate_model('DCGAN',     generate_dcgan))
    if EVAL_TARGET in ['ALL', 'Diffusion'] and CHECKPOINTS['Diffusion'].exists():
        results.append(evaluate_model('Diffusion', generate_diffusion))

    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv(OUT_DIR / 'results.csv', index=False)
        print(df.to_string(index=False))

# ## 8. Tabela de resultados e gráfico

# In[ ]:


if __name__ == '__main__' and EVAL_TARGET == 'ALL':
    df = pd.read_csv(OUT_DIR / 'results.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(df['model'], df['fid_mean'], yerr=df['fid_std'], capsize=6)
    axes[0].set_title('FID (↓ melhor)'); axes[0].set_ylabel('FID'); axes[0].grid(alpha=0.3)
    axes[1].bar(df['model'], df['kid_mean'], yerr=df['kid_std'], capsize=6)
    axes[1].set_title('KID mean (↓ melhor)'); axes[1].set_ylabel('KID'); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'comparison.png', bbox_inches='tight')
    # plt.show()

    # tabela para o relatório LaTeX
    print('\nModelo         | FID mean±std         | KID mean±std')
    print('-' * 58)
    for _, row in df.iterrows():
        print(f"{row['model']:14s} | {row['fid_mean']:7.2f} ± {row['fid_std']:5.2f}        | "
              f"{row['kid_mean']:.4f} ± {row['kid_std']:.4f}")

