import math
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from IAG.utils import get_device, save_samples


class GaussianDiffusion:
    """DDPM scheduler: stores schedule coefficients and applies forward/reverse diffusion."""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_prod = self._get_index(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alpha_prod * x_0 + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = self._get_index(self.betas, t, x.shape)
        sqrt_one_minus_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = 1. / torch.sqrt(self._get_index(self.alphas, t, x.shape))

        predicted_noise = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_t)

        if t_index == 0:
            return model_mean
        posterior_variance_t = self._get_index(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        model.eval()
        x = torch.randn(shape).to(self.device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long).to(self.device)
            x = self.p_sample(model, x, t, i)
        return x

    def _get_index(self, tensor, t, x_shape):
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, out_dim=None):
        super().__init__()
        self.out_dim = out_dim or dim
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, self.out_dim))
        self.conv1 = nn.Conv2d(dim, self.out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, dim)
        self.norm2 = nn.GroupNorm(4, self.out_dim)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, time_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.mlp(time_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return self.shortcut(x) + h


class PixelUNet(nn.Module):
    """U-Net noise predictor for 3-channel 32x32 ArtBench images."""

    def __init__(self, in_channels=3, model_channels=64):
        super().__init__()
        time_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down 1: 32 -> 16
        self.down1_res = ResnetBlock(model_channels, time_dim)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)

        # Down 2: 16 -> 8
        self.down2_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)

        # Middle (bottleneck at 8x8)
        self.mid_res1 = ResnetBlock(model_channels * 2, time_dim)
        self.mid_res2 = ResnetBlock(model_channels * 2, time_dim)

        # Up 2: 8 -> 16  (skip from down2_res: C*2, concat -> C*3)
        self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)
        self.up2_res = ResnetBlock(model_channels * 3, time_dim, out_dim=model_channels)

        # Up 1: 16 -> 32  (skip from down1_res: C, concat -> C*2)
        self.up1_conv = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1)
        self.up1_res = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels)

        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h_init = self.init_conv(x)

        h1 = self.down1_res(h_init, t_emb)
        h1_pool = self.down1_pool(h1)

        h2 = self.down2_res(h1_pool, t_emb)
        h2_pool = self.down2_pool(h2)

        h_mid = self.mid_res1(h2_pool, t_emb)
        h_mid = self.mid_res2(h_mid, t_emb)

        h_up2 = self.up2_conv(h_mid)
        h_up2 = torch.cat([h_up2, h2], dim=1)
        h_up2 = self.up2_res(h_up2, t_emb)

        h_up1 = self.up1_conv(h_up2)
        h_up1 = torch.cat([h_up1, h1], dim=1)
        h_up1 = self.up1_res(h_up1, t_emb)

        return self.out_conv(h_up1)


class DiffusionModel(nn.Module):
    """Wraps UNet + schedule so evaluate.py can call model(noise, None) for generation."""

    def __init__(self, unet, schedule):
        super().__init__()
        self.unet = unet
        self.schedule = schedule

    def forward(self, x, t):
        if t is None:
            return self.schedule.p_sample_loop(self.unet, x.shape)
        return self.unet(x, t)


def train_diffusion(dataloader, epochs=20, lr=0.0001, run_dir=None, sample_count=20, sample_nrow=10):
    device = get_device()
    schedule = GaussianDiffusion(device=str(device))
    unet = PixelUNet().to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    loss_log = []

    for epoch in range(epochs):
        unet.train()
        running = 0.0
        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch+1}/{epochs}")

        for imgs, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            noise = torch.randn_like(imgs)
            t = torch.randint(0, schedule.num_timesteps, (imgs.size(0),), device=device).long()
            x_t = schedule.q_sample(imgs, t, noise=noise)
            pred_noise = unet(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

        avg = running / len(dataloader)
        loss_log.append(avg)
        print(f"Diffusion Epoch {epoch+1:02d}/{epochs} | loss: {avg:.4f}")

        samples_dir = os.path.join(run_dir, "samples") if run_dir else "samples"
        checkpoint_path = os.path.join(run_dir, "diffusion_checkpoint.pth") if run_dir else "diffusion_checkpoint.pth"
        torch.save(unet.state_dict(), checkpoint_path)

        with torch.no_grad():
            sample = schedule.p_sample_loop(unet, shape=(sample_count, 3, 32, 32))
            save_samples(
                sample,
                os.path.join(samples_dir, f"diffusion_epoch_{epoch+1}.png"),
                nrow=sample_nrow,
            )

    if run_dir is not None:
        loss_path = os.path.join(run_dir, "loss.csv")
        with open(loss_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss"])
            for i, l in enumerate(loss_log, 1):
                writer.writerow([i, f"{l:.6f}"])
        print(f"Loss history saved to {loss_path}")

    return DiffusionModel(unet, schedule)