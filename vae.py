import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from IAG.utils import get_device, save_samples


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*8*8, latent_dim)
        self.fc_logvar = nn.Linear(64*8*8, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*8*8), nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


def train_vae(dataloader, epochs=20, lr=0.001, latent_dim=128, run_dir=None, sample_count=20, sample_nrow=10):
    device = get_device()
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_log = []
    samples_dir = os.path.join(run_dir, "samples") if run_dir else "samples"
    checkpoint_path = os.path.join(run_dir, "vae_checkpoint.pth") if run_dir else "vae_checkpoint.pth"

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for imgs, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(Loss=loss.item()/imgs.size(0))

        avg = total_loss / len(dataloader.dataset)
        loss_log.append(avg)
        print(f"VAE Epoch {epoch+1:02d}/{epochs} | loss: {avg:.4f}")

        torch.save(model.state_dict(), checkpoint_path)
        with torch.no_grad():
            sample_z = torch.randn(sample_count, latent_dim, device=device)
            save_samples(
                model.decoder(sample_z),
                os.path.join(samples_dir, f"vae_epoch_{epoch+1}.png"),
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

    return model