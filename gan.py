import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from IAG.utils import get_device, save_samples


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 50)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 50, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = torch.cat([z.view(z.size(0), -1, 1, 1), self.label_emb(labels).view(labels.size(0), -1, 1, 1)], 1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 32*32)
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        l = self.label_emb(labels).view(labels.size(0), 1, 32, 32)
        return self.model(torch.cat([x, l], 1)).view(-1, 1)


def train_gan(dataloader, epochs=20, lr=0.0002, latent_dim=100, run_dir=None, sample_count=20, sample_nrow=10):
    device = get_device()
    netG = Generator(latent_dim).to(device)
    netD = Discriminator().to(device)
    optG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    loss_log = []
    samples_dir = os.path.join(run_dir, "samples") if run_dir else "samples"
    checkpoint_path = os.path.join(run_dir, "gan_checkpoint.pth") if run_dir else "gan_checkpoint.pth"

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"GAN Epoch {epoch+1}/{epochs}")
        epoch_g, epoch_d, n = 0.0, 0.0, 0
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            bs = imgs.size(0)

            optD.zero_grad(set_to_none=True)
            out = netD(imgs, labels)
            loss_real = criterion(out, torch.ones(bs, 1, device=device))
            z = torch.randn(bs, latent_dim, device=device)
            gen_labels = torch.randint(0, 10, (bs,), device=device)
            fake = netG(z, gen_labels)
            out = netD(fake.detach(), gen_labels)
            loss_fake = criterion(out, torch.zeros(bs, 1, device=device))
            (loss_real + loss_fake).backward()
            optD.step()

            optG.zero_grad(set_to_none=True)
            out = netD(fake, gen_labels)
            lossG = criterion(out, torch.ones(bs, 1, device=device))
            lossG.backward()
            optG.step()

            epoch_g += lossG.item()
            epoch_d += (loss_real + loss_fake).item()
            n += 1
            pbar.set_postfix(G=lossG.item(), D=(loss_real+loss_fake).item())

        avg_g, avg_d = epoch_g / n, epoch_d / n
        loss_log.append((avg_g, avg_d))
        print(f"GAN Epoch {epoch+1:02d}/{epochs} | G: {avg_g:.4f}  D: {avg_d:.4f}")

        torch.save(netG.state_dict(), checkpoint_path)
        with torch.no_grad():
            sample_z = torch.randn(sample_count, latent_dim, device=device)
            sample_l = (torch.arange(sample_count, device=device) % 10).long()
            save_samples(
                netG(sample_z, sample_l),
                os.path.join(samples_dir, f"gan_epoch_{epoch+1}.png"),
                nrow=sample_nrow,
            )

    if run_dir is not None:
        loss_path = os.path.join(run_dir, "loss.csv")
        with open(loss_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "g_loss", "d_loss"])
            for i, (g, d) in enumerate(loss_log, 1):
                writer.writerow([i, f"{g:.6f}", f"{d:.6f}"])
        print(f"Loss history saved to {loss_path}")

    return netG