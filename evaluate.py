import torch
import numpy as np
import os
import csv
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from IAG.dataset import get_dataloader
from IAG.utils import get_device, set_seed


def calculate_metrics(model, model_type, data_root, num_samples=5000, batch_size=50):
    device = get_device()
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=100).to(device)

    loader = get_dataloader(data_root, batch_size=batch_size, train=False)
    count = 0
    for imgs, _ in loader:
        real_imgs = ((imgs.to(device) + 1) / 2 * 255).byte()
        fid.update(real_imgs, real=True)
        kid.update(real_imgs, real=True)
        count += imgs.size(0)
        if count >= num_samples:
            break

    model.eval()
    with torch.no_grad():
        for _ in range(0, num_samples, batch_size):
            if model_type == "gan":
                z = torch.randn(batch_size, 100, device=device)
                l = torch.randint(0, 10, (batch_size,), device=device)
                fake = model(z, l)
            elif model_type == "vae":
                z = torch.randn(batch_size, 128, device=device)
                fake = model.decoder(z)
            else:  # diffusion
                fake = model(torch.randn(batch_size, 3, 32, 32, device=device), None)

            fake_imgs = ((fake + 1) / 2 * 255).byte()
            fid.update(fake_imgs, real=False)
            kid.update(fake_imgs, real=False)

    return fid.compute().item(), kid.compute()[0].item()


def _append_results_row(csv_file, header, row_dict):
    """Append a row to csv_file, migrating existing header if needed."""
    if not os.path.isfile(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow([row_dict.get(col, "") for col in header])
        return

    with open(csv_file, mode="r", newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, [])

    if existing_header != header:
        with open(csv_file, mode="r", newline="") as f:
            dict_reader = csv.DictReader(f)
            existing_rows = list(dict_reader)

        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in existing_rows:
                writer.writerow([r.get(col, "") for col in header])

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row_dict.get(col, "") for col in header])


def run_full_evaluation(
    model,
    model_type,
    data_root,
    run_dir=None,
    num_runs=10,
    num_samples=5000,
    batch_size=50,
):
    print(
        f"\nRunning full evaluation ({num_runs} runs, {num_samples} samples, batch={batch_size}) for {model_type}..."
    )
    fids, kids = [], []

    for i in range(num_runs):
        set_seed(42 + i)
        f, k = calculate_metrics(model, model_type, data_root, num_samples=num_samples, batch_size=batch_size)
        fids.append(f)
        kids.append(k)
        print(f"Run {i+1}: FID={f:.2f}, KID={k:.4f}")

    fid_mean, fid_std = np.mean(fids), np.std(fids)
    kid_mean, kid_std = np.mean(kids), np.std(kids)

    # Save per-seed metrics to run_dir
    if run_dir is not None:
        metrics_path = os.path.join(run_dir, "metrics.csv")
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "FID", "KID"])
            for i, (fv, kv) in enumerate(zip(fids, kids)):
                writer.writerow([42 + i, f"{fv:.4f}", f"{kv:.4f}"])
            writer.writerow([])
            writer.writerow(["mean", f"{fid_mean:.4f}", f"{kid_mean:.4f}"])
            writer.writerow(["std", f"{fid_std:.4f}", f"{kid_std:.4f}"])
        print(f"Per-seed metrics saved to {metrics_path}")

    # Append summary to global results.csv (migrates old headers automatically)
    from datetime import datetime
    csv_file = "results.csv"
    header = [
        "Timestamp",
        "Model",
        "RunDir",
        "NumRuns",
        "NumSamples",
        "BatchSize",
        "FID_Mean",
        "FID_Std",
        "KID_Mean",
        "KID_Std",
    ]
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_type.upper(),
        "RunDir": run_dir or "",
        "NumRuns": str(num_runs),
        "NumSamples": str(num_samples),
        "BatchSize": str(batch_size),
        "FID_Mean": f"{fid_mean:.4f}",
        "FID_Std": f"{fid_std:.4f}",
        "KID_Mean": f"{kid_mean:.4f}",
        "KID_Std": f"{kid_std:.4f}",
    }
    _append_results_row(csv_file, header, row)

    print(f"\nFinal: FID={fid_mean:.4f}±{fid_std:.4f}  KID={kid_mean:.4f}±{kid_std:.4f}")
    print(f"Summary saved to results.csv")