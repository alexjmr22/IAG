import argparse
import os
import json
from datetime import datetime
from IAG.utils import set_seed
from IAG.dataset import get_dataloader
from IAG.gan import train_gan
from IAG.vae import train_vae
from IAG.diffusion import train_diffusion


def make_run_dir(model, epochs, lr, batch_size):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{epochs}ep_lr{lr}_bs{batch_size}"
    run_dir = os.path.join("results", model, run_name)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="ArtBench Generative Pipeline")
    parser.add_argument("model", choices=["gan", "vae", "diffusion"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")

    # Sampling (images saved during training)
    parser.add_argument("--sample_count", type=int, default=20, help="Override number of images saved per epoch")
    parser.add_argument("--sample_nrow", type=int, default=10, help="Grid columns when saving sample images")

    # Evaluation
    parser.add_argument("--eval_num_runs", type=int, default=10)
    parser.add_argument("--eval_num_samples", type=int, default=5000)
    parser.add_argument("--eval_batch_size", type=int, default=50)

    args = parser.parse_args()
    set_seed(args.seed)

    default_lrs = {"gan": 0.0002, "vae": 0.001, "diffusion": 0.0001}
    lr = args.lr or default_lrs[args.model]

    run_dir = make_run_dir(args.model, args.epochs, lr, args.batch_size)

    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "seed": args.seed,
        "data_root": args.data_root,
        "sample_count": args.sample_count,
        "sample_nrow": args.sample_nrow,
        "eval_num_runs": args.eval_num_runs,
        "eval_num_samples": args.eval_num_samples,
        "eval_batch_size": args.eval_batch_size,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Starting {args.model.upper()} pipeline... Run dir: {run_dir}")
    dataloader = get_dataloader(args.data_root, batch_size=args.batch_size, train=True)

    train_kwargs = {
        "run_dir": run_dir,
        "sample_nrow": args.sample_nrow,
        "sample_count": args.sample_count,
    }

    if args.model == "gan":
        model = train_gan(dataloader, epochs=args.epochs, lr=lr, **train_kwargs)
    elif args.model == "vae":
        model = train_vae(dataloader, epochs=args.epochs, lr=lr, **train_kwargs)
    elif args.model == "diffusion":
        model = train_diffusion(dataloader, epochs=args.epochs, lr=lr, **train_kwargs)

    print("\nStarting Evaluation (FID, KID, 10 Runs, 5000 Samples)...")
    from IAG.evaluate import run_full_evaluation
    run_full_evaluation(
        model,
        args.model,
        args.data_root,
        run_dir=run_dir,
        num_runs=args.eval_num_runs,
        num_samples=args.eval_num_samples,
        batch_size=args.eval_batch_size,
    )
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
