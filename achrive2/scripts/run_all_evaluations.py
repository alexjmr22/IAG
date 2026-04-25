#!/usr/bin/env python3
"""Run evaluation for all experiment folders under results/.

Usage examples:
  # evaluate only VAEs, skipping experiments already evaluated
  python3 scripts/run_all_evaluations.py --target VAE

  # evaluate all targets and force re-evaluation (override existing results)
  python3 scripts/run_all_evaluations.py --target ALL --force

This script calls scripts/04_evaluation.py with `EXP_NAME` set to each
experiment folder. It skips folders without checkpoints and (by default)
skips models already present in results.csv unless --force is given.
"""
import argparse
import os
import subprocess
from pathlib import Path
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default='results', help='Root results directory')
    p.add_argument('--target', type=str, default='ALL', choices=['ALL', 'VAE', 'DCGAN', 'Diffusion'])
    p.add_argument('--force', action='store_true', help='Force re-evaluation even if results exist')
    p.add_argument('--python', type=str, default=sys.executable, help='Python executable to run evaluation')
    return p.parse_args()


def has_checkpoint(folder: Path, target: str) -> bool:
    mapping = {'VAE': 'vae_checkpoint.pth', 'DCGAN': 'dcgan_checkpoint.pt', 'Diffusion': 'diffusion_checkpoint.pth'}
    if target == 'ALL':
        return any((folder / v).exists() for v in mapping.values())
    return (folder / mapping[target]).exists()


def model_present_in_results(results_csv: Path, model_name: str) -> bool:
    if not results_csv.exists():
        return False
    try:
        df = pd.read_csv(results_csv)
        return model_name in df['model'].values
    except Exception:
        return False


def main():
    args = parse_args()
    root = Path(args.results_dir)
    if not root.exists():
        print(f"Results dir not found: {root}")
        return

    targets = ['VAE', 'DCGAN', 'Diffusion'] if args.target == 'ALL' else [args.target]

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == 'evaluation':
            continue

        # skip if no relevant checkpoints
        if not has_checkpoint(child, args.target):
            print(f"Skipping {child.name}: no checkpoint for target {args.target}")
            continue

        # if target is ALL we run a single call with EVAL_TARGET=ALL
        if args.target == 'ALL':
            results_csv = child / 'results.csv'
            if results_csv.exists() and (not args.force):
                print(f"Skipping {child.name}: results.csv exists (use --force to override)")
                continue

            env = os.environ.copy()
            env.update({'EXP_NAME': child.name, 'EVAL_TARGET': 'ALL', 'FORCE_EVAL': '1' if args.force else '0'})
            print(f"Running evaluation for {child.name} (ALL)")
            subprocess.run([args.python, 'scripts/04_evaluation.py'], env=env)
            continue

        # otherwise iterate requested targets individually
        for t in targets:
            ckpt = {'VAE': 'vae_checkpoint.pth', 'DCGAN': 'dcgan_checkpoint.pt', 'Diffusion': 'diffusion_checkpoint.pth'}[t]
            if not (child / ckpt).exists():
                print(f"Skipping {child.name}/{t}: no checkpoint")
                continue

            results_csv = child / 'results.csv'
            if model_present_in_results(results_csv, t) and (not args.force):
                print(f"Skipping {child.name}/{t}: already present in results.csv")
                continue

            env = os.environ.copy()
            env.update({'EXP_NAME': child.name, 'EVAL_TARGET': t, 'FORCE_EVAL': '1' if args.force else '0'})
            print(f"Running evaluation for {child.name}/{t}")
            subprocess.run([args.python, 'scripts/04_evaluation.py'], env=env)


if __name__ == '__main__':
    main()
