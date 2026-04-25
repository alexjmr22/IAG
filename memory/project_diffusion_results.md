---
name: Diffusion Experiment Results
description: FID/KID for all Diffusion ablations; best overall: ch=96, cosine LR 4e-4, 100ep, full dataset, DDIM (FID=32.17, KID=0.0169)
type: project
---

# Diffusion Results Summary

## Best result: diff_prod_ddim_e100
**FID=32.17 | KID=0.0169**

Config:
- Channels: 96
- T Steps: 1000
- LR: 4e-4 (cosine annealing, 5-epoch linear warmup)
- Epochs: 100
- Batch Size: 128
- Beta: 0.0001→0.02 (linear schedule)
- Sampler: DDIM (100 steps)
- Dataset: Full ArtBench10 (50k images)

**Why:** ch=96 won over ch=64/128 in sweep; cosine LR+warmup helped stability; full dataset vs 20% subset was biggest gain; DDIM for fast sampling.

## Test 7 (DEV subset, cosine LR)
- diff_ch96_cosine: FID=65.73, KID=0.0353 — ch=96, LR=2e-4, cosine, 100ep, 20% subset
- diff_ch64_cosine: ~worse
- diff_ch96_cosine_lr4e4: better than 2e-4 — led to PROD config

## Test 4 (DEV subset, 50 epochs, no cosine)
- diff_ch96: FID≈100.2 (best of sweep 4)
- diff_best_combo (ch=64): FID≈107.2

## Test 3 (DEV subset sweeps, 50 epochs)
- ch128: FID≈187 (worse than ch=64/96)
- T=1000 + ch=64 + LR=2e-4: baseline anchor
