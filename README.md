# ArtBench Generative Project (Simple Version)

Minimal structure for training and evaluating GAN, VAE, and Diffusion on ArtBench-10.

## Project Structure

- `dataset.py`: ArtBench-10 data loader.
- `gan.py`: Conditional DCGAN architecture and training logic.
- `vae.py`: Variational Autoencoder architecture and training logic.
- `diffusion.py`: Minimal Diffusion model architecture and training logic.
- `evaluate.py`: FID calculation helper.
- `utils.py`: Common helpers (device, seed, saving).
- `pipeline.py`: Main entry point.

## Usage

### Training & Eval

```bash
python pipeline.py gan --epochs 20
python pipeline.py vae --epochs 20
python pipeline.py diffusion --epochs 20
```

### Examples

Generate the same number of sample images per epoch (useful for fair visual comparison):

```bash
python pipeline.py gan --epochs 20 --sample_count 16 --sample_nrow 4
python pipeline.py diffusion --epochs 20 --sample_count 16 --sample_nrow 4
```

Run a faster evaluation during debugging:

```bash
python pipeline.py gan --epochs 2 --eval_num_runs 3 --eval_num_samples 1000 --eval_batch_size 50
```

### Options

- `--epochs`: Number of iterations (Default: 20).
- `--batch_size`: Defaults to 64.
- `--seed`: Defaults to 42.

#### Sampling (images saved during training)

- `--sample_count`: How many images are generated and saved per epoch into `results/<model>/<run>/samples/*.png`.
  - If omitted, each model keeps its own default (GAN=10, VAE=64, Diffusion=16).
- `--sample_nrow`: Number of columns used when saving the image grid (passed to `torchvision.utils.save_image`).

#### Evaluation (FID/KID)

- `--eval_num_runs`: Number of repeated evaluation runs (seeds 42..42+N-1) used to compute mean/std.
- `--eval_num_samples`: Number of real/fake images used to compute FID/KID per run.
- `--eval_batch_size`: Batch size used during evaluation/generation.

## Output

- `gan_checkpoint.pth`: Model weights.
- `vae_checkpoint.pth`: Model weights.
- `diffusion_checkpoint.pth`: Model weights.
- `samples/`: Visual samples generated during training.

### Notes

- The global `results.csv` now stores the evaluation settings used (`NumRuns`, `NumSamples`, `BatchSize`) and will auto-migrate older headers.
