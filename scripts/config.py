import os

# Default to "PROD" if not exported
PROFILE_NAME = os.environ.get('RUN_PROFILE', 'PROD').upper()

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

PROFILES = {
    'TEST': {
        # Testing pipeline: incredibly fast runs to ensure no crashes
        'vae_epochs': 2,
        'dcgan_epochs': 2,
        'diffusion_epochs': 2,
        
        'use_subset': True,       # Use 20% subset to be fast
        'save_samples': False,    # Don't save image grids to disk during intermediary epochs
        
        'eval_samples': 100,      # Generate and test only 100 images
        'eval_seeds': 1           # Only do 1 seed instead of 10
    },
    'DEV': {
        # Feature Engineering: moderate lengths to analyze image quality and intermediate metrics
        'vae_epochs': 30,
        'dcgan_epochs': 50,
        'diffusion_epochs': 50,
        
        'use_subset': True,       # Use 20% subset for faster iterative cycle
        'save_samples': True,     # Save samples so you can see if it's learning (grid art)
        
        # Stronger evaluation for feature engineering: more samples + seeds
        'eval_samples': 2000,
        'eval_seeds': 3
    },
    'PROD': {
        # Final Report: strict compliance with Project Prompt (Enunciado)
        'vae_epochs': 50,         # Minimum requested
        'dcgan_epochs': 100,      # Minimum requested
        'diffusion_epochs': 100,  # Or more if needed
        
        'use_subset': False,      # MUST USE FULL ArtBench-10 (100%)
        'save_samples': True,     # Output beautiful final training plots
        
        'eval_samples': 5000,     # Enunciado mandatory (5000 generated vs 5000 real)
        'eval_seeds': 10          # Enunciado mandatory (run 10 times for mean and std)
    }
}

cfg = Config(**PROFILES.get(PROFILE_NAME, PROFILES['PROD']))

# Override com variáveis de ambiente se definidas
if 'VAE_EPOCHS' in os.environ:
    cfg.vae_epochs = int(os.environ['VAE_EPOCHS'])
if 'DCGAN_EPOCHS' in os.environ:
    cfg.dcgan_epochs = int(os.environ['DCGAN_EPOCHS'])
if 'DIFFUSION_EPOCHS' in os.environ:
    cfg.diffusion_epochs = int(os.environ['DIFFUSION_EPOCHS'])

print(f"-> Profile: [{PROFILE_NAME}] carregado!")
print(f"-> VAE Epochs: {cfg.vae_epochs} (env override: {'VAE_EPOCHS' in os.environ})")
