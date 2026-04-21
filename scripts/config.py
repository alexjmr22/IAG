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
        'vae_epochs': 10,
        'dcgan_epochs': 20,
        'diffusion_epochs': 20,
        
        'use_subset': True,       # Use 20% subset for faster iterative cycle
        'save_samples': True,     # Save samples so you can see if it's learning (grid art)
        
        'eval_samples': 1000,     # Enough to get a rough idea of FID
        'eval_seeds': 2           # Do 2 seeds to see variance
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

print(f"-> Profile: [{PROFILE_NAME}] carregado!")
