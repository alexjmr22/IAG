import os
import subprocess
import sys
import argparse
from pathlib import Path

EXPERIMENTS = {
    '1': [ # PC 1 — VAE/CVAE/VQ-VAE
        # Basic Sweeps
        {'id': 'default_vae',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'vae_lat16',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '16'}},
        {'id': 'vae_lat32',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '32'}},
        {'id': 'vae_lat64',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '64'}},
        {'id': 'vae_lat96',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '96'}},
        {'id': 'vae_lat128',    'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '128'}},
        {'id': 'vae_lat256',    'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '256'}},

        {'id': 'vae_beta0',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.0'}},
        {'id': 'vae_beta002', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.02'}},
        {'id': 'vae_beta005', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05'}},
        {'id': 'vae_beta01',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1'}},
        {'id': 'vae_beta015', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.15'}},
        {'id': 'vae_beta02',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.2'}},
        {'id': 'vae_beta05',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.5'}},
        {'id': 'vae_beta1',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '1.0'}},
        {'id': 'vae_beta2',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '2.0'}},

        {'id': 'vae_lr1e2', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-2'}},
        {'id': 'vae_lr5e3', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '5e-3'}},
        {'id': 'vae_lr2e3', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '2e-3'}},
        {'id': 'vae_lr1e3', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-3'}},
        {'id': 'vae_lr5e4', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '5e-4'}},
        {'id': 'vae_lr1e4', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-4'}},

        # Combos & Advanced
        {'id': 'vae_best_combo',                  'target': 'VAE',    'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1',  'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3'}},
        {'id': 'vae_combo_full',                  'target': 'VAE',    'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3'}},
        {'id': 'vae_r3_beta01_lat128_lr2e3_e50',  'target': 'VAE',    'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1',  'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50'}},
        {'id': 'vae_r5_t2_cosine',                'target': 'VAE',    'env': {'RUN_PROFILE':'DEV', 'VAE_COSINE_LR': 'true'}},
        {'id': 'vae_r5_t3_kl_annealing',          'target': 'VAE',    'env': {'RUN_PROFILE':'DEV', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        {'id': 'vae_r5_t4_both',                  'target': 'VAE',    'env': {'RUN_PROFILE':'DEV', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        {'id': 'vae_r5_final_t5_perceptual_loss', 'target': 'VAE',    'env': {'RUN_PROFILE':'DEV', 'VAE_PERCEPTUAL_LOSS': 'true'}},
        {'id': 'cvae_r5_t5_conditional',          'target': 'CVAE',   'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        {'id': 'vq_vae_r5_t6_quantized',          'target': 'VQ_VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true'}},

        # Production
        {'id': 'vae_prod_champion_beta005', 'target': 'VAE', 'env': {'RUN_PROFILE':'PROD', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '200'}},
        {'id': 'vae_prod_runnerup_beta01',  'target': 'VAE', 'env': {'RUN_PROFILE':'PROD', 'VAE_BETA': '0.1',  'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '200'}},
    ],
    '2': [ # PC 2 — GAN
        # Basic Sweeps
        {'id': 'default_dcgan', 'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'dcgan_lat32',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32'}},
        {'id': 'dcgan_lat64',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '64'}},
        {'id': 'dcgan_lat100',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '100'}},
        {'id': 'dcgan_ngf64',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '64',  'DCGAN_NDF': '64'}},
        {'id': 'dcgan_ngf128',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128'}},
        {'id': 'dcgan_beta09',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_BETA1': '0.9'}},
        {'id': 'dcgan_lr2e4',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LR': '2e-4'}},

        # Asymmetric LR & Combos
        {'id': 'dcgan_lat32_ngf128',      'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128'}},
        {'id': 'dcgan_asym_lr',           'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},
        {'id': 'dcgan_lat32_asym_lr',     'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},
        {'id': 'dcgan_ngf128_ndf64',      'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '64'}},
        {'id': 'dcgan_lat32_ngf128_asym', 'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},

        # Spectral Norm, WGAN-GP, cDCGAN
        {'id': 'dcgan_cosine',   'target': 'DCGAN',  'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_COSINE': '1'}},
        {'id': 'dcgan_spectral', 'target': 'DCGAN',  'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_SPECTRAL': '1'}},
        {'id': 'dcgan_cosine_sn','target': 'DCGAN',  'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_COSINE': '1', 'DCGAN_SPECTRAL': '1'}},
        {'id': 'wgan_gp',        'target': 'WGAN',   'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},
        {'id': 'cdcgan',         'target': 'cDCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},

        # StyleGAN
        {'id': 'stylegan_default',    'target': 'StyleGAN', 'env': {'RUN_PROFILE':'DEV',  'DCGAN_NGF': '64',  'DCGAN_NDF': '64',  'DCGAN_EPOCHS': '100'}},
        {'id': 'stylegan_ngf128',     'target': 'StyleGAN', 'env': {'RUN_PROFILE':'DEV',  'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},
        {'id': 'stylegan_map8',       'target': 'StyleGAN', 'env': {'RUN_PROFILE':'DEV',  'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'STYLEGAN_MAP_LAYERS': '8'}},
        {'id': 'stylegan_wdim256',    'target': 'StyleGAN', 'env': {'RUN_PROFILE':'DEV',  'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'STYLEGAN_WDIM': '256'}},
        {'id': 'stylegan_nomix',      'target': 'StyleGAN', 'env': {'RUN_PROFILE':'DEV',  'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'STYLEGAN_MIX_PROB': '0.0'}},
        {'id': 'stylegan_r1gamma1',   'target': 'StyleGAN', 'env': {'RUN_PROFILE':'DEV',  'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'STYLEGAN_R1_GAMMA': '1.0'}},
        {'id': 'stylegan_ngf128_200ep','target': 'StyleGAN', 'env': {'RUN_PROFILE':'PROD', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '200'}},

        # Diagnostics
        {'id': 'wgan_no_cosine',          'target': 'WGAN',  'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_LR_G': '2e-4', 'DCGAN_LR_D': '2e-4', 'DCGAN_BETA1': '0.5', 'WGAN_COSINE': '0'}},
        {'id': 'wgan_ncritic2',           'target': 'WGAN',  'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_LR_G': '2e-4', 'DCGAN_LR_D': '2e-4', 'DCGAN_BETA1': '0.5', 'WGAN_COSINE': '0', 'WGAN_N_CRITIC': '2'}},
        {'id': 'dcgan_spectral_lat32',    'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_SPECTRAL': '1'}},
        {'id': 'dcgan_lat32_100ep',       'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32',  'DCGAN_EPOCHS': '100'}},
        {'id': 'dcgan_ngf128_100ep',      'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},
        {'id': 'dcgan_ngf128_ndf64_100ep','target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '64',  'DCGAN_EPOCHS': '100'}},
        {'id': 'dcgan_lat32_ngf128_ndf64','target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '64'}},

        # Production
        {'id': 'dcgan_spectral_200ep', 'target': 'DCGAN', 'env': {'RUN_PROFILE':'PROD', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_LR': '2e-4', 'DCGAN_EPOCHS': '200'}},
    ],
    '3': [ # PC 3 — Diffusion
        # Basic Sweeps
        {'id': 'default_diff', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'diff_T100',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '100'}},
        {'id': 'diff_T250',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '250'}},
        {'id': 'diff_T500',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '500'}},
        {'id': 'diff_ch32',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '32'}},
        {'id': 'diff_ch64',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '64'}},
        {'id': 'diff_ch96',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96'}},
        {'id': 'diff_ch128',   'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '128'}},
        {'id': 'diff_lr1e3',   'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '1e-3'}},
        {'id': 'diff_lr2e5',   'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '2e-5'}},
        {'id': 'diff_lr5e5',   'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '5e-5'}},

        # Timesteps & Beta
        {'id': 'diff_best_combo',     'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4'}},
        {'id': 'diff_T1500',          'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1500'}},
        {'id': 'diff_T2000',          'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000'}},
        {'id': 'diff_T2000_ch64',     'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4'}},
        {'id': 'diff_beta_high',      'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.04'}},
        {'id': 'diff_beta_low',       'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.01'}},
        {'id': 'diff_beta_low_combo', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4', 'DIFF_BETA_END': '0.01'}},
        {'id': 'diff_combo_v2',       'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4', 'DIFF_BETA_END': '0.01'}},

        # Architecture & Scheduling
        {'id': 'diff_ch112',              'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '112', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4'}},
        {'id': 'diff_ch96_e100',          'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96',  'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100'}},
        {'id': 'diff_best_combo_e100',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '64',  'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100'}},
        {'id': 'diff_ch96_cosine',        'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5'}},
        {'id': 'diff_ch64_cosine',        'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '64', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5'}},
        {'id': 'diff_ch96_cosine_lr4e4',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5'}},

        # Production
        {'id': 'diff_prod_ddim_e100', 'target': 'Diffusion',    'env': {'RUN_PROFILE':'PROD', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5', 'DIFF_SAMPLER': 'ddim', 'DIFF_DDIM_STEPS': '100'}},
        {'id': 'diff_ema_e100',       'target': 'DiffusionEMA', 'env': {'RUN_PROFILE':'PROD', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5', 'DIFF_SAMPLER': 'ddim', 'DIFF_DDIM_STEPS': '100', 'DIFF_EMA_DECAY': '0.9999'}},
        {'id': 'diff_prod_ddim_e200', 'target': 'DiffusionEMA', 'env': {'RUN_PROFILE':'PROD', 'DIFF_CHANNELS': '96', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '200', 'DIFF_SAMPLER': 'ddim', 'DIFF_DDIM_STEPS': '100', 'DIFF_EMA_DECAY': '0.9999'}},
    ]
}

_ALL_EXP_VARS = [
    # VAE
    'VAE_LATENT_DIM', 'VAE_BETA', 'VAE_LR', 'VAE_EPOCHS',
    'VAE_COSINE_LR', 'VAE_KL_ANNEALING_EPOCHS', 'VAE_PERCEPTUAL_LOSS',
    # GAN
    'DCGAN_LATENT', 'DCGAN_NGF', 'DCGAN_NDF', 'DCGAN_BETA1',
    'DCGAN_LR', 'DCGAN_LR_G', 'DCGAN_LR_D', 'DCGAN_EPOCHS',
    'DCGAN_COSINE', 'DCGAN_SPECTRAL',
    'WGAN_COSINE', 'WGAN_N_CRITIC',
    'STYLEGAN_MAP_LAYERS', 'STYLEGAN_WDIM', 'STYLEGAN_MIX_PROB', 'STYLEGAN_R1_GAMMA',
    # Diffusion
    'DIFF_CHANNELS', 'DIFF_T_STEPS', 'DIFF_LR', 'DIFF_EPOCHS',
    'DIFF_BETA_START', 'DIFF_BETA_END', 'DIFF_WARMUP_EPOCHS',
    'DIFF_SAMPLER', 'DIFF_DDIM_STEPS', 'DIFF_EMA_DECAY',
]

_SCRIPT_MAP = {
    'VAE':         '01_vae.py',
    'CVAE':        '03_cvae.py',
    'VQ_VAE':      '04_vq_vae.py',
    'DCGAN':       '02_dcgan.py',
    'WGAN':        '05_wgan_gp.py',
    'cDCGAN':      '06_cdcgan.py',
    'StyleGAN':    '07_stylegan.py',
    'Diffusion':   '03_diffusion.py',
    'DiffusionEMA':'03b_diffusion_ema.py',
}


def run_script(script_path, extra_env):
    python_exe = sys.executable
    if '.venv' not in python_exe:
        venv_python = Path(os.getcwd()) / '.venv' / 'Scripts' / 'python.exe'
        if venv_python.exists():
            python_exe = str(venv_python)

    env = os.environ.copy()
    for v in _ALL_EXP_VARS:
        env.pop(v, None)
    env.update(extra_env)

    print(f"\n[{extra_env.get('EXP_NAME')}] >> Running {script_path.name}...")
    result = subprocess.run([python_exe, str(script_path)], env=env)
    if result.returncode != 0:
        print(f"!!! FAILURE in {script_path.name} under {extra_env.get('EXP_NAME')} — continuing.")
    else:
        print(f"[{extra_env.get('EXP_NAME')}] << {script_path.name} OK.")


def main():
    parser = argparse.ArgumentParser(description='Grid Search Orchestrator')
    parser.add_argument('--pc', required=True, choices=['1', '2', '3'], help='PC ID (1=VAE, 2=GAN, 3=Diff)')
    args = parser.parse_args()

    experiments = EXPERIMENTS[args.pc]
    print(f"\n{'='*55}")
    print(f"  PC {args.pc} — {len(experiments)} experiments")
    print(f"{'='*55}\n")

    scripts_dir = Path(__file__).resolve().parent

    for exp in experiments:
        print(f"\n--- {exp['id']} ---")
        exp_env = exp['env'].copy()
        exp_env['EXP_NAME'] = exp['id']
        exp_env['EVAL_TARGET'] = exp['target']

        target = exp['target']
        run_script(scripts_dir / _SCRIPT_MAP[target], exp_env)

        if target == 'WGAN':
            exp_env['EVAL_TARGET'] = 'DCGAN'

        run_script(scripts_dir / '04_evaluation.py', exp_env)

    print(f"\n{'='*55}")
    print(f"  PC {args.pc} — todas as experiências concluídas.")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
