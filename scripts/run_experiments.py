import os
import subprocess
import sys
import argparse
from pathlib import Path

# --- THE MIGHTY GRID SEARCH ORCHESTRATOR --- #

EXPERIMENTS = {
    '1': [ # PC 1 — VAE feature-engineering sweep
        {'id': 'default_vae', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'vae_lat16',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '16'}},
        {'id': 'vae_lat32',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '32'}},
        {'id': 'vae_lat64',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '64'}},
        {'id': 'vae_lat128', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '128'}},
        {'id': 'vae_lat256', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '256'}},
        {'id': 'vae_beta0',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.0'}},
        {'id': 'vae_beta01', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1'}},
        {'id': 'vae_beta05', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.5'}},
        {'id': 'vae_beta2',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '2.0'}},
        {'id': 'vae_lr5e3',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '5e-3'}},
        {'id': 'vae_lr1e3',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-3'}},
        {'id': 'vae_lr5e4',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '5e-4'}},
        {'id': 'vae_lr1e4',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-4'}}
    ],
    '2': [ # PC 2 — DCGAN sweeps
        {'id': 'default_dcgan',    'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'dcgan_lat32',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32'}},
        {'id': 'dcgan_lat100',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '100'}},
        {'id': 'dcgan_lat256',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '256'}},
        {'id': 'dcgan_ngf32',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '32',  'DCGAN_NDF': '32'}},
        {'id': 'dcgan_ngf64',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '64',  'DCGAN_NDF': '64'}},
        {'id': 'dcgan_ngf128',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128'}},
        {'id': 'dcgan_beta09',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_BETA1': '0.9'}},
        {'id': 'dcgan_lr1e3',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LR': '1e-3'}},
        {'id': 'dcgan_lr2e4',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LR': '2e-4'}}
    ],
    '3': [ # PC 3 — Diffusion sweeps
        {'id': 'default_diff',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'diff_T100',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '100'}},
        {'id': 'diff_T250',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '250'}},
        {'id': 'diff_T500',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '500'}},
        {'id': 'diff_ch32',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '32'}},
        {'id': 'diff_ch64',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '64'}},
        {'id': 'diff_ch128', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '128'}},
        {'id': 'diff_lr1e3', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '1e-3'}},
        {'id': 'diff_lr2e5', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '2e-5'}}
    ]
}

def run_script(script_path, extra_env):
    # 1. Pega no ambiente atual do terminal (onde pode estar o RUN_PROFILE=TEST)
    env = os.environ.copy()
    # 2. SOBREPOSIÇÃO: Se a experiência definir um perfil específico, ele GANHA.
    env.update(extra_env)
    print(f"\n[{extra_env.get('EXP_NAME')}] >> Running {script_path.name}...")
    
    result = subprocess.run([sys.executable, str(script_path)], env=env)
    
    if result.returncode != 0:
        print(f"!!! CRITICAL FAILURE in {script_path.name} under {extra_env.get('EXP_NAME')} !!!")
        print(f"Proceeding to next experiment regardless.")
    else:
        print(f"[{extra_env.get('EXP_NAME')}] << Completed {script_path.name} SUCCESSFULLY.")


def main():
    parser = argparse.ArgumentParser(description="Grid Search Automated Orchestrator")
    parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3'], help="ID do Computador (1, 2, ou 3)")
    args = parser.parse_args()

    pc_experiments = EXPERIMENTS[args.pc]
    print(f"\n=======================================================")
    print(f"  INICIANDO BATERIA DE TESTES LABORATORIAIS NO PC {args.pc}")
    print(f"  TOTAL DE TESTES A CORRER: {len(pc_experiments)}")
    print(f"=======================================================\n")

    root_dir = Path(__file__).resolve().parent.parent

    for exp in pc_experiments:
        print(f"\n-------------------------------------------------------")
        print(f" \n\n🚀 EXPERIÊNCIA: {exp['id']}")
        # Inject standard required environment variables
        exp_env = exp['env'].copy()
        exp_env['EXP_NAME'] = exp['id']
        exp_env['EVAL_TARGET'] = exp['target']
        
        target = exp['target']
        
        if target == 'VAE':
            run_script(root_dir / 'scripts' / '01_vae.py', exp_env)
        elif target == 'DCGAN':
            run_script(root_dir / 'scripts' / '02_dcgan.py', exp_env)
        elif target == 'Diffusion':
            run_script(root_dir / 'scripts' / '03_diffusion.py', exp_env)
            
        # Avaliação comum obrigatória logo após qualquer treino (o eval deteta o EVAL_TARGET e a EXP_NAME)
        run_script(root_dir / 'scripts' / '04_evaluation.py', exp_env)

    print("\n\n=======================================================")
    print(" 🎉 TODAS AS EXPERIÊNCIAS DO PC CONCLUÍDAS COM SUCESSO! 🎉")
    print("=======================================================")

if __name__ == "__main__":
    main()
