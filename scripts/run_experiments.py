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
    ],
    '4': [ # PC 4 — Diffusion best-combo + explorações direcionadas pelos sweeps
        {'id': 'diff_best_combo', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_CHANNELS': '64',  'DIFF_LR': '2e-4'}},
        {'id': 'diff_T2000',      'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000'}},
        {'id': 'diff_ch96',       'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96'}},
        {'id': 'diff_lr5e5',      'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '5e-5'}},
        {'id': 'diff_beta_high',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.04'}},
        {'id': 'diff_beta_low',   'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.01'}},
        {'id': 'diff_combo_v2',   'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '5e-5', 'DIFF_CHANNELS': '96'}},
    ],
    '5': [ # PC 5 — VAE best-combo + explorações direcionadas pelos sweeps
        # 1. Âncora: melhor beta + melhor latent dim individualmente
        {'id': 'vae_best_combo',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '64',  'VAE_LR': '1e-3'}},
        # 2. Beta — o sweep mostrou melhoria monotónica de 0.7→0.1; testar ainda mais baixo
        {'id': 'vae_beta005',      'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05'}},
        # 3. Beta — zona entre 0.1 e 0.5 nunca explorada; testar 0.2
        {'id': 'vae_beta02',       'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.2'}},
        # 4. LR — tendência ainda crescente em 5e-3, topo não encontrado; testar 1e-2
        {'id': 'vae_lr1e2',        'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-2'}},
        # 5. LR — ponto intermédio entre o melhor (5e-3) e o default (1e-3)
        {'id': 'vae_lr2e3',        'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '2e-3'}},
        # 6. Latent dim — lat=64 ganhou, lat=128 e 256 estagnaram; testar 96 (meio-termo)
        {'id': 'vae_lat96',        'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '96'}},
        # 7. Combo completo: os três melhores parâmetros individuais combinados
        {'id': 'vae_combo_full',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '64',  'VAE_LR': '5e-3'}},
        # 8. Combo agressivo: aposta no LR mais alto + beta mínimo + lat=64
        {'id': 'vae_combo_bold',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '64', 'VAE_LR': '5e-3'}},
    ]
}

def run_script(script_path, extra_env):
    # Use the correct python executable from the virtual environment
    python_exe = sys.executable
    if ".venv" not in python_exe:
        # Fallback to local .venv if run from global python
        venv_python = Path(os.getcwd()) / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            python_exe = str(venv_python)

    # 1. Pega no ambiente atual do terminal
    env = os.environ.copy()
    # 2. Limpar variáveis de experimento para não contaminar runs sem essa var definida
    _exp_vars = [
        'VAE_LATENT_DIM', 'VAE_BETA', 'VAE_LR',
        'DCGAN_LATENT', 'DCGAN_NGF', 'DCGAN_NDF', 'DCGAN_BETA1', 'DCGAN_LR',
        'DIFF_CHANNELS', 'DIFF_T_STEPS', 'DIFF_LR', 'DIFF_BETA_START', 'DIFF_BETA_END',
    ]
    for v in _exp_vars:
        env.pop(v, None)
    # 3. SOBREPOSIÇÃO com os valores do experimento atual
    env.update(extra_env)
    print(f"\n[{extra_env.get('EXP_NAME')}] >> Running {script_path.name}...")
    
    result = subprocess.run([python_exe, str(script_path)], env=env)
    
    if result.returncode != 0:
        print(f"!!! CRITICAL FAILURE in {script_path.name} under {extra_env.get('EXP_NAME')} !!!")
        print(f"Proceeding to next experiment regardless.")
    else:
        print(f"[{extra_env.get('EXP_NAME')}] << Completed {script_path.name} SUCCESSFULLY.")


def main():
    parser = argparse.ArgumentParser(description="Grid Search Automated Orchestrator")
    parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5'], help="ID do Computador (1, 2, 3, 4, ou 5)")
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
