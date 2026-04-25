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
    '4': [ # PC 4 — DCGAN follow-up: combinar melhores achados + assimetria G/D
        # ── Hipótese 1: combinar os dois melhores individuais (lat32 + ngf128) ──
        {'id': 'dcgan_lat32_ngf128',     'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128'}},

        # ── Hipótese 2: preencher o gap da curva latent_dim (32 → 64 → 100) ──
        {'id': 'dcgan_lat64',            'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '64'}},

        # ── Hipótese 3: LR assimétrico (G mais lento, D mais rápido) ──
        # Teoria: D mais bem treinado = gradientes mais informativos para G
        {'id': 'dcgan_asym_lr',          'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},

        # ── Hipótese 4: lat32 com LR assimétrico ──
        {'id': 'dcgan_lat32_asym_lr',    'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},

        # ── Hipótese 5: capacidade assimétrica G/D (G maior, D default) ──
        # ngf128 com ndf=64: gerador mais expressivo sem tornar D demasiado poderoso
        {'id': 'dcgan_ngf128_ndf64',     'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '64'}},

        # ── Hipótese 6: melhor combo completo (lat32 + ngf128 + LR assimétrico) ──
        {'id': 'dcgan_lat32_ngf128_asym','target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},
    ],
    '6': [ # PC 6 — DCGAN melhorias: Cosine, Spectral Norm, WGAN-GP, cDCGAN
        # base: parâmetros do ngf128_100ep (melhor modelo encontrado)
        {'id': 'dcgan_cosine',     'target': 'DCGAN',  'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_COSINE': '1'}},
        {'id': 'dcgan_spectral',   'target': 'DCGAN',  'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_SPECTRAL': '1'}},
        {'id': 'dcgan_cosine_sn',  'target': 'DCGAN',  'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_COSINE': '1', 'DCGAN_SPECTRAL': '1'}},
        {'id': 'wgan_gp',          'target': 'WGAN',   'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},
        {'id': 'cdcgan',           'target': 'cDCGAN', 'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},
    ],
    '5': [ # PC 5 — DCGAN follow-up v2: mais épocas nos top-3 + combo em falta
        # ── Top performers com 100 épocas (curvas ainda desciam em e50) ──
        {'id': 'dcgan_lat32_100ep',         'target': 'DCGAN', 'env': {'RUN_PROFILE': 'DEV', 'DCGAN_LATENT': '32',  'DCGAN_EPOCHS': '100'}},
        {'id': 'dcgan_ngf128_100ep',        'target': 'DCGAN', 'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},
        {'id': 'dcgan_ngf128_ndf64_100ep',  'target': 'DCGAN', 'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '64',  'DCGAN_EPOCHS': '100'}},

        # ── Combo não testado: lat32 + gerador grande + D mais fraco ──
        # Hipótese: lat32 resolve o bottleneck de compressão; ndf64 evita D demasiado dominante
        {'id': 'dcgan_lat32_ngf128_ndf64',  'target': 'DCGAN', 'env': {'RUN_PROFILE': 'DEV', 'DCGAN_LATENT': '32',  'DCGAN_NGF': '128', 'DCGAN_NDF': '64'}},
    ],
    '8': [ # PC 8 — StyleGAN: exploração sistemática de hiperparâmetros
        # ── Fase 1: Baseline comparável (re-run limpo com eval correcto) ────────
        # Comparação directa com dcgan_spectral (FID=71.17, mesmo ngf e epochs)
        {'id': 'stylegan_default', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '64', 'DCGAN_NDF': '64',
            'DCGAN_EPOCHS': '100'}},

        {'id': 'stylegan_ngf128', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100'}},

        # ── Fase 2: Melhor combo conhecido — mais epochs ─────────────────────
        # dcgan_spectral_200ep (FID=60) prova que 200ep compensa muito.
        # Aplica o mesmo raciocínio ao StyleGAN com ngf=128.
        {'id': 'stylegan_ngf128_200ep', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '200'}},

        # ── Fase 3: Exploração específica do StyleGAN ────────────────────────

        # Mapping network mais profunda (paper usa 8 camadas; nós usamos 4).
        # Mais camadas → W mais disentangled → FID potencialmente melhor.
        {'id': 'stylegan_map8', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100', 'STYLEGAN_MAP_LAYERS': '8'}},

        # Espaço W maior (256 vs 128).
        # Mais capacidade de representação em W → possivelmente melhor FID
        # mas pode ser mais difícil de treinar.
        {'id': 'stylegan_wdim256', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100', 'STYLEGAN_WDIM': '256'}},

        # Style mixing desligado — mede o custo/benefício desta regularização.
        # Paper: mixing 90% melhora robustez mas pode custar 0.1–0.3 FID global.
        {'id': 'stylegan_nomix', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100', 'STYLEGAN_MIX_PROB': '0.0'}},

        # R1 mais suave (γ=1 vs γ=10 default).
        # R1 alto pode sobre-regularizar o discriminador → gerador aprende menos.
        {'id': 'stylegan_r1gamma1', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100', 'STYLEGAN_R1_GAMMA': '1.0'}},
    ],
    '7': [ # PC 7 — Novos modelos: StyleGAN, WGAN diagnóstico, DCGAN mais epochs
        # ── 1. StyleGAN baseline ──────────────────────────────────────────────
        {'id': 'stylegan_default', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '64', 'DCGAN_NDF': '64',
            'DCGAN_EPOCHS': '100'}},

        # ── 2. StyleGAN com maior capacidade (ngf=128) ────────────────────────
        {'id': 'stylegan_ngf128', 'target': 'StyleGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100'}},

        # ── 3. WGAN diagnóstico: remover Cosine LR (hipótese principal de degradação) ──
        {'id': 'wgan_no_cosine', 'target': 'WGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100', 'DCGAN_LR_G': '2e-4', 'DCGAN_LR_D': '2e-4',
            'DCGAN_BETA1': '0.5', 'WGAN_COSINE': '0'}},

        # ── 4. WGAN com N_CRITIC=2 e sem Cosine (menos overhead) ──
        {'id': 'wgan_ncritic2', 'target': 'WGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '100', 'DCGAN_LR_G': '2e-4', 'DCGAN_LR_D': '2e-4',
            'DCGAN_BETA1': '0.5', 'WGAN_COSINE': '0', 'WGAN_N_CRITIC': '2'}},

        # ── 5. dcgan_spectral com 200 epochs ─────────────────────────────────
        {'id': 'dcgan_spectral_200ep', 'target': 'DCGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128',
            'DCGAN_EPOCHS': '200', 'DCGAN_SPECTRAL': '1'}},

        # ── 6. SN + latent=32 (combo nunca testado) ───────────────────────────
        {'id': 'dcgan_spectral_lat32', 'target': 'DCGAN', 'env': {
            'RUN_PROFILE': 'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128',
            'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_SPECTRAL': '1'}},
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
    parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5', '6', '7', '8'], help="ID do Computador (1–8)")
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
        elif target == 'WGAN':
            run_script(root_dir / 'scripts' / '05_wgan_gp.py', exp_env)
            # gerador usa arquitectura DCGenerator — eval como DCGAN
            exp_env['EVAL_TARGET'] = 'DCGAN'
        elif target == 'cDCGAN':
            run_script(root_dir / 'scripts' / '06_cdcgan.py', exp_env)
        elif target == 'StyleGAN':
            run_script(root_dir / 'scripts' / '07_stylegan.py', exp_env)
            exp_env['EVAL_TARGET'] = 'StyleGAN'

        # Avaliação comum obrigatória logo após qualquer treino (o eval deteta o EVAL_TARGET e a EXP_NAME)
        run_script(root_dir / 'scripts' / '04_evaluation.py', exp_env)

    print("\n\n=======================================================")
    print(" 🎉 TODAS AS EXPERIÊNCIAS DO PC CONCLUÍDAS COM SUCESSO! 🎉")
    print("=======================================================")

if __name__ == "__main__":
    main()
