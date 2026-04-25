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
        # Âncora: melhor combo individual do sweep 3 (T=1000 e ch=64 empatam com default; lr=2e-4 é o único bom)
        {'id': 'diff_best_combo',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4'}},
        # T-steps: tendência monotónica clara no sweep (T100<T250<T500<T1000); explorar T=1500 e T=2000 para confirmar
        {'id': 'diff_T1500',         'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1500'}},
        {'id': 'diff_T2000',         'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000'}},
        # T=2000 + ch=64: combo da tendência mais forte com o melhor canal
        {'id': 'diff_T2000_ch64',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4'}},
        # ch96: channels — ch128 piorou (FID 187), mas vale testar intermédio entre 64 e 128
        {'id': 'diff_ch96',          'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96'}},
        # LR: 5e-5 — ponto entre o melhor (2e-4) e o pior (2e-5); o sweep não cobriu esta zona
        {'id': 'diff_lr5e5',         'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '5e-5'}},
        # Beta schedule: não explorado no sweep 3 — testar isolado para perceber sensibilidade
        {'id': 'diff_beta_high',     'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.04'}},
        {'id': 'diff_beta_low',      'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.01'}},
        # Beta low + melhores params: se beta_low ganhar isolado, este é o combo natural
        {'id': 'diff_beta_low_combo','target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4', 'DIFF_BETA_END': '0.01'}},
        # Combo v2: T=2000 + beta_low + ch=64 — aposta máxima se ambas as tendências forem positivas
        {'id': 'diff_combo_v2',      'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4', 'DIFF_BETA_END': '0.01'}},
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
    
        {'id': 'vae_beta1',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '1.0'}},

        # 7. Combo completo: os três melhores parâmetros individuais combinados
        {'id': 'vae_combo_full',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '64',  'VAE_LR': '5e-3'}},
        # 8. Combo agressivo: aposta no LR mais alto + beta mínimo + lat=64
        {'id': 'vae_combo_bold',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '64', 'VAE_LR': '5e-3'}},
    ],
    '6': [ # PC 6 — Ronda 3: VAE pragmatic tests (30 + 50 epochs)
        # TESTE 1: LR effect (β=0.1, lr=2e-3, e=30) — Compara com vae_beta01 (lr=1e-3)
        {'id': 'vae_r3_beta01_lat128_lr2e3_e30', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3'}},
        
        # TESTE 2: β refinement (β=0.15, lr=2e-3, e=30) — Posiciona β entre 0.1 e 0.2
        {'id': 'vae_r3_beta015_lat128_lr2e3_e30', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.15', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3'}},
        
        # TESTE 3: Epochs effect (β=0.1, lr=2e-3, e=50) — Valida se 50 epochs melhora vs 30
        {'id': 'vae_r3_beta01_lat128_lr2e3_e50', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50'}},
    ],
    '7': [ # PC 7 — Ronda 5: Schedulers + Nova Arquiteturas (β=0.1, lat=128, lr=2e-3, 50-100 ep)
        # T1: β-VAE Baseline (sem schedulers, 100 epochs para comparação)
        {'id': 'vae_r5_t1_baseline_100ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '100'}},
        
        # T2: β-VAE + Cosine Annealing LR (50 epochs)
        {'id': 'vae_r5_t2_cosine', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true'}},
        
        # T3: β-VAE + KL Annealing (50 epochs)
        {'id': 'vae_r5_t3_kl_annealing', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        
        # T4: β-VAE + Cosine + KL Annealing (Both, 50 epochs)
        {'id': 'vae_r5_t4_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        
        # T5: CVAE (Conditional VAE, 50 epochs)
        {'id': 'cvae_r5_t5_conditional', 'target': 'CVAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        
        # T6: VQ-VAE (Vector Quantized VAE, 50 epochs)
        {'id': 'vq_vae_r5_t6_quantized', 'target': 'VQ_VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true'}},
    ],
    '8': [ # PC 8 — Ronda 5 Final: 6 testes com fixes críticos (VAE_KL normalization, Cosine LR, KL Annealing, Perceptual Loss)
        # T1: β-VAE Baseline 150 epochs (FID esperado: 130-140)
        {'id': 'vae_r5_final_t1_baseline_150ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150'}},
        
        # T2: β-VAE + Cosine LR (FID esperado: 135-142)
        {'id': 'vae_r5_final_t2_cosine_100ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_COSINE_LR': 'true'}},
        
        # T3: β-VAE + KL Annealing (FIX para bug de warmup) (FID esperado: 128-138)
        {'id': 'vae_r5_final_t3_kl_annealing_fixed', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
        
        # T4: β-VAE + Cosine LR + KL Annealing (Both fixes) (FID esperado: 120-130)
        {'id': 'vae_r5_final_t4_cosine_kl_both', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15'}},
        
        # T5: β-VAE + Perceptual Loss (λ=0.1) (FID esperado: 115-125)
        {'id': 'vae_r5_final_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
        
        # T8: β-VAE + ALL Techniques (Cosine + KL Annealing + Perceptual) (FID esperado: 110-120)
        {'id': 'vae_r5_final_t8_all_techniques', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '15', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
    ],
    '9': [ # PC 9 — Ronda 5b: Corrected Baseline + Core Techniques (KL fixed, no Cosine)
        # T1: β-VAE Corrected Baseline (150 epochs, β=0.1, KL properly normalized)
        {'id': 'vae_r5_corrected_t1_baseline', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150'}},
        
        # T2: T1 + KL Annealing (35 epochs) — Bowman et al 2016 (prevents posterior collapse)
        {'id': 'vae_r5_corrected_t2_kl_annealing', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_KL_ANNEALING_EPOCHS': '35'}},
        
        # T3: T1 + Perceptual Loss (λ=0.1) — Johnson et al 2016 (improves visual quality)
        {'id': 'vae_r5_corrected_t3_perceptual', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
        
        # T4: T1 + Both Techniques (KL Annealing + Perceptual) — Combined improvements
        {'id': 'vae_r5_corrected_t4_both_techniques', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150', 'VAE_KL_ANNEALING_EPOCHS': '35', 'VAE_PERCEPTUAL_LOSS': '0.1'}},
    ],
    '10': [ # PC 10 — Ronda 5c: Simple β Tuning + Optimized Alternative Architectures (150 epochs VAE, 100 epochs alternatives, original KL)
        # T1: β=0.05 (fraco) — β mais pequeno porque KL original é já forte
        {'id': 'vae_r5_simple_t1_beta005', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150'}},
        
        # T2: β=0.1 (original) — baseline para comparação (confirmação de PC7 T1)
        {'id': 'vae_r5_simple_t2_beta01', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150'}},
        
        # T3: β=0.02 (muito fraco) — explorar regularização mínima
        {'id': 'vae_r5_simple_t3_beta002', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.02', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '150'}},
        
        # T4: CVAE Optimized (Conditional VAE, 100 epochs, β=0.15) — condicionalidade require β mais forte
        {'id': 'vae_r5_optimized_t4_cvae_beta015', 'target': 'CVAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.15', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '100'}},
        
        # T5: VQ-VAE Optimized (Vector Quantized VAE, 100 epochs, lr=5e-3) — quantização é mais estável com LR mais alto, sem β
        {'id': 'vae_r5_optimized_t5_vqvae_lr005', 'target': 'VQ_VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.005', 'VAE_EPOCHS': '100'}},
    ],
    '11': [ # PC 11 — PRODUÇÃO: Top 2 Vencedores (para deployment, dataset completo, 200 épocas)
        # T1: VAE Champion (β=0.05, 200 epochs) — BEST FID 140.22 @ 150ep, esperado melhor @ 200ep
        {'id': 'vae_prod_champion_beta005', 'target': 'VAE', 'env': {'RUN_PROFILE':'PROD', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '200'}},
        
        # T2: VAE Runner-Up (β=0.1, 200 epochs) — Runner-up FID 141.28 @ 150ep, esperado melhor @ 200ep
        {'id': 'vae_prod_runnerup_beta01', 'target': 'VAE', 'env': {'RUN_PROFILE':'PROD', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '200'}},
    ]
    ,'12': [ # PC 12 — PRODUÇÃO: Teoricamente calibrado para 50k dataset (β=0.01, 200 epochs)
        # T1: VAE teórico (β=0.01, 200 epochs) — β ajustado para dataset grande conforme literatura (Higgins et al., β=0.005-0.05 para 50k samples)
        {'id': 'vae_prod_beta001_200ep', 'target': 'VAE', 'env': {'RUN_PROFILE':'PROD', 'VAE_BETA': '0.01', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '200'}},
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
        'VAE_LATENT_DIM', 'VAE_BETA', 'VAE_LR', 'VAE_EPOCHS', 'VAE_COSINE_LR', 'VAE_KL_ANNEALING_EPOCHS', 'VAE_PERCEPTUAL_LOSS',
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
    parser.add_argument('--pc', type=str, required=True, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], help="ID do Computador (1-12, 11-12=PROD)")
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
        elif target == 'CVAE':
            run_script(root_dir / 'scripts' / '03_cvae.py', exp_env)
        elif target == 'VQ_VAE':
            run_script(root_dir / 'scripts' / '04_vq_vae.py', exp_env)
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
