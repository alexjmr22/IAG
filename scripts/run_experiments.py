import os
import subprocess
import sys
import argparse
from pathlib import Path

# --- THE MIGHTY GRID SEARCH ORCHESTRATOR --- #

EXPERIMENTS = {
    '1': [ # PC 1 — VAE/CVAE/VQ-VAE COMPLETE Battery (Consolidated from PC 1-11 of VAE branch)
        # --- Basic Sweeps (Latent / Beta / LR) ---
        {'id': 'default_vae',   'target': 'VAE', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'vae_lat16',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '16'}},
        {'id': 'vae_lat32',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '32'}},
        {'id': 'vae_lat64',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '64'}},
        {'id': 'vae_lat96',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '96'}},
        {'id': 'vae_lat128',    'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '128'}},
        {'id': 'vae_lat256',    'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '256'}},
        
        {'id': 'vae_beta0',      'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.0'}},
        {'id': 'vae_beta002',    'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.02'}},
        {'id': 'vae_beta005',    'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05'}},
        {'id': 'vae_beta01',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1'}},
        {'id': 'vae_beta015',    'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.15'}},
        {'id': 'vae_beta02',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.2'}},
        {'id': 'vae_beta05',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.5'}},
        {'id': 'vae_beta1',      'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '1.0'}},
        {'id': 'vae_beta2',      'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '2.0'}},

        {'id': 'vae_lr1e2',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-2'}},
        {'id': 'vae_lr5e3',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '5e-3'}},
        {'id': 'vae_lr2e3',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '2e-3'}},
        {'id': 'vae_lr1e3',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-3'}},
        {'id': 'vae_lr5e4',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '5e-4'}},
        {'id': 'vae_lr1e4',     'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LR': '1e-4'}},

        # --- Directed Combos & Advanced Techniques (Ronda 3-5) ---
        {'id': 'vae_best_combo', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3'}},
        {'id': 'vae_combo_full', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3'}},
        {'id': 'vae_r3_beta01_lat128_lr2e3_e50', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50'}},
        
        {'id': 'vae_r5_t2_cosine',        'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_COSINE_LR': 'true'}},
        {'id': 'vae_r5_t3_kl_annealing',  'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        {'id': 'vae_r5_t4_both',          'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        {'id': 'vae_r5_final_t5_perceptual_loss', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_PERCEPTUAL_LOSS': 'true'}},
        
        {'id': 'cvae_r5_t5_conditional',  'target': 'CVAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true', 'VAE_KL_ANNEALING_EPOCHS': '10'}},
        {'id': 'vq_vae_r5_t6_quantized',  'target': 'VQ_VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_LATENT_DIM': '128', 'VAE_LR': '2e-3', 'VAE_EPOCHS': '50', 'VAE_COSINE_LR': 'true'}},

        # --- Final Production Runs ---
        {'id': 'vae_prod_champion_beta005', 'target': 'VAE', 'env': {'RUN_PROFILE':'PROD', 'VAE_BETA': '0.05', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '200'}},
        {'id': 'vae_prod_runnerup_beta01',  'target': 'VAE', 'env': {'RUN_PROFILE':'PROD', 'VAE_BETA': '0.1', 'VAE_LATENT_DIM': '128', 'VAE_LR': '0.002', 'VAE_EPOCHS': '200'}},
    ],
    '2': [ # PC 2 — GAN COMPLETE Battery (Consolidated from multiple PCs)
        # --- Basic Sweeps (Latent / Architecture) ---
        {'id': 'default_dcgan',    'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV'}},
        {'id': 'dcgan_lat32',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32'}},
        {'id': 'dcgan_lat64',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '64'}},
        {'id': 'dcgan_lat100',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '100'}},
        {'id': 'dcgan_ngf64',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '64',  'DCGAN_NDF': '64'}},
        {'id': 'dcgan_ngf128',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128'}},
        {'id': 'dcgan_beta09',  'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_BETA1': '0.9'}},
        {'id': 'dcgan_lr2e4',   'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LR': '2e-4'}},

        # --- Asymmetrical & Hyperparameter Combinations ---
        {'id': 'dcgan_lat32_ngf128',     'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128'}},
        {'id': 'dcgan_asym_lr',          'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},
        {'id': 'dcgan_lat32_asym_lr',    'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},
        {'id': 'dcgan_ngf128_ndf64',     'target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '64'}},
        {'id': 'dcgan_lat32_ngf128_asym','target': 'DCGAN', 'env': {'RUN_PROFILE':'DEV', 'DCGAN_LATENT': '32', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_LR_G': '1e-4', 'DCGAN_LR_D': '4e-4'}},

        # --- Advanced GANs: Spectral Norm, WGAN-GP, cDCGAN ---
        {'id': 'dcgan_cosine',     'target': 'DCGAN',  'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_COSINE': '1'}},
        {'id': 'dcgan_spectral',   'target': 'DCGAN',  'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_SPECTRAL': '1'}},
        {'id': 'dcgan_cosine_sn',  'target': 'DCGAN',  'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'DCGAN_COSINE': '1', 'DCGAN_SPECTRAL': '1'}},
        {'id': 'wgan_gp',          'target': 'WGAN',   'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},
        {'id': 'cdcgan',           'target': 'cDCGAN', 'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100'}},

        # --- StyleGAN Explorations ---
        {'id': 'stylegan_ngf128_200ep', 'target': 'StyleGAN', 'env': {'RUN_PROFILE': 'PROD', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '200'}},
        {'id': 'stylegan_map8', 'target': 'StyleGAN', 'env': {'RUN_PROFILE': 'DEV', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_EPOCHS': '100', 'STYLEGAN_MAP_LAYERS': '8'}},

        # --- Final Production Run ---
        {'id': 'dcgan_spectral_200ep', 'target': 'DCGAN', 'env': {'RUN_PROFILE':'PROD', 'DCGAN_NGF': '128', 'DCGAN_NDF': '128', 'DCGAN_LR': '2e-4', 'DCGAN_EPOCHS': '200'}},
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
        {'id': 'diff_ch96',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96'}},
        {'id': 'diff_ch128', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '128'}},
        {'id': 'diff_lr1e3', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '1e-3'}},
        {'id': 'diff_lr2e5', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '2e-5'}},
        {'id': 'diff_lr5e5', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_LR': '5e-5'}},

        # --- High-Resolution Timesteps & Beta Explorations ---
        {'id': 'diff_best_combo',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4'}},
        {'id': 'diff_T1500',         'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1500'}},
        {'id': 'diff_T2000',         'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000'}},
        {'id': 'diff_T2000_ch64',    'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4'}},
        {'id': 'diff_beta_high',     'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.04'}},
        {'id': 'diff_beta_low',      'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_BETA_END': '0.01'}},
        {'id': 'diff_beta_low_combo','target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '1000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4', 'DIFF_BETA_END': '0.01'}},
        {'id': 'diff_combo_v2',      'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_T_STEPS': '2000', 'DIFF_CHANNELS': '64', 'DIFF_LR': '2e-4', 'DIFF_BETA_END': '0.01'}},

        # --- Architecture & Scheduling (Cosine / ch=112) ---
        {'id': 'diff_ch112',         'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '112', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4'}},
        {'id': 'diff_ch96_e100',     'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96',  'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100'}},
        {'id': 'diff_best_combo_e100','target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '64',  'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100'}},
        {'id': 'diff_ch96_cosine',        'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5'}},
        {'id': 'diff_ch64_cosine',        'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '64', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '2e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5'}},
        {'id': 'diff_ch96_cosine_lr4e4',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5'}},

        # --- Production Runs (DDIM & EMA) ---
        {'id': 'diff_prod_ddim_e100',  'target': 'Diffusion', 'env': {'RUN_PROFILE':'PROD', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5', 'DIFF_SAMPLER': 'ddim', 'DIFF_DDIM_STEPS': '100'}},
        {'id': 'diff_ema_e100',        'target': 'DiffusionEMA', 'env': {'RUN_PROFILE':'PROD', 'DIFF_CHANNELS': '96', 'DIFF_T_STEPS': '1000', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '100', 'DIFF_WARMUP_EPOCHS': '5', 'DIFF_SAMPLER': 'ddim', 'DIFF_DDIM_STEPS': '100', 'DIFF_EMA_DECAY': '0.9999'}},
        {'id': 'diff_prod_ddim_e200', 'target': 'DiffusionEMA', 'env': {'RUN_PROFILE':'PROD', 'DIFF_CHANNELS': '96', 'DIFF_LR': '4e-4', 'DIFF_EPOCHS': '200', 'EMA': 'true'}}
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
        print(f" \n\n[EXPERIENCE]: {exp['id']}")
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
        elif target == 'DCGAN' or target == 'WGAN' or target == 'cDCGAN' or target == 'StyleGAN':
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
