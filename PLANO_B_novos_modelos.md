# Novos Modelos — WGAN-GP e LDM

> Todo o código deve ser baseado nos notebooks das aulas (notebooks 4 e 5).
> Arquitecturas, loops de treino e helpers devem ser copiados/adaptados
> a partir desses notebooks — não inventados do zero.

---

## Princípio de testes para cada novo modelo

Seguir sempre esta ordem antes de explorar novos hiperparâmetros:

1. **Baseline comparável** — mesmos parâmetros do modelo equivalente já testado.
   Isola o efeito da mudança de arquitectura/loss. Sem este passo não há
   interpretação possível.

2. **Melhor combo conhecido** — usar a melhor combinação encontrada nos testes
   anteriores da mesma categoria. Responde a: "o novo modelo consegue superar
   o que já temos nas melhores condições?"

3. **Exploração própria** — hiperparâmetros específicos do novo modelo.
   Só faz sentido depois de ter 1 e 2 como referência.

---

## B1 · WGAN-GP — `05_wgan.py`

**Base de código:** Notebook 4 + `02_dcgan.py` existente.

**Porquê:** O DCGAN usa BCE loss, que força o discriminador a outputs [0,1]
e causa colapso de modo (ver `dcgan_beta09`, FID=286). O WGAN-GP substitui
por distância de Wasserstein — o treino é mais estável, sem colapso, e a
loss tem significado geométrico directo.

**Arquitectura:**
- `DCGenerator` — cópia directa do notebook 4 / `02_dcgan.py`
- `DCCritic` — igual ao `DCDiscriminator` do notebook 4, mas **sem `nn.Sigmoid()`** no final
- Loop de treino: critic actualizado N vezes por cada update do gerador
- Loss substituída por Wasserstein + gradient penalty

**Ficheiros a criar/modificar:**
- CRIAR `scripts/05_wgan.py`
- MODIFICAR `scripts/04_evaluation.py` — suporte a `EVAL_TARGET='WGAN'`
- MODIFICAR `scripts/config.py` — adicionar `wgan_epochs`
- MODIFICAR `scripts/run_experiments.py` — adicionar PC5

**Testes PC5 (por ordem):**

```python
'5': [
    # 1. Baseline — mesmos params do default_dcgan
    {'id': 'wgan_default',      'target': 'WGAN', 'env': {'RUN_PROFILE':'DEV'}},

    # 2. Melhor combo do DCGAN (lat32+ngf128 foi o melhor do PC2+PC4)
    {'id': 'wgan_lat32_ngf128', 'target': 'WGAN', 'env': {'RUN_PROFILE':'DEV',
                                 'WGAN_LATENT':'32', 'WGAN_NGF':'128', 'WGAN_NDF':'128'}},

    # 3. Exploração WGAN-específica
    {'id': 'wgan_lambda5',   'target': 'WGAN', 'env': {'RUN_PROFILE':'DEV', 'WGAN_LAMBDA_GP':'5'}},
    {'id': 'wgan_lambda20',  'target': 'WGAN', 'env': {'RUN_PROFILE':'DEV', 'WGAN_LAMBDA_GP':'20'}},
    {'id': 'wgan_ncritic2',  'target': 'WGAN', 'env': {'RUN_PROFILE':'DEV', 'WGAN_N_CRITIC':'2'}},
]
```

**Comparações-chave:**

| Pergunta | Comparação |
|---|---|
| WGAN vs DCGAN (mesma arquitectura) | `wgan_default` vs `default_dcgan` |
| Melhor WGAN vs melhor DCGAN | `wgan_lat32_ngf128` vs `dcgan_lat32_ngf128` (PC4) |
| Qual λ_GP é melhor | `wgan_lambda5` vs `wgan_default` vs `wgan_lambda20` |

---

## B2 · LDM — `06_ldm.py`

**Base de código:** Notebook 5 (secção 2 — Latent Diffusion) + código
esboçado nos comentários do `03_diffusion.py` (linhas 394–427).

**Porquê:** O LDM comprime imagens com um VAE e treina difusão no espaço
latente — é o paradigma da Stable Diffusion. O denoiser opera num espaço
muito menor (ex: 8×4×4 vs 3×32×32) e por isso aprende mais depressa, usa
menos memória, e tipicamente produz o melhor FID de todos os modelos.

**Arquitectura (duas fases):**
- **Fase 1** — `SpatialVAE`: encoder que produz latentes espaciais 2D
  `[B, C_lat, 4, 4]` (baseado no VAE do notebook 5, adaptado para RGB 32×32).
  Diferente do `ConvVAE` do `01_vae.py` que produz vectores 1D.
- **Fase 2** — `LatentDenoiseNetwork`: cópia directa do notebook 5, opera
  sobre os latentes 2D com o VAE frozen.

**Porquê não reutilizar o `ConvVAE` do `01_vae.py`:** O `ConvVAE` produz
latentes 1D `[B, 128]`. O `LatentDenoiseNetwork` do notebook 5 espera
latentes 2D `[B, C, H, W]`. A solução limpa é o `SpatialVAE` com heads
convolucionais (como no notebook 5), sem precisar de reshapes artificiais.

**Ficheiros a criar/modificar:**
- CRIAR `scripts/06_ldm.py`
- MODIFICAR `scripts/04_evaluation.py` — suporte a `EVAL_TARGET='LDM'`
- MODIFICAR `scripts/config.py` — adicionar `ldm_vae_epochs`, `ldm_epochs`
- MODIFICAR `scripts/run_experiments.py` — adicionar PC6

**Testes PC6 (por ordem):**

```python
'6': [
    # 1. Baseline
    {'id': 'ldm_default',    'target': 'LDM', 'env': {'RUN_PROFILE':'DEV'}},

    # 2. Transferir melhor T_steps do PC3 (preencher com resultado real do PC3)
    {'id': 'ldm_best_T',     'target': 'LDM', 'env': {'RUN_PROFILE':'DEV',
                              'LDM_T_STEPS': '<melhor T do PC3>'}},
    # 2b. Cosine schedule (se diff_cosine mostrou melhoria no PC3)
    {'id': 'ldm_cosine',     'target': 'LDM', 'env': {'RUN_PROFILE':'DEV', 'LDM_SCHEDULE':'cosine'}},

    # 3. Exploração LDM-específica (dimensão do espaço latente)
    {'id': 'ldm_ch4',        'target': 'LDM', 'env': {'RUN_PROFILE':'DEV', 'LDM_LAT_CH':'4'}},
    {'id': 'ldm_ch16',       'target': 'LDM', 'env': {'RUN_PROFILE':'DEV', 'LDM_LAT_CH':'16'}},
]
```

**Comparações-chave:**

| Pergunta | Comparação |
|---|---|
| LDM vs Diffusion pixel (mesma config) | `ldm_default` vs `default_diff` |
| Cosine ajuda no espaço latente? | `ldm_cosine` vs `ldm_default` |
| Compressão ideal | `ldm_ch4` vs `ldm_default` vs `ldm_ch16` |

---

## Ordem global de execução

```
1. PC1 + vae_beta1     → zero risco, corre já
2. PC3 + diff_cosine   → pequena mod ao 03_diffusion.py, corre já
                         (resultados alimentam o LDM — saber o melhor T_steps e schedule)
3. PC5 (WGAN)          → independente, pode correr em paralelo com PC3
4. PC6 (LDM)           → depende dos resultados do PC3 para preencher ldm_best_T
```

Nunca correr a Fase 3 sem ter a Fase 1 feita — sem baseline não há interpretação.
