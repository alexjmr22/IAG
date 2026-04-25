# Testes Extra — Sem novos ficheiros

Estes testes usam os scripts e modelos já existentes.
Só é necessário adicionar entradas no `run_experiments.py` (e uma pequena
modificação ao `03_diffusion.py` no caso A2).

---

## A1 · `vae_beta1` — β-VAE com β = 1.0

**Porquê:** O PC1 testa β = 0, 0.1, 0.5, 2.0 mas **falta β = 1.0**,
que é o ponto de referência teórico (ELBO standard de Kingma & Welling 2013).
Sem ele não conseguimos dizer se β=0.7 (o default actual) é melhor ou pior
do que o caso canónico da literatura.

**O que fazer:**
- Adicionar ao PC1 em `run_experiments.py`:
  ```python
  {'id': 'vae_beta1', 'target': 'VAE', 'env': {'RUN_PROFILE':'DEV', 'VAE_BETA': '1.0'}}
  ```
- Zero alterações ao `01_vae.py` — já lê `VAE_BETA` como env var.

---

## A2 · `diff_cosine` — Cosine noise schedule

**Base:** O próprio notebook 5 menciona que o schedule linear
*"can be swapped for Cosine for better efficiency"*.

**Porquê:** O schedule linear destrói a imagem demasiado rápido nos primeiros
passos e é redundante nos últimos. O cosine schedule (Nichol & Dhariwal 2021 —
"Improved DDPM") tem uma transição mais suave. Melhora o FID tipicamente 15–30%
sem alterar a arquitectura nem o número de epochs.

**O que fazer:**

1. Modificar `03_diffusion.py` (3 alterações simples):
   - Nos hiperparâmetros: `SCHEDULE = os.environ.get('DIFF_SCHEDULE', 'linear')`
   - Adicionar função antes da classe `GaussianDiffusion`:
     ```python
     def _cosine_betas(num_timesteps, s=0.008):
         steps = num_timesteps + 1
         t = torch.linspace(0, num_timesteps, steps) / num_timesteps
         ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
         ac = ac / ac[0]
         return (1 - ac[1:] / ac[:-1]).clamp(0, 0.999)
     ```
   - No `__init__` da `GaussianDiffusion`: aceitar `schedule='linear'` e usar
     `_cosine_betas()` quando `schedule == 'cosine'` em vez de `torch.linspace()`

2. Adicionar ao PC3 em `run_experiments.py`:
   ```python
   {'id': 'diff_cosine', 'target': 'Diffusion', 'env': {'RUN_PROFILE':'DEV', 'DIFF_SCHEDULE': 'cosine'}}
   ```

---

## Estratégia de comparação

| Teste | Compara com | Pergunta |
|---|---|---|
| `vae_beta1` | `default_vae` (β=0.7) e toda a curva β | β=1.0 é melhor ou pior que β=0.7? |
| `diff_cosine` | `default_diff` (linear, T=1000) | O schedule cosine melhora o FID sem mais epochs? |

Depois de ter estes resultados, o `diff_cosine` informa também o teste do LDM
(se cosine ajuda no pixel space, provavelmente ajuda também no espaço latente).
