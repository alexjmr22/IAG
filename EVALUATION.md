**Avaliação — Uso e orquestração**

Resumo
- Use `scripts/04_evaluation.py` para calcular FID e KID para uma experiência (`EXP_NAME`) e alvo (`EVAL_TARGET`).
- Use `scripts/run_all_evaluations.py` para varrer `results/` e rodar avaliações automaticamente em várias pastas.

scripts/run_all_evaluations.py
- O que faz: percorre `results/` e para cada subpasta que contenha checkpoints executa `scripts/04_evaluation.py` com `EXP_NAME` apontando para essa subpasta.
- Opções principais:
  - `--target`: `ALL` (padrão), `VAE`, `DCGAN`, `Diffusion` — escolhe quais modelos avaliar.
  - `--force`: força reavaliação mesmo se `results.csv` já existir.
  - `--results-dir`: diretoria raiz dos resultados (padrão `results`).

Exemplos:
```bash
# avaliar só VAEs, pulando já avaliados
python3 scripts/run_all_evaluations.py --target VAE

# avaliar todos os modelos em todas as pastas (gera/atualiza results.csv para cada pasta)
python3 scripts/run_all_evaluations.py --target ALL

# forçar override dos resultados já existentes
python3 scripts/run_all_evaluations.py --target ALL --force
```

Como funciona internamente
- Para cada pasta `results/<exp>/`:
  - Se não existirem checkpoints relevantes, a pasta é ignorada.
  - Se `--target ALL`, o script chama `04_evaluation.py` com `EXP_NAME=<exp>` e `EVAL_TARGET=ALL`.
  - Se `--target` for um modelo específico, o script chama `04_evaluation.py` com `EXP_NAME=<exp>` e `EVAL_TARGET=<target>` para esse modelo (ou salta se o checkpoint estiver ausente).
  - Se `results/<exp>/results.csv` existir, por padrão é pulado (não há override). Use `--force` para reavaliar e sobrescrever.

Notas sobre precisão e memória
- As métricas FID/KID dependem de estatísticas sobre as ativações — o processo de avaliação foi escrito para processar as atualizações das métricas em batches pequenos para evitar picos de memória. Isso não altera a exatidão do resultado (apenas reduz uso de memória).
- Em macOS com MPS a execução pode mostrar avisos de `resource_tracker` ou ser terminada por OOM; se encontrar problemas:
  - Rode o script com menos sementes/amostras (edite `config.py`) para debug.
  - Use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` ao executar para reduzir pressão do sistema.

Saída
- Cada pasta avaliada receberá (ou terá atualizada) a `results.csv` com as linhas por modelo: `model,fid_mean,fid_std,kid_mean,kid_std`.
- Um sumário agregado é gerado em `results/evaluation/summary_{EVAL_TARGET}.csv` quando o modo de varrimento é usado com `EXP_NAME=ALL`.

Problemas comuns
- "Killed" durante cálculo das métricas: reduzir tamanho de batch das atualizações das métricas (variável `METRIC_BATCH` em `scripts/04_evaluation.py`) ou forçar execução em CPU.
- Leaked semaphores/avisos de multiprocessing: diminuir `num_workers` para 0/1 em ambientes macOS instáveis.

Se quiser, adiciono um modo `--dry-run` que apenas imprime o que seria feito sem executar nada.

---

## Resultados dos Treinos — Diffusion Model

### Teste 3 — Sweep inicial (pasta `DIFF/`)

Parâmetros base: T=1000, ch=64, lr=2e-4, 50 épocas, DEV.

| Experiência        | Parâmetro variado         | FID ↓      | FID std  | KID ↓      | KID std    |
|--------------------|---------------------------|------------|----------|------------|------------|
| `default_diff`     | base (ch=64, T=1000)      | 106.89     | 2.27     | 0.0667     | 0.0061     |
| `diff_T100`        | T=100                     | 156.09     | 0.42     | 0.1495     | 0.0108     |
| `diff_T250`        | T=250                     | 128.98     | 1.28     | 0.1064     | 0.0097     |
| `diff_T500`        | T=500                     | 113.76     | 1.57     | 0.0930     | 0.0087     |
| `diff_ch32`        | channels=32               | 203.46     | 1.67     | 0.2099     | 0.0148     |
| `diff_ch64`        | channels=64               | 108.75     | 2.18     | 0.0686     | 0.0061     |
| `diff_ch128`       | channels=128              | 187.81     | 1.62     | 0.1700     | 0.0131     |
| `diff_lr1e3`       | lr=1e-3                   | 420.12     | 3.35     | 0.4586     | 0.0443     |
| `diff_lr2e5`       | lr=2e-5                   | 292.23     | 1.63     | 0.3209     | 0.0153     |

**Conclusões Teste 3:** T mais alto melhora consistentemente. ch=64 bate ch=128 (overfitting) e ch=32 (subparam.). lr=2e-4 é claramente o melhor — lr=1e-3 diverge, lr=2e-5 não converge.

---

### Teste 4 — Exploração dirigida (pasta `DIFF_ate_65fid/`)

Parâmetros base: T=1000, ch=64, lr=2e-4, 50 épocas, DEV. Variações sobre os melhores do Teste 3.

| Experiência             | Parâmetros variados                              | FID ↓      | FID std  | KID ↓      | KID std    |
|-------------------------|--------------------------------------------------|------------|----------|------------|------------|
| `diff_best_combo`       | T=1000, ch=64, lr=2e-4                           | 107.19     | 2.20     | 0.0670     | 0.0061     |
| `diff_T1500`            | T=1500                                           | 314.81     | 0.67     | 0.2930     | 0.0176     |
| `diff_T2000`            | T=2000                                           | 293.86     | 1.55     | 0.2748     | 0.0157     |
| `diff_T2000_ch64`       | T=2000, ch=64, lr=2e-4                           | 288.19     | 1.41     | 0.2674     | 0.0148     |
| `diff_ch96`             | channels=96                                      | **100.21** | 2.04     | 0.0734     | 0.0076     |
| `diff_lr5e5`            | lr=5e-5                                          | 218.35     | 2.46     | 0.2107     | 0.0117     |
| `diff_beta_high`        | beta_end=0.04                                    | 242.14     | 0.26     | 0.2711     | 0.0157     |
| `diff_beta_low`         | beta_end=0.01                                    | 467.27     | 3.91     | 0.5412     | 0.0472     |
| `diff_beta_low_combo`   | T=1000, ch=64, lr=2e-4, beta_end=0.01            | 469.09     | 3.75     | 0.5445     | 0.0472     |
| `diff_combo_v2`         | T=2000, ch=64, lr=2e-4, beta_end=0.01            | 579.23     | 2.45     | 0.6979     | 0.0454     |

**Conclusões Teste 4:** T>1000 piora (regime de sobre-difusão). ch=96 é o melhor individual (FID 100.21). beta_low colapsa completamente — beta_end=0.02 (default) é ótimo. ch=96 torna-se o novo anchor.

---

### Teste 6 — Mais épocas + ch=112 (pasta `DIFF_ate_65fid/`)

Parâmetros base: T=1000, lr=2e-4, DEV; top configs do Teste 4 com 100 épocas.

| Experiência              | Parâmetros variados                   | FID ↓      | FID std  | KID ↓      | KID std    |
|--------------------------|---------------------------------------|------------|----------|------------|------------|
| `diff_ch96_e100`         | ch=96, T=1000, lr=2e-4, 100ep         | 193.54     | 2.53     | 0.1539     | 0.0141     |
| `diff_best_combo_e100`   | ch=64, T=1000, lr=2e-4, 100ep         | 155.25     | 2.16     | 0.1407     | 0.0123     |
| `diff_ch112`             | ch=112, T=1000, lr=2e-4, 50ep         | 264.89     | 3.55     | 0.2364     | 0.0250     |

**Conclusões Teste 6:** 100 épocas sem scheduler piora (overfitting DEV). ch=112 é pior que ch=96. Scheduler é necessário para treinos longos.

---

### Teste 7 — Cosine LR Scheduler (pasta `DIFF_ate_65fid/`)

Parâmetros base: T=1000, 100 épocas, warmup 5ep, DEV. Adiciona cosine LR decay.

| Experiência                | Parâmetros variados                              | FID ↓      | FID std  | KID ↓      | KID std    |
|----------------------------|--------------------------------------------------|------------|----------|------------|------------|
| `diff_ch96_cosine`         | ch=96, lr=2e-4, cosine, 100ep                    | 78.28      | 0.76     | 0.0492     | 0.0062     |
| `diff_ch64_cosine`         | ch=64, lr=2e-4, cosine, 100ep                    | 107.32     | 2.33     | 0.0828     | 0.0086     |
| `diff_ch96_cosine_lr4e4`   | ch=96, lr=4e-4, cosine, 100ep                    | **65.73**  | 0.52     | **0.0353** | 0.0052     |

**Conclusões Teste 7:** Cosine scheduler com warmup melhora significativamente. ch=96 + lr=4e-4 + cosine é o melhor DEV (FID 65.73, KID 0.0353). LR mais alto (4e-4) beneficia do decaimento suave do cosine.

---

### Teste 8 — PROD com DDIM (pastas `results/` e `diff_prod_ddim_e100/`)

Config: ch=96, T=1000, lr=4e-4, cosine (warmup 5ep/10ep), DDIM 100 steps, dataset completo (PROD).

| Experiência             | Parâmetros variados                                | FID ↓      | FID std  | KID ↓      | KID std    |
|-------------------------|----------------------------------------------------|------------|----------|------------|------------|
| `diff_prod_ddim_e100`   | PROD, ch=96, lr=4e-4, cosine, 100ep, DDIM-100      | **32.17**  | 0.41     | **0.0169** | 0.0037     |
| `diff_prod_ddim_e250`   | PROD, ch=96, lr=4e-4, cosine, 250ep, DDIM-100      | —          | —        | —          | —          |

> `diff_prod_ddim_e250`: sem resultados disponíveis (treino não concluído ou ficheiro ausente).

**Conclusões Teste 8:** Dataset completo (PROD) melhora dramaticamente — FID 32.17 vs 65.73 DEV. DDIM acelera sampling sem perda de qualidade.

---

### Teste 9 — EMA (pasta `results/`)

Config: ch=96, T=1000, lr=4e-4, cosine, DDIM-100, EMA decay=0.9999, PROD.

| Experiência         | Parâmetros variados                                        | FID ↓      | FID std  | KID ↓      | KID std    |
|---------------------|------------------------------------------------------------|------------|----------|------------|------------|
| `diff_ema_e100`     | PROD, ch=96, lr=4e-4, cosine, 100ep, EMA=0.9999           | 48.72      | 0.71     | 0.0359     | 0.0061     |
| `diff_ema_e200`     | PROD, ch=96, lr=4e-4, cosine, 200ep, EMA=0.9999           | **28.97**  | 0.38     | **0.0144** | 0.0035     |

**Conclusões Teste 9:** EMA com 200 épocas atinge o melhor FID global (28.97). EMA a 100ep (48.72) fica abaixo do DDIM sem EMA (32.17) — EMA precisa de mais épocas para estabilizar. Com 200ep, EMA supera o PROD sem EMA.

---

## Ranking Global — Diffusion

| Rank | Experiência               | Config                                            | FID ↓      | KID ↓      | Modo  |
|------|---------------------------|---------------------------------------------------|------------|------------|-------|
| 🥇 1 | `diff_ema_e200`           | ch=96, cosine lr=4e-4, 200ep, EMA, DDIM, PROD     | **28.97**  | **0.0144** | PROD  |
| 🥈 2 | `diff_prod_ddim_e100`     | ch=96, cosine lr=4e-4, 100ep, DDIM, PROD          | 32.17      | 0.0169     | PROD  |
| 🥉 3 | `diff_ema_e100`           | ch=96, cosine lr=4e-4, 100ep, EMA, DDIM, PROD     | 48.72      | 0.0359     | PROD  |
| 4    | `diff_ch96_cosine_lr4e4`  | ch=96, cosine lr=4e-4, 100ep                      | 65.73      | 0.0353     | DEV   |
| 5    | `diff_ch96_cosine`        | ch=96, cosine lr=2e-4, 100ep                      | 78.28      | 0.0492     | DEV   |
| 6    | `diff_ch96`               | ch=96, T=1000, lr=2e-4, 50ep                      | 100.21     | 0.0734     | DEV   |
| 7    | `default_diff`            | ch=64, T=1000, lr=2e-4, 50ep                      | 106.89     | 0.0667     | DEV   |
