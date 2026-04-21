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
