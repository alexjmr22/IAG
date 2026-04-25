# Exponential Moving Average (EMA) no Diffusion Model

## O que é EMA

Durante o treino de uma rede neuronal, os pesos oscilam a cada batch — o optimizer dá passos que às vezes sobem, às vezes descem, à medida que tenta minimizar a loss. O EMA mantém uma segunda cópia dos pesos que é uma **média suavizada** de todos os valores que os pesos foram tendo ao longo do treino.

A fórmula é simples:

```
shadow_t = decay × shadow_{t-1} + (1 - decay) × params_t
```

Com `decay = 0.9999`, o modelo EMA dá 99.99% de peso ao histórico passado e apenas 0.01% ao passo atual. O resultado é uma versão dos pesos que ignora as oscilações de curto prazo e representa o "centro de gravidade" do treino.

---

## Diferença entre `03_diffusion.py` e `03b_diffusion_ema.py`

| | `03_diffusion.py` | `03b_diffusion_ema.py` |
|---|---|---|
| Pesos usados no sampling | Pesos diretos do optimizer | Pesos EMA (shadow weights) |
| Checkpoint guardado | `diffusion_checkpoint.pth` | `diffusion_checkpoint.pth` + `diffusion_ema_checkpoint.pth` |
| Classe extra | — | `EMAModel` |
| Custo computacional | base | ~+1% (uma cópia extra em memória + update por step) |
| Ganho esperado em FID | base | −1 a −3 pontos |

O treino em si é **idêntico** — mesma loss, mesmo optimizer, mesma arquitetura UNet. A única diferença é que, após cada `opt.step()`, o EMA é atualizado:

```python
# 03_diffusion.py
opt.zero_grad(); loss.backward(); opt.step()

# 03b_diffusion_ema.py
opt.zero_grad(); loss.backward(); opt.step()
ema.update(model)  # linha extra
```

E no sampling, os pesos EMA são aplicados temporariamente:

```python
ema.apply(model)                          # substitui pesos pelo EMA
samples = _sample(model, shape=(...))     # gera imagens
ema.restore(model)                        # repõe pesos do optimizer
```

---

## Como funciona a classe `EMAModel`

```python
ema = EMAModel(model, decay=0.9999)
```

Ao criar, copia todos os pesos do modelo para `self.shadow` em float32.

### `update(model)` — chamado após cada optimizer step

```python
for k, v in model.state_dict().items():
    self.shadow[k] = decay * self.shadow[k] + (1 - decay) * v.float()
```

Actualiza cada parâmetro independentemente. Não interfere com os gradientes nem com o optimizer.

### `apply(model)` + `restore(model)` — usado no sampling e nas amostras intermédias

```python
ema.apply(model)    # guarda backup dos pesos reais, carrega shadow
# ... sampling ...
ema.restore(model)  # repõe os pesos reais para continuar o treino
```

O modelo nunca "sabe" que está a usar pesos EMA — do ponto de vista do forward pass é transparente.

### `state_dict()` / `load_state_dict()` — persistência

Os shadow weights são guardados separadamente do modelo:

```
results/diff_ema_e100/
├── diffusion_checkpoint.pth      ← pesos do optimizer (útil para retomar treino)
└── diffusion_ema_checkpoint.pth  ← pesos EMA (usado na avaliação final)
```

---

## Porquê o EMA melhora o FID

O FID mede a distância entre a distribuição das imagens geradas e a distribuição das imagens reais. Os pesos do optimizer no final do treino estão num ponto "ruidoso" — o último passo de gradiente pode ter sido uma má direção.

Os pesos EMA estão numa média de milhares de passos, o que equivale a um ponto mais central e estável no espaço de parâmetros. Empiricamente, modelos de difusão com EMA geram imagens mais consistentes e com menos artefactos, porque o sampler executa 100-1000 passos de denoising — qualquer instabilidade nos pesos acumula-se ao longo da cadeia.

É uma técnica padrão em DDPM, Stable Diffusion, e praticamente todos os modelos de difusão modernos.

---

## Como correr

```bash
# PC 9: diff_ema_e100 (100 épocas) + diff_ema_e200 (200 épocas)
python scripts/run_experiments.py --pc 9

# Ou diretamente com variáveis de ambiente custom:
EXP_NAME=diff_ema_teste RUN_PROFILE=PROD DIFF_CHANNELS=96 DIFF_EMA_DECAY=0.9999 \
    python scripts/03b_diffusion_ema.py
```

O `04_evaluation.py` detecta automaticamente `EVAL_TARGET=DiffusionEMA` e carrega o `diffusion_ema_checkpoint.pth` em vez do raw.
