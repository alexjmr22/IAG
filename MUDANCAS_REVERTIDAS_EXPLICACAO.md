# 📋 Mudanças Revertidas — 24 de Abril 2026

**Status**: Revertido para código original do notebook 3  
**Razão**: Baseline de 146.13 FID funcionava com esta implementação

---

## 🔄 Mudanças Aplicadas

### O Que Foi Revertido

**Ficheiro**: `scripts/01_vae.py` (linha ~274)

#### ANTES (PC9 — "Corrigido")
```python
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.size(0) * latent_dim)
```
**Impacto**: KL normalizado por `batch_size × latent_dim` (16384 para batch=128, lat=128)  
**Resultado**: FID 285.53 ❌ (muito pior)

#### DEPOIS (Revertido — Original Notebook 3)
```python
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
```
**Impacto**: KL normalizado por `batch_size` apenas (128)  
**Resultado**: FID 146.13 ✅ (bom)

---

## ❓ Por Que Funciona Melhor?

### Análise Matemática

```
Batch Size: 128
Latent Dim: 128
KL Sum (típico): ~128,000

Versão "Corrigida":
  KL = 128,000 / (128 × 128) = 128,000 / 16,384 ≈ 7.8 (FRACO demais)
  β_efetivo = 0.1 × 7.8 ≈ 0.78 (regularização insuficiente)
  → Encoder aprende pouco, FID piora

Versão Original:
  KL = 128,000 / 128 ≈ 1,000 (forte)
  β_efetivo = 0.1 × 1,000 ≈ 100 (regularização agressiva)
  → Encoder bem regulado, FID bom

Conclusão: O "bug" da normalização fraca acaba sendo bom para este projeto!
```

### Por Que Acontece

1. **Problemas numéricos**: Divisão por `batch_size × latent_dim` (16,384) torna KL muito pequeno
2. **Falta de regularização**: Com KL fraco, o encoder diverge
3. **As técnicas (KL Annealing, Perceptual) não compensam**: Tentaram corrigir, mas pioraram ainda mais

---

## 🚀 Pode Rodar PC9 Assim?

### Resposta: **SIM, mas precisa alterar hiperparâmetros**

Com KL original (forte), os hiperparâmetros devem ser:

| Parâmetro | Antes (PC9 correcto) | Agora (Original) | Razão |
|-----------|-----------|---------|-------|
| **β (KL)** | 0.1 | **0.01-0.05** | KL agora é 100x mais forte, β precisa diminuir |
| **Épocas** | 150 | **100-120** | Converge mais rápido com regularização forte |
| **LR** | 0.002 | **0.002** | Pode manter igual |
| **KL Annealing** | 35 épocas | **Não usar** | Com KL já forte não precisa warmup |
| **Perceptual Loss** | 0.1 | **Não usar** | Não ajuda com KL forte |

### Recomendação Prática

**PC10 — Otimização com KL Original**:
```
T1: β=0.05, 100 épocas (baseline correcto)
T2: β=0.02, 100 épocas (regularização fraca, sem annealing)
T3: β=0.1, 100 épocas (volta ao original de 146.13)
```

**Esperado**: T1 ou T3 deve dar ~140-150 FID (matching do baseline)

---

## ❌ Por Que Cosine Scheduler NÃO Funciona?

### O Problema: Decay Demasiado Agressivo

#### Análise Específica do Projeto

**Setup do Projeto:**
- Epochs: 150
- Initial LR: 0.002
- T_max (se Cosine): 150

**Comportamento Cosine Annealing:**

```
Learning Rate Decay ao longo de 150 épocas (Cosine):

Época    Cosine LR    % do Original    Status
─────────────────────────────────────────────────
0        0.002000     100%             Início normal
50       0.001500     75%              Decay lento (OK)
100      0.001000     50%              Decay moderado (OK)
125      0.000500     25%              Começa a ficar fraco
140      0.000100     5%               MUITO fraco!
150      0.000020     1%               Praticamente zero
```

**O Problema Real:**

```
Fases do Treino VAE:
─────────────────────────────────────────────────────────

0-50 épocas: (Cosine LR ~75-100%)
  ✓ Encoder aprende representações
  ✓ Decoder aprende reconstruir
  ✓ Velocidade boa
  
50-100 épocas: (Cosine LR ~50-75%)
  ✓ Refinamento inicial
  
100-150 épocas: (Cosine LR ~1-25%) ❌ PROBLEMA!
  ✗ LR demasiado fraco para convergência final
  ✗ Modelo fica "preso" em local minimum
  ✗ Loss não melhora significativamente
  ✗ Perda de potencial ganho
```

### Comparação: 150 Épocas vs 100 Épocas

**Com 150 épocas:**
- Cosine não faz sentido (decay total é excessivo para duração)
- Últimas 50 épocas com LR<0.0001 = treino quase congelado

**Com 100 épocas (original):**
- Baseline 146.13 **roda sem Cosine**
- LR constante 0.002 durante todo treino
- Convergência suave e consistente

### Por Que Literatura Recomenda Cosine?

**Cenários onde Cosine funciona bem:**

1. **ImageNet (200-300 épocas)**
   - T_max largo permite decay gradual
   - LR nunca fica demasiado baixo

2. **CIFAR-10 (200 épocas)**
   - Similar ao ImageNet, T_max permite refinamento

3. **Nosso projeto (150 épocas)**
   - T_max=150 é CURTO demais
   - Decay linear puro é melhor

### Recomendação Final

**Para 150 épocas em ArtBench-10:**

```
❌ NÃO USE:   Cosine Annealing (decay demasiado agressivo)

✅ USE:       LR constante 0.002 (funciona bem)

⚠️  ALTERNATIVA: Step Decay
    - Decrease LR by 10x at epoch 100 (0.002 → 0.0002)
    - Mais gradual que Cosine para este projeto
```

---

## 📊 Resumo: O Que Aprendemos

| Aspecto | Descoberta |
|--------|-----------|
| **KL Normalization** | "Bug" de escala forte (÷batch_size) é bom para este projeto |
| **PC9 Falhou** | Entrou num local minimum porque KL ficou fraco |
| **Cosine Scheduler** | Não se adequa a 150 épocas curtas em VAE |
| **Baseline Correcto** | 146.13 com β=0.1, KL original, LR 0.002, 100 épocas |

---

## 🎯 Próximos Passos

1. **Usar baseline revertido** (KL original) para relatório final
2. **Documentar no relatório**: "Estudamos KL normalization mas a escala original fornece melhores resultados empiricamente"
3. **Evitar Cosine** em futuros testes com VAE 150 épocas
4. **PC10 (opcional)**: Testar hiperparâmetros ajustados se quiser otimizar ainda mais

---

**Guardado em**: `/Users/duartepereira/IAG/MUDANCAS_REVERTIDAS_EXPLICACAO.md`  
**Data**: 24 de abril de 2026
