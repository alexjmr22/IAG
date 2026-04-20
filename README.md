# Project 1 — ArtBench Generative Modeling

**Cadeira**: Generative AI — FCTUC 2025/2026  
**Prazo**: 19 de Abril de 2026 (via Inforestudante)  
**Relatório**: formato Springer LNCS, máx. 12 páginas (sem referências)

---

## O que é pedido (resumo do enunciado)

O objetivo é **aprender modelos generativos** a partir de obras de arte e gerar novas imagens plausíveis. O dataset usado é o **ArtBench-10**: 50 000 imagens de treino, resolução 32×32 RGB, 10 estilos artísticos.

### Modelos obrigatórios

| Família       | Mínimo exigido     | Extra (bonus)            |
|---------------|--------------------|--------------------------|
| Autoencoder   | VAE                | β-VAE, CVAE, VQ-VAE      |
| GAN           | DCGAN              | WGAN-GP, cDCGAN, StyleGAN|
| Diffusion     | DDPM               | DDIM, CFG, LDM           |

### Protocolo de avaliação obrigatório

1. Gerar **5 000 amostras** por modelo
2. Amostrar **5 000 imagens reais** do ArtBench
3. Calcular **FID** (5k vs 5k imagens completas)
4. Calcular **KID** em **50 subsets de tamanho 100** → reportar média ± desvio-padrão
5. Repetir tudo com **10 seeds diferentes**
6. Reportar **FID: média ± std** e **KID: média ± std** por modelo

> Pré-processamento, contagem de amostras e código de avaliação devem ser **idênticos** entre modelos.

---

## Estrutura do repositório

```
IAG/
├── notebooks/                   ← trabalho principal (notebooks por ordem)
│   ├── 00_setup_data.ipynb      ← carregamento e exploração do dataset
│   ├── 01_vae.ipynb             ← treino e amostras do VAE
│   ├── 02_dcgan.ipynb           ← treino e amostras do DCGAN
│   ├── 03_diffusion.ipynb       ← treino e amostras do DDPM
│   └── 04_evaluation.ipynb      ← protocolo FID/KID completo
│
├── utils/
│   └── artbench_dataset.py      ← loader HuggingFace / Kaggle (fornecido)
│
├── data/                        ← dataset local (gitignored)
│   └── ArtBench-10/             ←   descarregar do Kaggle se necessário
│
├── results/                     ← checkpoints e amostras geradas (gitignored)
│   ├── vae/
│   ├── dcgan/
│   ├── diffusion/
│   └── evaluation/
│
└── TP/                          ← material fornecido pelos docentes
    ├── _GENAI__TP1_Enunciado_2026_v2.pdf
    └── TP1-alunos-src-only/
        ├── requirements.txt
        ├── scripts/artbench_local_dataset.py
        └── student_start_pack/
            ├── ArtBench10_Student_Start_Pack.ipynb
            └── training_20_percent.csv   ← subset 20% para desenvolvimento
```

---

## Passo a passo

### 1. Instalar dependências

```bash
pip install -r TP/TP1-alunos-src-only/requirements.txt
```

### 2. Explorar o dataset — `00_setup_data.ipynb`

- Carrega o ArtBench-10 via HuggingFace (`P1B3/artbench-10`) ou Kaggle local
- Aplica o **subset de 20%** para desenvolvimento rápido (`USE_SUBSET = True`)
- Visualiza amostras por estilo e distribuição de classes
- Cria o `DataLoader` PyTorch pronto a usar nos notebooks seguintes

> Para usar o dataset completo basta pôr `USE_SUBSET = False` em qualquer notebook.

### 3. Treinar o VAE — `01_vae.ipynb`

- Arquitetura Encoder (CNN → μ, log σ²) + Decoder (ConvTranspose2d)
- Trick de reparametrização: z = μ + ε·σ
- Loss = reconstrução (MSE) + β·KL divergência
- Guarda checkpoint em `results/vae/vae_checkpoint.pth`
- **Começar com subset 20%**, depois treinar no dataset completo para avaliação final

### 4. Treinar o DCGAN — `02_dcgan.ipynb`

- Generator: z (100×1×1) → imagem (3×32×32) com ConvTranspose2d + BN + ReLU
- Discriminator: imagem → score real/fake com Conv2d + BN + LeakyReLU
- Loss adversarial BCE; inicialização de pesos conforme o paper DCGAN
- Guarda checkpoint em `results/dcgan/dcgan_checkpoint.pth`

### 5. Treinar o DDPM — `03_diffusion.ipynb`

- Schedule linear de betas (β₁=1e-4 → β_T=0.02, T=1000)
- Forward process: q(x_t | x_0) = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
- U-Net com sinusoidal time embeddings para prever ε
- Sampling DDPM reverso passo a passo
- Guarda checkpoint em `results/diffusion/diffusion_checkpoint.pth`

### 6. Avaliação completa — `04_evaluation.ipynb`

- Carrega os 3 checkpoints
- Executa o protocolo obrigatório (10 seeds × FID + KID)
- Gera tabela e gráfico de comparação
- Guarda resultados em `results/evaluation/results.csv`

---

## Dicas importantes

- **Subset primeiro**: começa sempre com `USE_SUBSET = True` para iterar rápido. Só treina no dataset completo quando a configuração estiver definida.
- **Seeds fixas**: usa seeds explícitas em todos os experimentos para garantir reprodutibilidade.
- **Checkpoints frequentes**: guarda o modelo periodicamente para não perder progresso.
- **Extensões valem bónus**: CVAE, WGAN-GP, DDIM, geração condicional por estilo, etc.

---

## Referências principais

- Kingma & Welling (2013) — VAE: [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Radford et al. (2015) — DCGAN: [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Ho et al. (2020) — DDPM: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Liao et al. (2022) — ArtBench: [arXiv:2206.11404](https://arxiv.org/abs/2206.11404)
- Dataset Kaggle: [alexanderliao/artbench10](https://www.kaggle.com/datasets/alexanderliao/artbench10)
