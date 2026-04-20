# Relatório de Projeto: Modelagem Gerativa no ArtBench-10

## 1. Introdução
Este projeto foca-se na implementação e avaliação de modelos gerativos (GAN, VAE e Diffusion) utilizando o dataset **ArtBench-10**. O objetivo é gerar imagens de arte (32x32 pixels) que capturem a essência de 10 estilos artísticos distintos (ex: Impressionismo, Renascimento, etc.).

## 2. Estrutura do Projeto
O projeto foi simplificado para uma arquitetura plana e funcional:
- `dataset.py`: Carregamento eficiente dos batches nativos (pickle) do ArtBench.
- `gan.py`: Implementação de uma **Conditional DCGAN** (GAN Convolucional Condicional).
- `vae.py`: Implementação de um **Variational Autoencoder** convolucional.
- `diffusion.py`: Versão simplificada de um modelo de Difusão (UNet).
- `evaluate.py`: Módulo de avaliação estatística rigorosa.
- `pipeline.py`: Orquestrador central que executa o treino e a avaliação.
- `utils.py`: Funções auxiliares de performance, seeds e hardware.

## 3. Detalhes de Implementação por Componente

### 3.1 Dataset e Pré-processamento (`dataset.py`)
- **Carregamento Local**: Implementámos uma classe `ArtBenchDataset` que lê diretamente os ficheiros binários (pickle) da estrutura nativa do ArtBench-10.
- **Normalização**: As imagens (32x32) são convertidas para tensores e normalizadas para o intervalo `[-1, 1]`, que é o ideal para ativações `Tanh` em modelos gerativos.
- **Paralelização**: O Dataloader utiliza `num_workers=4` no Windows com `persistent_workers`, garantindo que o GPU nunca fique à espera que o CPU carregue imagens do disco.

### 3.2 Conditional GAN (`gan.py`)
- **Arquitetura DCGAN**: Baseada em convoluções transpostas e camadas de BatchNorm para estabilidade.
- **Estrutura Condicional**: Adicionámos camadas de `nn.Embedding` para as 10 classes de estilos artísticos. O estilo é injetado no Gerador como uma dimensão extra do vetor latente e no Discriminador como um canal extra na imagem de entrada.
- **Treino**: Utiliza o otimizador Adam com hiperparâmetros sugeridos na literatura (`lr=0.0002`, `betas=(0.5, 0.999)`).

### 3.3 Variational Autoencoder (`vae.py`)
- **Compressão Espacial**: O encoder reduz a imagem de 3x32x32 para um espaço latente de baixa dimensão através de convoluções com stride.
- **Truque da Reparametrização**: Essencial para permitir o backpropagation através de variáveis estocásticas (mu e logvar).
- **Função de Perda**: Combinação da perda de reconstrução (MSE) com a divergência KL (KLD) para regularizar o espaço latente.

### 3.4 Diffusion Model (`diffusion.py`)
- **Modelo Simplificado**: Implementámos uma estrutura UNet minimalista.
- **Conceito**: Diferente das GANs, este modelo aprende a "limpar" ruído gaussiano adicionado gradualmente às imagens originais.
- **Treino**: Foca-se em minimizar o erro quadrático médio (MSE) entre o ruído real e o ruído previsto pelo modelo.

### 3.5 Avaliação e Métricas (`evaluate.py`)
- **Protocolo Estatístico**: Implementámos a automação exigida no enunciado para correr **10 avaliações independentes** com sementes diferentes (42 a 51).
- **FID (Fréchet Inception Distance)**: Lógica implementada via `torchmetrics` que compara a distribuição das 5000 imagens geradas com a distribuição de referência (conjunto de teste).
- **KID (Kernel Inception Distance)**: Implementado como uma métrica de validação adicional, sendo mais robusta para datasets de tamanho médio como o ArtBench.
- **Relatório de Resultados**: O pipeline calcula e imprime a **Média** e o **Desvio Padrão**, fornecendo um intervalo de confiança para o teu relatório.

### 3.6 Utilidades e Performance (`utils.py`)
- **Reprodutibilidade**: Centralização das funções `set_seed` para garantir que todos os experimentos são reprodutíveis.
- **Hardware Agnostic**: O código deteta automaticamente se tens um NVIDIA GPU ou CPU.
- **System Tuning**: Ativação do `cudnn.benchmark` para aceleração específica na arquitetura da RTX 3060.

## 5. Como Reproduzir os Resultados
Para treinar o modelo e gerar os dados estatísticos finais:
```powershell
python pipeline.py gan --epochs 20
```

Os resultados são salvos em:
- `samples/`: Grelhas de imagens geradas durante o treino.
- `gan_checkpoint.pth`: Pesos do modelo treinado.
- Terminal: Tabela final com os valores de FID/KID.
