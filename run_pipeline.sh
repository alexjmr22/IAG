#!/bin/bash
# Pipeline de Automacao para Treino e Avaliacao do ArtBench-10

PROFILE=${1:-PROD}

echo "🚀 A iniciar a Pipeline Automatica no perfil: [$PROFILE]"
echo "------------------------------------------------------"

export RUN_PROFILE=$PROFILE

# 1. Ativar o Virtual Environment
echo "A ativar o ambiente virtual (venv)..."
source venv/bin/activate
echo "Ambiente ativado."

# 2. Executar 01 VAE
echo "A executar script 01_vae.py ..."
python scripts/01_vae.py
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha ao treinar o modelo VAE. A interromper a pipeline."
    exit 1
fi

# 3. Executar 02 DCGAN
echo "A executar script 02_dcgan.py ..."
python scripts/02_dcgan.py
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha ao treinar o modelo DCGAN. A interromper a pipeline."
    exit 1
fi

# 4. Executar 03 Diffusion
echo "A executar script 03_diffusion.py ..."
python scripts/03_diffusion.py
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha ao treinar o modelo Diffusion. A interromper a pipeline."
    exit 1
fi

# 5. Executar 04 Avaliacao Final
echo "A executar script 04_evaluation.py (FID e KID) ..."
python scripts/04_evaluation.py
if [ $? -ne 0 ]; then
    echo "[ERRO] Falha na avaliacao dos modelos. A interromper a pipeline."
    exit 1
fi

echo ""
echo "PIPELINE CONCLUIDA COM SUCESSO!"
echo "Consultar as curvas, amostras geradas e metricas finais na pasta /results."
