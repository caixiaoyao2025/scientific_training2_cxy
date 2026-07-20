#!/bin/bash
# ============================================================
# 集群上每天运行（环境销毁后需要重新 setup_cluster.sh）
# ============================================================
set -e

ENV_DIR=/caixiaoyao/envs/biomni311
MODEL_DIR=/caixiaoyao/ollama_models

echo "=== 激活 Conda 环境 ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_DIR

echo "=== 启动 Ollama（模型从持久路径加载）==="
export OLLAMA_MODELS=$MODEL_DIR
ollama serve &
sleep 3

echo "=== 运行 Bio-Agent ==="
python biomni_test.py
