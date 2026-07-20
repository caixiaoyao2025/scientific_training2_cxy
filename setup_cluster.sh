#!/bin/bash
# ============================================================
# 集群一键安装（只用跑一次，所有东西都在 /caixiaoyao/ 下持久）
# ============================================================
set -e

ENV_DIR=/caixiaoyao/envs/biomni311
MODEL_DIR=/caixiaoyao/ollama_models

echo "=== Step 1: Conda 环境（持久路径: $ENV_DIR）==="
conda create -p $ENV_DIR python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_DIR

echo "=== Step 2: Python 依赖 ==="
pip install biomni langchain-core langchain-openai langchain-anthropic langchain-community langgraph nest_asyncio mcp pandas requests beautifulsoup4 pyyaml tabulate python-dateutil -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Step 3: Ollama ==="
if ! command -v ollama &> /dev/null; then
    echo "安装 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== Step 4: 下载 14B 模型（持久路径: $MODEL_DIR）==="
mkdir -p $MODEL_DIR
export OLLAMA_MODELS=$MODEL_DIR
ollama serve &
OLLAMA_PID=$!
sleep 3
ollama pull qwen2.5:14b
kill $OLLAMA_PID 2>/dev/null

echo ""
echo "=============================="
echo "✅ 安装完成！"
echo "以后每次运行: bash run_cluster.sh"
echo "=============================="
