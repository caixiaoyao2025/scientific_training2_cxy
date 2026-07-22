#!/bin/bash
# ============================================================
# 集群上每天运行
# ============================================================
set -e

ENV_DIR=/caixiaoyao/envs/biomni311
MODEL_PATH=/caixiaoyao/ollama_models/qwen2.5-14b/qwen2.5-14b-instruct-q6_k.gguf

echo "=== 激活 Conda 环境 ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_DIR

echo "=== 启动 LLM Server (llama-cpp-python) ==="
export MODEL_PATH=$MODEL_PATH
nohup python llama_server.py > /tmp/llama_server.log 2>&1 &
LLAMA_PID=$!
echo "LLM server PID: $LLAMA_PID"

# Wait for model to load (12GB Q6_K needs ~30-60s)
echo "Waiting for model to load..."
for i in $(seq 1 60); do
    if curl -s http://localhost:11434/v1/models > /dev/null 2>&1; then
        echo "LLM server is running (ready after ${i}s)"
        break
    fi
    sleep 5
    echo "  still loading... (${i}x5s)"
done

if ! curl -s http://localhost:11434/v1/models > /dev/null 2>&1; then
    echo "ERROR: LLM server failed to start. Check /tmp/llama_server.log"
    tail -20 /tmp/llama_server.log
    exit 1
fi

echo "=== 运行 Bio-Agent 测试 ==="
python biomni_test.py

echo "=== 停止 LLM Server ==="
kill $LLAMA_PID 2>/dev/null || true
