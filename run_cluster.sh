#!/bin/bash
# ============================================================
# 集群上每天运行
# ============================================================
set -e

ENV_DIR=/caixiaoyao/envs/biomni311
MODEL_PATH=/caixiaoyao/ollama_models/qwen2.5-14b/qwen2.5-14b-instruct-q4_k_m.gguf

echo "=== 激活 Conda 环境 ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_DIR

echo "=== 修补 Biomni make_mcp_wrapper bug ==="
A1_PY=$ENV_DIR/lib/python3.11/site-packages/biomni/agent/a1.py
# fix: loop.create_task() returns unawaited Task object → empty observations
python -c "
with open('$A1_PY', 'r') as f:
    lines = f.readlines()
patched = False
for i, line in enumerate(lines):
    if 'return loop.create_task(async_tool_call())' in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines[i] = indent + 'return asyncio.run(async_tool_call())\n'
        # remove the preceding 'try:' and 'get_running_loop' and 'except' lines
        for j in range(i-1, max(i-4, 0), -1):
            if 'get_running_loop' in lines[j] or 'try:' in lines[j].strip():
                lines[j] = ''
            if 'except RuntimeError:' in lines[j]:
                lines[j] = ''
                break
        patched = True
        break
if patched:
    with open('$A1_PY', 'w') as f:
        f.writelines(lines)
    print('Patched a1.py successfully')
else:
    print('a1.py already patched or pattern not found')
"

echo "=== 启动 LLM Server (llama-cpp-python) ==="
export MODEL_PATH=$MODEL_PATH
nohup python llama_server.py > /tmp/llama_server.log 2>&1 &
LLAMA_PID=$!
echo "LLM server PID: $LLAMA_PID"
sleep 5

# Verify server is up
if ! curl -s http://localhost:11434/v1/models > /dev/null 2>&1; then
    echo "ERROR: LLM server failed to start. Check /tmp/llama_server.log"
    cat /tmp/llama_server.log
    exit 1
fi
echo "LLM server is running"

echo "=== 运行 Bio-Agent 测试 ==="
python biomni_test.py

echo "=== 停止 LLM Server ==="
kill $LLAMA_PID 2>/dev/null || true
