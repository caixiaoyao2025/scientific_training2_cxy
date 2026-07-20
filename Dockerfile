FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    MCP_APP_ROOT=/app \
    MCP_DATA_ROOT=/data \
    MCP_REGISTRY_PATH=/app/registry.yaml \
    MCP_USER_REGISTRY_PATH=/data/mcp_registry.yaml \
    MCP_TOOL_BIN_ROOT=/data/mcp_tools/bin \
    MCP_TRANSPORT=stdio \
    MCP_HOST=127.0.0.1 \
    MCP_PORT=8000 \
    MCP_PATH=/mcp

RUN pip install --no-cache-dir --timeout 120 \
    fastmcp \
    pyyaml \
    pandas \
    tabulate \
    pypdf

COPY server.py registry.yaml /app/
RUN mkdir -p /data/mcp_outputs

VOLUME ["/data"]
EXPOSE 8000
ENTRYPOINT ["python", "-u", "/app/server.py"]
