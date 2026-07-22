import os
import re
import types
import sys
import functools

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

from biomni.agent import A1

agent = A1(
    path="data",
    llm="qwen2.5:14b",
    source="Custom",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    expected_data_lake_files=[],
)
agent.add_mcp(config_path="./mcp_config_cluster.yaml")

# Fix: create fake mcp_servers.bio_mcp module so "from mcp_servers.bio_mcp import xxx" works
mcp_servers_mod = types.ModuleType("mcp_servers")
bio_mcp_mod = types.ModuleType("mcp_servers.bio_mcp")
mcp_servers_mod.bio_mcp = bio_mcp_mod
sys.modules["mcp_servers"] = mcp_servers_mod
sys.modules["mcp_servers.bio_mcp"] = bio_mcp_mod

for name, func in agent._custom_functions.items():
    setattr(bio_mcp_mod, name, func)

result = agent.go("Show all available tools using list_registered_tools")
print(result)
