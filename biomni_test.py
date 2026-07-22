import os
import re
import types
import sys
import functools

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

from biomni.agent import A1
from biomni.tool.support_tools import _persistent_namespace

agent = A1(
    path="data",
    llm="qwen2.5:14b",
    source="Custom",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    expected_data_lake_files=[],
)
agent.add_mcp(config_path="./mcp_config_cluster.yaml")

# Wrap all MCP functions to auto-print results (exec() doesn't capture return values)
def make_robust_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is not None:
            print(result)
        return result
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

for name, func in list(agent._custom_functions.items()):
    wrapped = make_robust_wrapper(func)
    agent._custom_functions[name] = wrapped

# Create fake mcp_servers.bio_mcp module so "from mcp_servers.bio_mcp import xxx" works
mcp_servers_mod = types.ModuleType("mcp_servers")
bio_mcp_mod = types.ModuleType("mcp_servers.bio_mcp")
mcp_servers_mod.bio_mcp = bio_mcp_mod
sys.modules["mcp_servers"] = mcp_servers_mod
sys.modules["mcp_servers.bio_mcp"] = bio_mcp_mod

for name, func in agent._custom_functions.items():
    setattr(bio_mcp_mod, name, func)

# Also inject directly into REPL namespace so direct calls work
for name, func in agent._custom_functions.items():
    _persistent_namespace[name] = func

result = agent.go("Show all available tools using list_registered_tools")
print(result)
