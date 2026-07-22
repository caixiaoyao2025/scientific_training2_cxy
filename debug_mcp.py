"""Debug: test MCP function call directly"""
import os, sys, types, asyncio

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

# Test 1: call the wrapper directly
print("=== Test 1: call wrapper directly ===")
func = agent._custom_functions.get("list_registered_tools")
if func:
    print(f"Function: {func}")
    print(f"Module: {func.__module__}")
    result = func()
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
else:
    print("Function not found in _custom_functions")

# Test 2: check what _custom_functions contains
print("\n=== Test 2: _custom_functions keys ===")
for name in agent._custom_functions:
    print(f"  {name}")

# Test 3: test via fake module
print("\n=== Test 3: test via fake module ===")
mcp_servers_mod = types.ModuleType("mcp_servers")
bio_mcp_mod = types.ModuleType("mcp_servers.bio_mcp")
mcp_servers_mod.bio_mcp = bio_mcp_mod
sys.modules["mcp_servers"] = mcp_servers_mod
sys.modules["mcp_servers.bio_mcp"] = bio_mcp_mod

for name, fn in agent._custom_functions.items():
    setattr(bio_mcp_mod, name, fn)

from mcp_servers.bio_mcp import list_registered_tools
print(f"Imported function: {list_registered_tools}")
result = list_registered_tools()
print(f"Result type: {type(result)}")
print(f"Result: {result}")
