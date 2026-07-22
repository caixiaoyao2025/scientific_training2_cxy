import os
import re
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

# Fix 1: replace "Import file: mcp_servers.*" so model doesn't try to import
agent.system_prompt = re.sub(
    r"Import file: mcp_servers\.[^\n]+\n=+\n",
    "The following MCP tools are available directly in your namespace (call them as plain functions, NO import needed):\n",
    agent.system_prompt,
)

# Fix 2: wrap all custom functions to auto-print results
# exec() doesn't capture return values, so we force-print them
def make_printing_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is not None:
            print(result)
        return result
    return wrapper

for name, func in agent._custom_functions.items():
    agent._custom_functions[name] = make_printing_wrapper(func)

result = agent.go("Use list_registered_tools tool to show all available tools")
print(result)
