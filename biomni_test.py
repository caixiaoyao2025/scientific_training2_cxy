import os
import re
import types
import sys
import asyncio
import functools

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

from biomni.agent import A1
from biomni.tool.support_tools import _persistent_namespace
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import ClientSession

agent = A1(
    path="data",
    llm="qwen2.5:14b",
    source="Custom",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    expected_data_lake_files=[],
)
agent.add_mcp(config_path="./mcp_config_cluster.yaml")

# Replace ALL broken MCP wrappers with a fixed version
# The original make_mcp_wrapper in a1.py returns None due to loop.create_task() bug
MCP_CMD = "python"
MCP_ARGS = ["server.py"]

def make_fixed_wrapper(tool_name, description=""):
    """Create a working wrapper that calls MCP tool via asyncio.run() in a new thread"""
    def wrapper(**kwargs):
        async def call():
            params = StdioServerParameters(command=MCP_CMD, args=MCP_ARGS)
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, kwargs)
                    content = result.content[0]
                    if hasattr(content, "text"):
                        return content.text
                    return str(content)
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, call())
                return future.result(timeout=60)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"
    wrapper.__name__ = tool_name
    wrapper.__doc__ = description or f"MCP tool: {tool_name}"
    return wrapper

# Get tool info from the already-registered tools and rebuild wrappers
tool_names = list(agent._custom_functions.keys())
print(f"Replacing {len(tool_names)} broken MCP wrappers...")

for name in tool_names:
    fixed = make_fixed_wrapper(name)
    agent._custom_functions[name] = fixed

# Create fake mcp_servers.bio_mcp module
mcp_servers_mod = types.ModuleType("mcp_servers")
bio_mcp_mod = types.ModuleType("mcp_servers.bio_mcp")
mcp_servers_mod.bio_mcp = bio_mcp_mod
sys.modules["mcp_servers"] = mcp_servers_mod
sys.modules["mcp_servers.bio_mcp"] = bio_mcp_mod

for name, func in agent._custom_functions.items():
    setattr(bio_mcp_mod, name, func)
    _persistent_namespace[name] = func

result = agent.go("Show all available tools using list_registered_tools")
print(result)
