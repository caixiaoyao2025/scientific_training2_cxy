import os
import re
import types
import sys
import asyncio
import concurrent.futures

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

from biomni.agent import A1
from biomni.tool.support_tools import _persistent_namespace
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import ClientSession

api_key = os.environ.get("SILICONFLOW_API_KEY") or "sk-lufftravmhzgdpudfvbvzwrlfyctebizytmtlbynyiohtkij"

agent = A1(
    path="data",
    llm="Qwen/Qwen2.5-72B-Instruct",
    source="Custom",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
    expected_data_lake_files=[],
)
agent.add_mcp(config_path="./mcp_config_cluster.yaml")

def make_working_wrapper(tool_name):
    def wrapper(*args, **kwargs):
        async def call():
            params = StdioServerParameters(command="python", args=["server.py"])
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, kwargs)
                    content = result.content[0]
                    if hasattr(content, "text"):
                        return content.text
                    return str(content)
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, call()).result(timeout=120)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"
    wrapper.__name__ = tool_name
    wrapper.__doc__ = f"MCP tool: {tool_name}"
    return wrapper

for name in list(agent._custom_functions.keys()):
    agent._custom_functions[name] = make_working_wrapper(name)

agent.system_prompt = re.sub(
    r"Import file: mcp_servers\.[^\n]+\n=+\n",
    "The following MCP tools are available directly in your namespace (call them as plain functions, NO import needed):\n",
    agent.system_prompt,
)

mcp_servers_mod = types.ModuleType("mcp_servers")
bio_mcp_mod = types.ModuleType("mcp_servers.bio_mcp")
mcp_servers_mod.bio_mcp = bio_mcp_mod
sys.modules["mcp_servers"] = mcp_servers_mod
sys.modules["mcp_servers.bio_mcp"] = bio_mcp_mod
for name, func in agent._custom_functions.items():
    setattr(bio_mcp_mod, name, func)
    _persistent_namespace[name] = func

print("Launching Gradio interface...")
agent.launch_gradio_demo(share=True, server_name="0.0.0.0")
