import os
import re
import types
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

import server as srv
srv.ensure_runtime_directories()
srv.load_and_register_registry()

from biomni.agent import A1
from biomni.tool.support_tools import _persistent_namespace

api_key = os.environ.get("SILICONFLOW_API_KEY") or "sk-lufftravmhzgdpudfvbvzwrlfyctebizytmtlbynyiohtkij"

agent = A1(
    path="data",
    llm="Qwen/Qwen3-235B-A22B",
    source="Custom",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
    expected_data_lake_files=[],
)

# Disable tool retriever — avoids repeated full-prompt API calls that burn TPM
agent.config.use_tool_retriever = False

def make_direct_wrapper(tool_name):
    def wrapper(*args, **kwargs):
        try:
            result = srv.execute_registered_tool(tool_name, kwargs)
            if isinstance(result, list):
                parts = []
                for item in result:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif hasattr(item, "data"):
                        parts.append(f"[Image: {item.mimeType}]")
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"
    wrapper.__name__ = tool_name
    wrapper.__doc__ = f"Bio tool: {tool_name}"
    return wrapper

for name in srv.registered_tool_names:
    if not hasattr(agent, '_custom_functions'):
        agent._custom_functions = {}
    agent._custom_functions[name] = make_direct_wrapper(name)

agent.system_prompt = re.sub(
    r"Import file: mcp_servers\.[^\n]+\n=+\n",
    "The following tools are available directly in your namespace:\n",
    agent.system_prompt,
)

# Aggressively trim system prompt — keep core instructions + our7 tool names
# Qwen3-235B has 131K context; keep prompt under 30K chars (~7500 tokens) for safety
MAX_PROMPT_CHARS = 30000
if len(agent.system_prompt) > MAX_PROMPT_CHARS:
    agent.system_prompt = agent.system_prompt[:MAX_PROMPT_CHARS]
    print(f"Truncated system prompt to {MAX_PROMPT_CHARS} chars")

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
