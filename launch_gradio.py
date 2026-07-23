import os
import re
import types
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

# Ensure bioinformatics tools are on PATH (Colab apt-get installs them here)
_extra_paths = ["/usr/bin", "/usr/local/bin", "/usr/sbin"]
os.environ["PATH"] = ":".join(_extra_paths) + ":" + os.environ.get("PATH", "")

import server as srv
srv.ensure_runtime_directories()
srv.load_and_register_registry()

from biomni.agent import A1
from biomni.tool.support_tools import _persistent_namespace

api_key = os.environ.get("SILICONFLOW_API_KEY") or "sk-lufftravmhzgdpudfvbvzwrlfyctebizytmtlbynyiohtkij"

agent = A1(
    path="data",
    llm="Qwen/Qwen2.5-72B-Instruct-128K",
    source="Custom",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
    expected_data_lake_files=[],
)

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

# Force Qwen to use <execute> tags instead of markdown code blocks
agent.system_prompt += (
    "\n\n=== CRITICAL FORMATTING RULE ===\n"
    "EVERY response that contains code to run MUST include an <execute> tag block.\n"
    "Format: <execute>\\n#!bash\\n<your commands>\\n</execute>\n"
    "Or: <execute>\\n#!python\\n<your code>\\n</execute>\n"
    "Do NOT use markdown code blocks (```) for code execution.\n"
    "Always output <execute> tags directly. This is mandatory.\n"
    "=== END RULE ===\n"
)

# ── Monkey-patch: wrap LLM to convert markdown code blocks to <execute> tags ──
# Qwen2.5 outputs ```bash ... ``` but Biomni's execute() only reads <execute> tags.
# We intercept the LLM response and inject <execute> tags when missing.
from langchain_core.messages import AIMessage

_orig_llm = agent.llm

class _LLMProxy:
    """Thin proxy that wraps ChatOpenAI and post-processes responses."""
    def __init__(self, llm):
        object.__setattr__(self, "_llm", llm)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_llm"), name)

    def invoke(self, messages, *args, **kwargs):
        response = object.__getattribute__(self, "_llm").invoke(messages, *args, **kwargs)
        content = response.content if hasattr(response, "content") else str(response)

        # Handle list content (some providers return list of blocks)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    part = block.get("text") or block.get("content") or ""
                    if isinstance(part, str):
                        text_parts.append(part)
            content = "".join(text_parts)

        if not isinstance(content, str):
            return response

        if "<execute>" not in content and "<solution>" not in content:
            code_block_re = re.compile(r'```(?:bash|sh|shell|python|py|r)?\s*\n(.*?)```', re.DOTALL)
            blocks = code_block_re.findall(content)
            if blocks:
                extracted = "\n".join(blocks)
                first_type = re.search(r'```(\w+)', content)
                code_type = "python"
                if first_type and first_type.group(1).lower() in ("bash", "sh", "shell", "cli"):
                    code_type = "bash"
                fixed = content + f"\n\n<execute>\n#!{code_type}\n{extracted}\n</execute>"
                response = AIMessage(content=fixed)

        return response

object.__setattr__(agent, "llm", _LLMProxy(_orig_llm))

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
