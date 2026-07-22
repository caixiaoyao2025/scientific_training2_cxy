"""Demo B: Biomni WITH our MCP server - CAN do fastp QC"""
import os
import re
import types
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

from biomni.agent import A1

agent = A1(
    path="data",
    llm="Qwen/Qwen2.5-72B-Instruct",
    source="Custom",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-lufftravmhzgdpudfvbvzwrlfyctebizytmtlbynyiohtkij",
    expected_data_lake_files=[],
)

# Connect our MCP server
agent.add_mcp(config_path="./mcp_config_cluster.yaml")

# Fix system prompt
agent.system_prompt = re.sub(
    r"Import file: mcp_servers\.[^\n]+\n=+\n",
    "The following MCP tools are available directly in your namespace (call them as plain functions, NO import needed):\n",
    agent.system_prompt,
)

# Fix fake module
mcp_servers_mod = types.ModuleType("mcp_servers")
bio_mcp_mod = types.ModuleType("mcp_servers.bio_mcp")
mcp_servers_mod.bio_mcp = bio_mcp_mod
sys.modules["mcp_servers"] = mcp_servers_mod
sys.modules["mcp_servers.bio_mcp"] = bio_mcp_mod
for name, func in agent._custom_functions.items():
    setattr(bio_mcp_mod, name, func)

print("=" * 60)
print("WITH MCP SERVER")
print("=" * 60)
result = agent.go(
    "Run fastp quality control on data/sample.fastq. "
    "Output filtered reads to data/sample_filtered.fastq, "
    "HTML report to data/report.html, JSON report to data/report.json."
)
print(result)
