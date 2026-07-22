"""Demo A: Biomni WITHOUT our MCP server - cannot do fastp QC"""
import os

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

# NO add_mcp() call - Biomni only has built-in tools
print("=" * 60)
print("WITHOUT MCP SERVER")
print("=" * 60)
result = agent.go(
    "Run fastp quality control on data/sample.fastq. "
    "Output filtered reads to data/sample_filtered.fastq, "
    "HTML report to data/report.html, JSON report to data/report.json."
)
print(result)
