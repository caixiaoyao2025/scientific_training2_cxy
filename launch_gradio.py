import os
import re
import types
import sys
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

from biomni.agent import A1
from biomni.tool.support_tools import _persistent_namespace
import gradio as gr

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

agent.system_prompt = re.sub(
    r"Import file: mcp_servers\.[^\n]+\n=+\n",
    "The following tools are available directly in your namespace:\n",
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

def chat(message, history):
    try:
        result = agent.go(message)
        return result
    except Exception as e:
        return f"Error: {e}"

def upload_file(file):
    if file is None:
        return "No file selected"
    filename = os.path.basename(file.name)
    dest = os.path.join("data", filename)
    shutil.copy(file.name, dest)
    return f"Uploaded to data/{filename}"

with gr.Blocks(title="Bioinformatics Tool-Discovery Agent") as demo:
    gr.Markdown("# Bioinformatics Tool-Discovery Agent")
    gr.Markdown("上传文件，然后输入生信任务，AI 自动选择工具并执行")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传文件", file_types=[".fastq", ".fq", ".fasta", ".fa", ".bam", ".bed", ".csv", ".tsv", ".pdf"])
            upload_btn = gr.Button("上传")
            upload_status = gr.Textbox(label="状态", interactive=False)
            upload_btn.click(fn=upload_file, inputs=file_input, outputs=upload_status)

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat,
                examples=[
                    "Run fastp quality control on data/sample.fastq and generate an HTML report",
                    "List all available tools",
                ],
            )

demo.launch(server_name="0.0.0.0", share=True)
