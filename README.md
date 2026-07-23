# Tool-Discovery Agent for Bioinformatics

自动化发现生信工具、转换为 MCP 接口、交付给下游 Bio-Agent 的端到端系统。

## 核心流程

```
PubMed 论文检索 → GitHub 链接提取 → 标准化 → 清洗 → MCP Registry → Biomni Agent 调用
```

## 快速开始

### Google Colab（推荐）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caixiaoyao2025/scientific_training2_cxy/blob/main/colab_demo.ipynb)

Notebook 会自动安装 Python 3.11 环境、下载依赖、启动 Gradio 界面。取消注释顶部的代码块可运行工具发现 Agent。

### 本地运行

```bash
pip install requests beautifulsoup4 pyyaml biomni "gradio>=5.0,<6.0" langchain-openai nest_asyncio mcp fastmcp

# 启动 Gradio 界面
python launch_gradio.py

# 运行工具发现 Pipeline
python run_pipeline.py "bioinformatics protein engineering tools" 5
```

## 组件说明

| 文件 | 功能 |
|------|------|
| `agent.py` | PubMed 搜索 + 论文 HTML 提取 GitHub 链接 |
| `convert.py` | GitHub 仓库 → 标准化工具格式 |
| `clean.py` | 过滤无关工具、补充标签 |
| `discovery_to_registry.py` | 工具库 → MCP YAML registry |
| `merge_to_mcp.py` | 合并到 MCP server registry |
| `run_pipeline.py` | 一键运行完整发现流程 |
| `server.py` | FastMCP 生信工具服务端（9个内置工具） |
| `launch_gradio.py` | Gradio 界面启动器（Biomni + MCP 工具） |

## 内置工具

| 工具 | 功能 |
|------|------|
| `fastp_qc` | FASTQ 质量控制 |
| `samtools_flagstat` | BAM 比对统计 |
| `bedtools_intersect` | BED 文件交集运算 |
| `blastn_tabular` | BLASTN 表格输出 |
| `render_qc_png` | QC 指标可视化 |
| `extract_pdf_summary` | PDF 文本提取 |
| `picard_collect_alignment_summary` | Picard 比对指标 |

## 自动化发现

- **GitHub Actions**：每日定时运行，结果自动 commit 回 repo
- **本地定时**：运行 `run_discover.bat` 或配置 Windows 任务计划

## 项目结构

```
├── agent.py                    # PubMed 搜索 + GitHub 提取
├── convert.py                  # 标准化 + 质量评分
├── clean.py                    # 过滤 + 标签补充
├── discovery_to_registry.py    # 工具 → MCP registry
├── merge_to_mcp.py             # 合并到 MCP server
├── run_pipeline.py             # 完整流程编排
├── server.py                   # FastMCP 生信服务端
├── registry.yaml               # 内置工具定义
├── launch_gradio.py            # Gradio 启动器
├── colab_demo.ipynb            # Google Colab notebook
├── run_discover.bat            # Windows 定时发现脚本
├── create_test_data.py         # 生成模拟生信数据
├── .github/workflows/
│   └── discover.yml            # GitHub Actions 每日发现
├── data/
│   ├── sample.fastq            # 模拟 FASTQ（50 reads）
│   ├── sample.sam              # 模拟 SAM（30 alignments）
│   ├── sample.bed              # 模拟 BED（20 调控区）
│   └── sample_annotation.gff   # 模拟 GFF 注释
└── biomni_test.py              # Biomni 集成测试
```
