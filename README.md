# Tool-Discovery Agent for Bioinformatics

An automated system that discovers bioinformatics tools from PubMed papers, converts them to MCP interfaces, and delivers them to downstream Bio-Agents.

## Core Pipeline

```
PubMed Search → GitHub Link Extraction → Standardization → Cleaning → MCP Registry → Biomni Agent
```

## Quick Start

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caixiaoyao2025/scientific_training2_cxy/blob/main/colab_demo.ipynb)

The notebook automatically installs Python 3.11, dependencies, and launches a Gradio interface. Uncomment the top code block to run the Tool Discovery Agent.

### Local Setup

```bash
pip install requests beautifulsoup4 pyyaml biomni "gradio>=5.0,<6.0" langchain-openai nest_asyncio mcp fastmcp

# Launch Gradio interface
python launch_gradio.py

# Run the discovery pipeline
python run_pipeline.py "bioinformatics protein engineering tools" 5
```

## Components

| File | Description |
|------|-------------|
| `agent.py` | PubMed search + GitHub link extraction from paper HTML |
| `convert.py` | GitHub repos → standardized tool format |
| `clean.py` | Filter irrelevant tools, enrich tags |
| `discovery_to_registry.py` | Tool library → MCP YAML registry |
| `merge_to_mcp.py` | Merge into MCP server registry |
| `run_pipeline.py` | End-to-end pipeline orchestrator |
| `server.py` | FastMCP bioinformatics server (9 built-in tools) |
| `launch_gradio.py` | Gradio launcher (Biomni + MCP tools) |

## Built-in Tools

| Tool | Function |
|------|----------|
| `fastp_qc` | FASTQ quality control |
| `samtools_flagstat` | BAM alignment statistics |
| `bedtools_intersect` | BED file intersection |
| `blastn_tabular` | BLASTN tabular output |
| `render_qc_png` | QC metrics visualization |
| `extract_pdf_summary` | PDF text extraction |
| `picard_collect_alignment_summary` | Picard alignment metrics |

## Automated Discovery

- **GitHub Actions**: runs daily, commits results back to repo
- **Local scheduling**: run `run_discover.bat` or set up Windows Task Scheduler

## Project Structure

```
├── agent.py                    # PubMed search + GitHub extraction
├── convert.py                  # Standardization + quality scoring
├── clean.py                    # Filtering + tag enrichment
├── discovery_to_registry.py    # Tool → MCP registry conversion
├── merge_to_mcp.py             # Merge into MCP server
├── run_pipeline.py             # End-to-end pipeline
├── server.py                   # FastMCP bioinformatics server
├── registry.yaml               # Built-in tool definitions
├── launch_gradio.py            # Gradio launcher
├── colab_demo.ipynb            # Google Colab notebook
├── run_discover.bat            # Windows scheduling script
├── create_test_data.py         # Generate simulated bio data
├── .github/workflows/
│   └── discover.yml            # GitHub Actions daily discovery
├── data/
│   ├── sample.fastq            # Simulated FASTQ (50 reads)
│   ├── sample.sam              # Simulated SAM (30 alignments)
│   ├── sample.bed              # Simulated BED (20 regions)
│   └── sample_annotation.gff   # Simulated GFF annotations
└── biomni_test.py              # Biomni integration test
```
