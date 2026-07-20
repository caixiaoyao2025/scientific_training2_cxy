# Tool-Discovery Agent for Bioinformatics

An automated agent that discovers bioinformatics tools from PubMed papers, extracts GitHub repositories, and converts them into standardized [MCP (Model Context Protocol)](https://modelcontextprotocol.io) tool interfaces for downstream Bio-Agents.

## Overview

```
PubMed Paper → Full-text Fetch → GitHub Link Extraction → Standardization → Cleaning → MCP Registry
```

| Component | Description |
|---|---|
| `agent.py` | Searches PubMed, fetches paper full-text (via Jina Reader / direct HTML parsing), extracts GitHub links |
| `convert.py` | Standardizes raw discoveries into a uniform tool format with quality scoring |
| `clean.py` | Filters irrelevant tools and enriches tags |
| `discovery_to_registry.py` | Converts cleaned tools into MCP YAML registry format |
| `merge_to_mcp.py` | Merges discovered tools into the MCP server's user registry |
| `run_pipeline.py` | Runs the full end-to-end pipeline |
| `server.py` | FastMCP bioinformatics server with dynamic tool registration |

## Quick Start: Google Colab

The easiest way to try the discovery pipeline is on Google Colab — no local setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caixiaoyao-cxy/SCI1003_TEAM11/blob/main/colab_demo.ipynb)

The notebook runs the full discovery pipeline:
1. Searches PubMed for bioinformatics papers
2. Fetches paper full-text and extracts GitHub links
3. Standardizes and scores discovered tools
4. Generates an MCP-compatible registry YAML

## Local Usage

### Requirements

- Python 3.10+
- Docker (for MCP server)

```bash
pip install -r requirements.txt
```

### Run the Discovery Pipeline

```bash
python run_pipeline.py "bioinformatics protein engineering tools" 5
```

This will:
1. Search PubMed for 5 papers matching the query
2. Extract GitHub tool links from paper full-texts
3. Standardize, clean, and convert to MCP registry format
4. Merge into `data/mcp_registry.yaml`

### Run the MCP Server

```bash
# Build
docker build --platform=linux/amd64 -t bio-mcp .

# Run (stdio mode)
docker run --rm -i -v $(pwd)/data:/data bio-mcp

# Run (HTTP mode)
docker run -d --name bio-mcp-http -p 127.0.0.1:8765:8000 \
  -v $(pwd)/data:/data \
  -e MCP_TRANSPORT=streamable-http \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=8000 \
  -e MCP_PATH=/mcp \
  bio-mcp
```

### Built-In Tools

The MCP server ships with 7 built-in bioinformatics tools:

- `samtools_flagstat` — BAM alignment statistics
- `bedtools_intersect` — BED file intersection
- `blastn_tabular` — BLASTN tabular output
- `fastp_qc` — FASTQ quality control
- `render_qc_png` — QC metrics visualization
- `extract_pdf_summary` — PDF text extraction
- `picard_collect_alignment_summary` — Picard alignment metrics

Plus management tools: `install_bio_tool`, `append_tool_to_registry`, `list_registered_tools`, `reload_registry`.

## Project Structure

```
├── agent.py                    # PubMed search + GitHub extraction
├── convert.py                  # Standardization + quality scoring
├── clean.py                    # Filtering + tag enrichment
├── discovery_to_registry.py    # Tool → MCP registry conversion
├── merge_to_mcp.py             # Merge into MCP server registry
├── run_pipeline.py             # End-to-end pipeline orchestrator
├── server.py                   # FastMCP bioinformatics server
├── registry.yaml               # Built-in tool definitions
├── Dockerfile                  # Docker image for MCP server
├── colab_demo.ipynb            # Google Colab notebook
├── requirements.txt            # Python dependencies
├── tool_schema.json            # JSON schema for tool format
├── biomni_test.py              # Biomni agent integration test
├── mcp_config.yaml             # MCP config for Biomni
├── mcp_direct_test.py          # Direct MCP connection test
└── data/                       # Runtime data directory
```

## MCP Server Features

- **No Bioconda**: uses `python:3.11-slim-bookworm` with apt + pip
- **Dynamic registry**: built-in tools from `registry.yaml`, user tools from `data/mcp_registry.yaml`
- **Secure execution**: `shlex.split`, no `shell=True`, path firewall inside `/data`
- **Large-output handling**: previews small results, persists full output to `/data/mcp_outputs`
- **Multimodal outputs**: image, dataframe, PDF, text rendering
- **Dual transport**: stdio and streamable-http

## License

TBD.
