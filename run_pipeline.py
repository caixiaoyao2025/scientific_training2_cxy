"""
Tool-Discovery Agent Pipeline
=============================
完整流程：
1. 搜索 PubMed 生信论文
2. 获取全文，提取 GitHub 链接
3. 标准化为工具格式
4. 清洗过滤
5. 转换为 MCP registry 格式
6. 通过 append_tool_to_registry 注册到运行中的 MCP server
"""
import json
import subprocess
import sys
import os

def step1_discover(query="bioinformatics protein engineering tools", max_results=5):
    print("=" * 60)
    print("STEP 1: Discovering tools from PubMed papers...")
    print("=" * 60)
    from agent import search_papers, load_seen_papers, mark_paper_as_seen, fetch_html_from_doi, extract_github_links
    import time

    papers = search_papers(query, max_results=max_results)
    print(f"Found {len(papers)} papers")

    seen = load_seen_papers()
    results = []

    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('pmid') or paper.get('doi')
        if paper_id and paper_id in seen:
            print(f"  [{i}] Skipping (seen): {paper['title'][:50]}...")
            continue

        print(f"  [{i}] {paper['title'][:70]}...")
        html_content = None
        if paper.get('doi'):
            html_content = fetch_html_from_doi(paper['doi'])
        if not html_content:
            html_content = paper.get('abstract', '')
        if not html_content or len(html_content) < 50:
            if paper_id:
                mark_paper_as_seen(paper_id)
            continue

        github_links = extract_github_links(html_content)
        if github_links:
            results.append({
                "title": paper['title'][:100],
                "doi": paper['doi'],
                "github_links": github_links,
                "url": paper['url']
            })
            print(f"    Found {len(github_links)} GitHub links")
        else:
            print(f"    No GitHub links found")

        if paper_id:
            mark_paper_as_seen(paper_id)
        time.sleep(1)

    with open("github_from_html.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Discovered {len(results)} papers with tools")
    return results

def step2_convert():
    print("\n" + "=" * 60)
    print("STEP 2: Converting to standardized format...")
    print("=" * 60)
    from convert import load_raw_results, convert_to_standard, save_tool_library, generate_summary

    raw = load_raw_results("github_from_html.json")
    if not raw:
        print("No new data to convert")
        return []

    standardized = convert_to_standard(raw)
    save_tool_library(standardized, "tool_library.json")
    summary = generate_summary(standardized)
    print(f"Converted {len(standardized)} tools")
    return standardized

def step3_clean():
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning tool library...")
    print("=" * 60)
    import importlib
    import clean
    importlib.reload(clean)
    print("Clean done")

def step4_to_registry():
    print("\n" + "=" * 60)
    print("STEP 4: Converting to MCP registry format...")
    print("=" * 60)
    from discovery_to_registry import load_tool_library, convert_to_registry

    tools = load_tool_library()
    convert_to_registry(tools, "discovered_registry.yaml")
    print(f"Registry generated with {len(tools)} tools")

def step5_register_to_mcp():
    print("\n" + "=" * 60)
    print("STEP 5: Registering tools to MCP server...")
    print("=" * 60)
    from merge_to_mcp import merge_registries
    merge_registries()

def run_full_pipeline(query="bioinformatics protein engineering tools", max_results=5):
    print("TOOL-DISCOVERY AGENT PIPELINE")
    print("=" * 60)

    step1_discover(query=query, max_results=max_results)
    step2_convert()
    step3_clean()
    step4_to_registry()
    step5_register_to_mcp()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Files generated:")
    print("  - github_from_html.json     (raw discoveries)")
    print("  - tool_library.json         (standardized tools)")
    print("  - tool_library_clean.json   (cleaned tools)")
    print("  - discovered_registry.yaml  (MCP registry format)")
    print("  - data/mcp_registry.yaml    (auto-merged into MCP server)")
    print("\nNew tools are available in MCP server on next container start.")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "bioinformatics protein engineering tools"
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_full_pipeline(query=query, max_results=max_results)
