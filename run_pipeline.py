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

def step1_discover(query="bioinformatics protein engineering tools", max_results=5,
                   paper_timeout=30):
    print("=" * 60)
    print("STEP 1: Discovering tools from PubMed papers...")
    print("=" * 60)
    from agent import search_papers, load_seen_papers, mark_paper_as_seen, fetch_html_from_doi, extract_github_links
    import time
    import threading

    seen = load_seen_papers()

    # Keep paging PubMed until we collect `max_results` *unseen* papers.
    # Papers already processed in earlier runs don't count against the limit.
    # Pull a 50-paper window per request to reduce round-trips.
    batch = max(max_results, 50)
    start = 0
    papers = []
    while len(papers) < max_results and start < 200:
        page = search_papers(query, max_results=batch, retstart=start)
        if not page:
            break
        fresh = [p for p in page if (p.get('pmid') or p.get('doi')) not in seen]
        papers.extend(fresh)
        print(f"  batch@{start}: {len(page)} papers, {len(fresh)} new "
              f"(need {max_results - len(papers)} more)")
        if len(page) < batch:
            break
        start += batch
        time.sleep(0.5)
    papers = papers[:max_results]
    print(f"Found {len(papers)} new papers ({len(seen)} already seen total)")

    results = []

    def _fetch_with_timeout(doi):
        box = {}

        def worker():
            try:
                box["content"] = fetch_html_from_doi(doi)
            except Exception as exc:  # noqa: BLE001
                box["error"] = str(exc)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=paper_timeout)
        if t.is_alive():
            box["timed_out"] = True
        return box

    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('pmid') or paper.get('doi')
        if paper_id and paper_id in seen:
            print(f"  [{i}] Skipping (seen): {paper['title'][:50]}...")
            continue

        print(f"  [{i}] {paper['title'][:70]}...")
        html_content = None
        if paper.get('doi'):
            box = _fetch_with_timeout(paper['doi'])
            if box.get("timed_out"):
                print(f"    !! fetch timed out after {paper_timeout}s, using abstract")
            html_content = box.get("content")
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
    if not os.path.exists("tool_library.json"):
        print("No tool_library.json found, skipping clean.")
        return
    import importlib
    import clean
    importlib.reload(clean)
    print("Clean done")

def step3_6_execute():
    print("\n" + "=" * 60)
    print("STEP 3.6: Execution test (install + smoke run in isolated venv)...")
    print("=" * 60)
    if not os.path.exists("tool_verification.json"):
        print("No tool_verification.json found, skipping execution test.")
        return
    from execute_test import execute_tool_library
    print("Installing each verified/repo_ok tool into a venv and smoke-running it...")
    results = execute_tool_library("tool_verification.json", "tool_execution.json")
    n_pass = sum(1 for r in results if r.get("status") == "passed")
    n_env = sum(1 for r in results if r.get("status") == "env_issue")
    n_inc = sum(1 for r in results if r.get("status") == "incomplete")
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    n_time = sum(1 for r in results if r.get("status") == "timeout")
    print(f"  -> {n_pass} passed, {n_env} env_issue, {n_inc} incomplete, "
          f"{n_fail} failed, {n_time} timeout, {n_skip} skipped")

def step4_to_registry():
    print("\n" + "=" * 60)
    print("STEP 4: Converting to MCP registry format...")
    print("=" * 60)
    if not os.path.exists("tool_library_clean.json"):
        print("No tool_library_clean.json found, skipping registry conversion.")
        return
    from discovery_to_registry import load_tool_library, convert_to_registry

    tools = load_tool_library()
    convert_to_registry(tools, "discovered_registry.yaml",
                        verification_file="tool_verification.json",
                        require_passed=True)
    print(f"Registry generated with {len(tools)} tools")

def step3_5_verify():
    print("\n" + "=" * 60)
    print("STEP 3.5: Verifying discovered repos (clone/license/entry)...")
    print("=" * 60)
    if not os.path.exists("tool_library_clean.json"):
        print("No tool_library_clean.json found, skipping verification.")
        return
    from verify_repo import verify_tool_library
    from discovery_to_registry import load_tool_library
    tools = load_tool_library("tool_library_clean.json")
    print(f"Verifying {len(tools)} tools (blobless clone, no weight downloads)...")
    results = verify_tool_library(tools, out_json="tool_verification.json")
    ok = [r for r in results if r.get("status") in ("verified", "repo_ok")]
    bad = [r for r in results if r.get("status") not in ("verified", "repo_ok")]
    print(f"  -> {len(ok)} verified/repo_ok, {len(bad)} unverified")
    for r in bad[:10]:
        print(f"     EXCLUDED {r.get('tool','?')}: {r.get('reason','')[:70]}")

def step5_register_to_mcp():
    print("\n" + "=" * 60)
    print("STEP 5: Registering tools to MCP server...")
    print("=" * 60)
    if not os.path.exists("discovered_registry.yaml"):
        print("No discovered_registry.yaml found, skipping merge.")
        return
    from merge_to_mcp import merge_registries
    merge_registries()

def run_full_pipeline(query="bioinformatics protein engineering tools", max_results=5):
    print("TOOL-DISCOVERY AGENT PIPELINE")
    print("=" * 60)

    discovered = step1_discover(query=query, max_results=max_results)
    if not discovered:
        print("\nNo new papers with tools found. Skipping conversion/verification "
              "and resetting stale intermediates so old data is not re-processed.")
        for stale in ("tool_library.json", "tool_library_clean.json",
                      "tool_verification.json", "tool_execution.json",
                      "discovered_registry.yaml"):
            if os.path.exists(stale):
                os.remove(stale)
        print("PIPELINE COMPLETE (nothing new)")
        return
    step2_convert()
    step3_clean()
    step3_5_verify()
    step3_6_execute()
    step4_to_registry()
    step5_register_to_mcp()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Files generated:")
    print("  - github_from_html.json     (raw discoveries)")
    print("  - tool_library.json         (standardized tools)")
    print("  - tool_library_clean.json   (cleaned tools)")
    print("  - tool_verification.json    (repo verification evidence)")
    print("  - tool_execution.json       (install + smoke-run results)")
    print("  - excluded_tools.json       (rejected with reasons)")
    print("  - discovered_registry.yaml  (MCP registry format)")
    print("  - data/mcp_registry.yaml    (auto-merged into MCP server)")
    print("\nNew tools are available in MCP server on next container start.")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "bioinformatics protein engineering tools"
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_full_pipeline(query=query, max_results=max_results)
