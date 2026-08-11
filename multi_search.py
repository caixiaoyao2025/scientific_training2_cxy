"""Batch discovery across multiple bioinformatics keyword queries.

Pulls a 50-paper window per query, keeps only papers not already in
seen_papers.json, fetches fulltext (PMC first, then direct parse), and
collects GitHub links. Writes github_from_html.json.
"""
import json
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import (
    search_papers, load_seen_papers, mark_paper_as_seen,
    fetch_html_from_doi, extract_github_links,
)

KEYWORDS = [
    "single cell analysis tool",
    "metagenomics pipeline",
    "protein structure prediction",
    "CRISPR guide design",
    "variant calling tool",
    "RNA-seq differential expression",
    "drug target discovery",
    "phylogenetic inference",
    "epigenomics data analysis",
    "antibody design",
    "microbiome analysis workflow",
    "proteomics data analysis",
]

# One query per keyword, so every keyword gets searched exactly once.
QUERIES = [
    f"({kw}) AND (software OR tool OR pipeline) AND (code OR github OR repository)"
    for kw in KEYWORDS
]

MAX_PER_QUERY = 10
# stop a keyword once this many new papers collected; set NEW_PER_QUERY=0 (env)
# to disable the cap and scan the whole window for every keyword
_new = os.environ.get("NEW_PER_QUERY", "10")
NEW_PER_QUERY = int(_new) if _new.strip().isdigit() else 10
MAX_TOTAL_PER_QUERY = int(os.environ.get("MAX_TOTAL_PER_QUERY", "100"))
PAPER_TIMEOUT = 30
MAX_QUERIES = len(QUERIES)


def _fetch_with_timeout(doi, timeout=PAPER_TIMEOUT):
    box = {}

    def worker():
        try:
            box["content"] = fetch_html_from_doi(doi)
        except Exception as exc:  # noqa: BLE001
            box["error"] = str(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        box["timed_out"] = True
    return box


def discover():
    seen = load_seen_papers()
    print(f"seen papers so far: {len(seen)}")

    results = []
    used_keys = set()
    MAX_TOTAL_PER_QUERY = int(os.environ.get("MAX_TOTAL_PER_QUERY", "100"))
    new_cap = NEW_PER_QUERY if NEW_PER_QUERY > 0 else float("inf")
    for qi, query in enumerate(QUERIES[:MAX_QUERIES]):
        print(f"\n=== query {qi + 1}: {query[:70]}")
        start = 0
        scanned = 0
        new_count = 0
        while start < 200 and scanned < MAX_TOTAL_PER_QUERY and new_count < new_cap:
            page = search_papers(query, max_results=MAX_PER_QUERY, retstart=start)
            if not page:
                break
            fresh = [p for p in page if (p.get("pmid") or p.get("doi")) not in seen]
            scanned += len(page)
            print(f"  batch@{start}: {len(page)} papers, {len(fresh)} new "
                  f"(scanned {scanned}/{MAX_TOTAL_PER_QUERY}, new {new_count}/{NEW_PER_QUERY})")
            for paper in fresh:
                paper_id = paper.get("pmid") or paper.get("doi")
                if paper_id and paper_id in used_keys:
                    continue
                new_count += 1
                print(f"  [{paper['title'][:60]}] ...")
                html = None
                if paper.get("doi"):
                    box = _fetch_with_timeout(paper["doi"])
                    if box.get("timed_out"):
                        print(f"    !! timeout, using abstract")
                    html = box.get("content")
                if not html:
                    html = paper.get("abstract", "")
                if not html or len(html) < 50:
                    if paper_id:
                        seen.add(paper_id)
                    print("    !! no usable text")
                    continue
                links = extract_github_links(html)
                if links:
                    results.append({
                        "title": paper["title"][:100],
                        "doi": paper.get("doi", ""),
                        "github_links": links,
                        "url": paper.get("url", ""),
                    })
                    used_keys.add(paper_id)
                    print(f"    FOUND {len(links)} github links")
                else:
                    print("    no github links")
                if paper_id:
                    seen.add(paper_id)
                time.sleep(0.5)
            if len(page) < MAX_PER_QUERY:
                break
            start += MAX_PER_QUERY
            time.sleep(0.5)
        # persist seen progress after each query
        with open("seen_papers.json", "w", encoding="utf-8") as f:
            json.dump(sorted(seen), f, ensure_ascii=False, indent=2)
        # if no new paper was found in the whole window, skip this keyword
        if new_count == 0:
            print(f"  !! no new papers in {scanned} scanned, skipping this keyword")

    with open("github_from_html.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nTOTAL: {len(results)} papers with GitHub links")
    return results


if __name__ == "__main__":
    discover()
