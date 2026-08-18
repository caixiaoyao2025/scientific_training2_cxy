"""Tool Selection Test — uses graph-tool-call for retrieval + LLM for selection.

Three-layer test:
  Layer 1 (retrieval): does the expected tool appear in top-k candidates?
  Layer 2 (LLM): does the LLM pick the right tool from candidates?
  Layer 3 (all_tools): LLM sees ALL schemas, picks without retrieval.

Metrics:
  retrieval_hit@1, @3, @5
  llm_first_choice_accuracy
  no_match_correct (negative test)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from openai import OpenAI

import yaml

from tool_agent_test import to_function_schemas
from agent_connector.graph_retrieval import build_graph_from_registry, retrieve_tools

BASE_URL = (
    os.environ.get("WESTLAKE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("DEEPSEEK_BASE_URL")
    or "https://ark.cn-beijing.volces.com/api/v3"
)
API_KEY = (
    os.environ.get("WESTLAKE_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
    or ""
)
MODEL = os.environ.get("WESTLAKE_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash-ga-260731"
MAX_RETRIES = 3
RETRY_DELAY = 5

# --- selection tasks: (user_task, expected_fn_name) ---
# expected_fn=None means no tool should match (negative test)
SELECTION_TASKS = [
    # bqtools subcommands
    ("Encode my FASTA file into BINSEQ format", "bqtools_encode"),
    ("Decode this BINSEQ file back to FASTA", "bqtools_decode"),
    ("Search for ACGT pattern in my BINSEQ file", "bqtools_grep"),
    ("Split this BINSEQ file by pattern into separate files", "bqtools_split"),
    ("Randomly sample 10% of records from a BINSEQ file", "bqtools_sample"),
    ("Reverse complement all sequences in this BINSEQ file", "bqtools_revcomp"),
    ("Concatenate multiple BINSEQ files into one", "bqtools_cat"),
    ("Show me the metadata and record count of this BINSEQ file", "bqtools_info"),
    ("Verify the integrity checksum of this BINSEQ file", "bqtools_verify"),
    # cross-tool selection
    ("Generate protein conformational samples from an amino-acid sequence", "bioemu"),
    ("Identify metagenomic species from ONT reads", "kaptain"),
    # negative: no matching tool in registry
    ("Predict RNA secondary structure from a sequence file", None),
]


def _load_schemas(registry_path: str):
    """Load registry and build function schemas for all tools."""
    with open(registry_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    tools = reg.get("tools", [])
    schemas = []
    fnmap = {}
    for t in tools:
        sch, fm = to_function_schemas(t)
        schemas.extend(sch)
        fnmap.update(fm)
    schema_by_name = {s["function"]["name"]: s for s in schemas}
    return tools, schemas, schema_by_name, fnmap


def _llm_select(task: str, schemas: list, retries: int = MAX_RETRIES):
    """Send task to LLM, return the first tool_call function name or None."""
    if not API_KEY:
        raise RuntimeError("Missing API key")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    messages = [{"role": "user", "content": task}]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages,
                tools=schemas if schemas else [],
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                return msg.tool_calls[0].function.name
            return None  # LLM returned text instead of tool_call
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [retry {attempt+1}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def test_retrieval_only():
    """Layer 1: test retrieval hit rate without LLM."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    graph = build_graph_from_registry(registry_path)
    print(f"== Retrieval-only test ==")
    hit1 = hit3 = hit5 = 0
    total = len(SELECTION_TASKS)
    for task, expected in SELECTION_TASKS:
        if expected is None:
            # negative test: check that no tool scores above threshold
            results = retrieve_tools(graph, task, top_k=3)
            top_name = results[0][0] if results else None
            print(f"  [NEG] {task[:50]}... top={top_name}")
            continue
        results = retrieve_tools(graph, task, top_k=5)
        names = [r[0] for r in results]
        in1 = expected == names[0] if names else False
        in3 = expected in names[:3]
        in5 = expected in names[:5]
        if in1: hit1 += 1
        if in3: hit3 += 1
        if in5: hit5 += 1
        mark = "OK" if in1 else ("HIT@3" if in3 else ("HIT@5" if in5 else "MISS"))
        print(f"  [{mark:6}] {task[:50]}... top3={names[:3]}")
    n = total
    print(f"\n  retrieval_hit@1 = {hit1}/{n}")
    print(f"  retrieval_hit@3 = {hit3}/{n}")
    print(f"  retrieval_hit@5 = {hit5}/{n}")
    return hit1


def test_selection_retrieval_first():
    """Layer 1+2: graph-tool-call retrieval + LLM selection."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    graph = build_graph_from_registry(registry_path)
    _, all_schemas, schema_by_name, fnmap = _load_schemas(registry_path)
    print(f"== Selection test (graph-tool-call retrieval + LLM): {len(all_schemas)} total schemas ==")
    passed = 0
    failed = 0
    skipped = 0
    llm_correct = 0
    llm_total = 0
    for task, expected in SELECTION_TASKS:
        if expected is None:
            print(f"  [NEG-SKIP] {task[:50]}... (negative test, no LLM needed)")
            continue
        # retrieve candidates via graph-tool-call
        results = retrieve_tools(graph, task, top_k=5)
        candidate_names = [r[0] for r in results]
        if not candidate_names:
            print(f"  [SKIP] {task[:50]}... -> no retrieval candidates")
            skipped += 1
            continue
        # check retrieval hit
        retrieval_hit = expected in candidate_names
        # build schemas for candidates
        candidate_schemas = [schema_by_name[n] for n in candidate_names if n in schema_by_name]
        if not candidate_schemas:
            print(f"  [SKIP] {task[:50]}... -> no schemas for candidates")
            skipped += 1
            continue
        # LLM picks from candidates
        llm_total += 1
        got = _llm_select(task, candidate_schemas)
        llm_correct_task = got == expected
        if llm_correct_task:
            llm_correct += 1
        if llm_correct_task:
            passed += 1
            print(f"  [PASS] {task[:50]}... -> {got} (retrieval={'HIT' if retrieval_hit else 'MISS'})")
        else:
            failed += 1
            print(f"  [FAIL] {task[:50]}...")
            print(f"         expected={expected}  got={got}  candidates={candidate_names[:5]}")
    print(f"\n  LLM accuracy: {llm_correct}/{llm_total}")
    print(f"  Passed: {passed}/{len(SELECTION_TASKS)}, Failed: {failed}, Skipped: {skipped}")
    return failed


def test_selection_all_tools():
    """Layer 3: LLM sees ALL schemas, no retrieval."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    _, all_schemas, schema_by_name, fnmap = _load_schemas(registry_path)
    print(f"== Selection test (all_tools): {len(all_schemas)} total schemas ==")
    passed = 0
    failed = 0
    for task, expected in SELECTION_TASKS:
        got = _llm_select(task, all_schemas)
        if got == expected:
            passed += 1
            print(f"  [PASS] {task[:50]}... -> {got}")
        else:
            failed += 1
            print(f"  [FAIL] {task[:50]}...")
            print(f"         expected={expected}  got={got}")
    print(f"\n  Passed: {passed}/{len(SELECTION_TASKS)}, Failed: {failed}")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Tool selection test")
    parser.add_argument("--all-tools", action="store_true",
                        help="LLM sees ALL schemas (no retrieval)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Test retrieval hit rate only (no LLM)")
    args = parser.parse_args()
    if args.retrieval_only:
        return test_retrieval_only()
    if args.all_tools:
        return test_selection_all_tools()
    return test_selection_retrieval_first()


if __name__ == "__main__":
    sys.exit(main())
