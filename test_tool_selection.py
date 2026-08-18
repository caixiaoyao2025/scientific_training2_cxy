"""Tool Selection Test — Layer 1 of the three-layer agent pipeline.

Tests whether the LLM picks the CORRECT function given a user task and
all available tool schemas. This is distinct from tool_agent_test.py
which tests parameter filling (Layer 2/3: selection + invocation).

The selection test does NOT run any tools — it only checks the first
tool_call's function name against the expected function.

Two modes:
  1. retrieval_first: use tool_retrieval to filter candidates, then LLM
     picks from the filtered set (simulates the full pipeline)
  2. all_tools: LLM sees ALL schemas and must pick the right one
     (tests raw LLM selection without retrieval assistance)

Usage:
    python test_tool_selection.py              # retrieval_first mode
    python test_tool_selection.py --all-tools  # all_tools mode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from openai import OpenAI

import yaml

from tool_agent_test import to_function_schemas
from agent_connector.tool_retrieval import build_tool_index, retrieve_tools

MODEL = os.environ.get("WESTLAKE_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash-ga-260731"
MAX_RETRIES = 3
RETRY_DELAY = 5

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

if not API_KEY:
    raise RuntimeError(
        "Missing API key: set WESTLAKE_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY"
    )
if not BASE_URL.startswith(("http://", "https://")):
    raise RuntimeError(f"Invalid BASE_URL: {BASE_URL!r}")

# --- selection tasks: (user_task, expected_fn_name) ---
# Each task is a natural language request that should map to exactly one
# function. The expected_fn is the function the LLM should call FIRST.
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
    ("Predict RNA secondary structure from a sequence file", "bioemu"),
    ("Identify metagenomic species from ONT reads", "kaptain"),
]


def _load_schemas(registry_path: str):
    """Load registry and build function schemas for all tools."""
    with open(registry_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    tools = reg.get("tools", [])
    index = build_tool_index(tools)
    schemas = []
    fnmap = {}
    for t in tools:
        sch, fm = to_function_schemas(t)
        schemas.extend(sch)
        fnmap.update(fm)
    schema_by_name = {s["function"]["name"]: s for s in schemas}
    return tools, index, schemas, schema_by_name, fnmap


def _llm_select(task: str, schemas: list, retries: int = MAX_RETRIES):
    """Send task to LLM, return the first tool_call function name or None."""
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


def test_selection_retrieval_first():
    """Test: retrieval filters candidates, then LLM picks from filtered set."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    tools, index, all_schemas, schema_by_name, fnmap = _load_schemas(registry_path)
    print(f"== Selection test (retrieval_first): {len(all_schemas)} total schemas ==")
    passed = 0
    failed = 0
    skipped = 0
    for task, expected in SELECTION_TASKS:
        # retrieve top-k candidates
        candidates = retrieve_tools(task, index, top_k=8, min_score=2)
        if not candidates:
            print(f"[SKIP] {task[:50]}... -> no retrieval candidates")
            skipped += 1
            continue
        # build schemas for candidates only
        candidate_schemas = []
        for c in candidates:
            fn = c["fn_name"]
            if fn in schema_by_name:
                candidate_schemas.append(schema_by_name[fn])
        if not candidate_schemas:
            print(f"[SKIP] {task[:50]}... -> no schemas for candidates")
            skipped += 1
            continue
        # ask LLM to pick
        got = _llm_select(task, candidate_schemas)
        if got == expected:
            passed += 1
            print(f"[PASS] {task[:50]}... -> {got}")
        else:
            failed += 1
            cand_names = [c["fn_name"] for c in candidates]
            print(f"[FAIL] {task[:50]}...")
            print(f"       expected={expected}  got={got}")
            print(f"       candidates={cand_names}")
    print(f"\n== Results: {passed} passed, {failed} failed, {skipped} skipped "
          f"out of {len(SELECTION_TASKS)} ==")
    return failed


def test_selection_all_tools():
    """Test: LLM sees ALL tool schemas and must pick the right one."""
    registry_path = os.path.join(os.path.dirname(__file__), "data", "mcp_registry.yaml")
    tools, index, all_schemas, schema_by_name, fnmap = _load_schemas(registry_path)
    print(f"== Selection test (all_tools): {len(all_schemas)} total schemas ==")
    passed = 0
    failed = 0
    for task, expected in SELECTION_TASKS:
        got = _llm_select(task, all_schemas)
        if got == expected:
            passed += 1
            print(f"[PASS] {task[:50]}... -> {got}")
        else:
            failed += 1
            print(f"[FAIL] {task[:50]}...")
            print(f"       expected={expected}  got={got}")
    print(f"\n== Results: {passed} passed, {failed} failed "
          f"out of {len(SELECTION_TASKS)} ==")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Tool selection test")
    parser.add_argument("--all-tools", action="store_true",
                        help="Give LLM all schemas instead of retrieval-filtered")
    args = parser.parse_args()
    if args.all_tools:
        return test_selection_all_tools()
    return test_selection_retrieval_first()


if __name__ == "__main__":
    sys.exit(main())
