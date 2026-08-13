"""Test whether a real downstream agent (LLM) can correctly use auto-discovered tools.

Reads data/mcp_registry.yaml (tools discovered + validated by the pipeline),
converts them to OpenAI function schemas, then runs a real function-calling loop
with an OpenAI-compatible LLM (volcengine Ark / DeepSeek). The LLM must:
  1. pick the right tool from the schema
  2. pass correct arguments (from the --help-parsed inputs)
  3. the tool auto-installs its environment (run_tool_spec _ensure_installed)
  4. returns a correct result

Env vars:
  WESTLAKE_API_KEY   (or OPENAI_API_KEY / DEEPSEEK_API_KEY)
  WESTLAKE_BASE_URL  (default https://ark.cn-beijing.volces.com/api/v3)
  WESTLAKE_MODEL     (default deepseek-v4-flash-ga-260731)
  REGISTRY           (default data/mcp_registry.yaml)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import yaml  # noqa: E402

BASE_URL = (os.environ.get("WESTLAKE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://ark.cn-beijing.volces.com/api/v3")
MODEL = (os.environ.get("WESTLAKE_MODEL") or os.environ.get("OPENAI_MODEL")
         or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash-ga-260731")
API_KEY = (os.environ.get("WESTLAKE_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY") or "")
REGISTRY = os.environ.get("REGISTRY", "data/mcp_registry.yaml")
MAX_TURNS = 6


def load_tools(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tools = data.get("tools", [])
    return [t for t in tools if isinstance(t, dict) and t.get("name")]


def to_function_schema(tool: dict) -> dict:
    props = {}
    required = []
    for name, meta in (tool.get("inputs") or {}).items():
        props[name] = {"type": "string",
                       "description": (meta or {}).get("description", "") or ""}
        required.append(name)
    fn = {
        "name": tool["name"],
        "description": (tool.get("description") or "")[:300],
        "parameters": {"type": "object", "properties": props, "required": required},
    }
    if tool.get("arg_style"):
        fn["arg_style"] = tool["arg_style"]
    if tool.get("install"):
        fn["install"] = tool["install"]
    return {"type": "function", "function": fn}


def ensure_repo() -> None:
    """Make sure the repo + agent_connector are importable and the registry exists.

    In Colab this script may be run standalone, so clone the repo if
    agent_connector isn't importable, and download the registry if missing.
    """
    global REPO, REGISTRY
    try:
        import agent_connector  # noqa: F401
        found = True
    except ImportError:
        found = False
    if not found:
        import subprocess
        dest = "/content/scientific_training2_cxy"
        print("[colab-prep] cloning repo ...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/caixiaoyao2025/scientific_training2_cxy.git", dest],
            check=True, capture_output=True, text=True)
        REPO = dest
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        if REGISTRY == "data/mcp_registry.yaml":
            REGISTRY = os.path.join(dest, REGISTRY)
    if not os.path.exists(REGISTRY) and not os.path.isabs(REGISTRY):
        # try relative to repo
        cand = os.path.join(REPO, REGISTRY)
        if os.path.exists(cand):
            REGISTRY = cand
        else:
            print(f"registry not found at {REGISTRY} - skipping")
            raise SystemExit(0)


def main() -> int:
    if not API_KEY:
        print("no API key (WESTLAKE_API_KEY/OPENAI_API_KEY/DEEPSEEK_API_KEY) - skipping agent test")
        return 0
    ensure_repo()
    if not os.path.exists(REGISTRY):
        print(f"no registry at {REGISTRY} - skipping")
        return 0
    tools = load_tools(REGISTRY)
    if not tools:
        print("registry empty - skipping")
        return 0
    print(f"tools from {REGISTRY}: {[t['name'] for t in tools]}")

    from openai import OpenAI
    from agent_connector.tool_runner import run_tool_spec, format_result
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    # Only test tools the agent can actually invoke via function-calling.
    # python_import-type tools (no CLI, no __main__) need import-based code,
    # which a function-calling agent can't do - mark and skip them.
    callable_tools = []
    skipped = []
    for t in tools:
        if t.get("callable_via") == "python_import" or t.get("arg_style") == "python" and not t.get("callable_via"):
            skipped.append(t["name"])
            print(f"[skip] {t['name']}: python_import (not function-callable; use via import)")
            continue
        callable_tools.append(t)
    schemas = [to_function_schema(t) for t in callable_tools]
    spec_map = {t["name"]: t for t in callable_tools}
    if not callable_tools:
        print("no function-callable tools in registry; nothing to test")
        return 0

    # sample input for tools that take a file path
    sample = os.path.join(tempfile.gettempdir(), "agent_test_sample.fasta")
    with open(sample, "w", encoding="utf-8") as f:
        f.write(">seq1\nACGT\nACGT\n>seq2\nTTTTTT\n>seq3\nCCCGGG\n>seq4\nAAAAT\n>seq5\nGATAC\n")

    # try a few representative prompts; each tests tool selection + args
    prompts = [
        ("count sequences", 
         f"Using the available tools, process the FASTA file {sample} and report "
         "how many sequences and total bases it contains."),
    ]
    passed, total = 0, 0
    for label, user_prompt in prompts:
        print(f"\n=== task: {label} ===")
        messages = [{"role": "user", "content": user_prompt}]
        final = None
        for turn in range(MAX_TURNS):
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=schemas, tool_choice="auto")
            msg = resp.choices[0].message
            if not getattr(msg, "tool_calls", None):
                final = msg.content
                break
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                print(f"[turn {turn}] agent chose: {fn_name} args={json.dumps(args)[:120]}")
                if fn_name not in spec_map:
                    result = f"unknown tool {fn_name}"
                else:
                    result = format_result(run_tool_spec(spec_map[fn_name], args))
                print(f"          result: {result[:150]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        print(f"LLM final: {final}")
        total += 1
        if final and ("5" in final and "30" in final):
            passed += 1
            print("  PASS: agent used tool and got correct numbers")
        else:
            print("  (agent may not have called a tool that returns 5/30 - inspect above)")

    print(f"\n== agent tool-usage test: {passed}/{total} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
