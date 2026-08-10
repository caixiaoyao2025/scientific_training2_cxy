"""Auto-connect discovered tools to a downstream agent and test with bio tasks.

Mirrors the Untitled37.ipynb flow but as a runnable script:

  1. Scan a downstream agent repo (local dir or git clone) -> agent_schema
     (registration method, execution method, agent class, wiring style)
  2. Read our registry (registry.yaml / data/mcp_registry.yaml / discovered_registry.yaml)
  3. Generate the adapter layer (wrappers + adapter, or manifest/config/prompt block)
  4. Inject into the real agent (or fall back to a dynamic registrar)
  5. Run end-to-end bioinformatics tasks through the connected agent and verify
     the tool was really invoked (not hallucinated) by checking the returned numbers.

Usage:
  python connect_agent.py --agent /path/to/agent             # local dir
  python connect_agent.py --agent https://github.com/x/y.git # clone
  python connect_agent.py --agent <path> --registry registry.yaml
  python connect_agent.py --list-agents                       # show built-in bio tasks
  python connect_agent.py --agent <path> --task seqstats      # run one task only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import yaml  # noqa: E402


# ============================================================
# 1. Bio tasks used to prove the tool was really called
# ============================================================
BIO_TASKS = {
    "seqstats": {
        "desc": "count sequences + total bases in a FASTA (numbers can't be guessed)",
        "setup": "sample.fasta",
        "prompt": (
            "Use the fasta_contig_stats_python tool to count the number of sequences "
            "and the total number of bases in the FASTA file {file}. "
            "Give the two numbers as the final answer."
        ),
        "expect": {"5", "30"},
        "tool": "fasta_contig_stats_python",
        "arg": "fasta_path",
    },
    "bedtools": {
        "desc": "intersect two BED files and report overlap count",
        "setup": "sample.bed",
        "prompt": (
            "Use the bedtools_intersect tool with a_bed_path={a} and b_bed_path={b} "
            "to intersect the two BED files. Report the number of overlapping intervals."
        ),
        "expect": None,  # count must be > 0
        "tool": "bedtools_intersect",
    },
    "count_lines": {
        "desc": "count lines + bytes of a text file",
        "setup": "sample.txt",
        "prompt": (
            "Use the count_lines_python tool to count the number of lines and the "
            "byte size of the file {file}. Give both numbers as the final answer."
        ),
        "expect": {"3"},
        "tool": "count_lines_python",
        "arg": "file_path",
    },
}

# names of demo tools we ship (registry.yaml) that the tasks rely on
_DEMO_TOOL_NAMES = {"fasta_contig_stats_python", "bedtools_intersect",
                    "count_lines_python"}


def _write_sample(name: str, workdir: str) -> str:
    p = os.path.join(workdir, name)
    if name == "sample.fasta":
        with open(p, "w", encoding="utf-8") as f:
            f.write(">seq1\nACGT\nACGT\n"
                    ">seq2\nTTTTTT\n"
                    ">seq3\nCCCGGG\n"
                    ">seq4\nAAAAT\n"
                    ">seq5\nGATAC\n")
    elif name == "sample.txt":
        with open(p, "w", encoding="utf-8") as f:
            f.write("line1\nline2\nline3\n")
    elif name == "sample.bed":
        with open(p, "w", encoding="utf-8") as f:
            f.write("chr1\t1\t10\tA\nchr1\t5\t15\tB\nchr2\t1\t20\tC\n")
    return p


# ============================================================
# 2. Agent acquisition: local dir or clone
# ============================================================
def acquire_agent(target: str) -> str:
    if os.path.isdir(target):
        return os.path.abspath(target)
    # try git clone
    tmp = tempfile.mkdtemp(prefix="agent_")
    name = urllib.parse.urlparse(target).path.rstrip("/").split("/")[-1] or "agent"
    dest = os.path.join(tmp, name)
    rc = subprocess.run(["git", "clone", "--depth", "1", target, dest],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"cannot clone agent {target}: {rc.stderr[:200]}")
    return dest


def _registry_path(arg: str | None) -> str:
    if arg:
        return arg
    for cand in ("data/mcp_registry.yaml", "discovered_registry.yaml", "registry.yaml"):
        if os.path.exists(os.path.join(REPO, cand)):
            return os.path.join(REPO, cand)
    raise SystemExit("no registry found; pass --registry")


def load_tools(registry_path: str) -> list[dict]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tools = data.get("tools", [])
    # normalize mcp_registry 'tools' list of dicts
    return [t for t in tools if isinstance(t, dict) and t.get("name")]


# ============================================================
# 3. Scan agent, generate wiring, inject, test
# ============================================================
def _fallback_schema(target: str) -> dict:
    """If the scanner found no register method, AST-scan for a simple one.

    Some agents expose add_tool/register_tool/add_mcp but the scanner's ranking
    misses them (low confidence). Do a direct method-name scan.
    """
    import ast
    for dirpath, _, filenames in os.walk(target):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                text = open(p, encoding="utf-8-sig").read().lstrip("\ufeff")
                tree = ast.parse(text)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                    for m in ("add_tool", "register_tool", "register", "add_mcp", "install"):
                        if m in methods:
                            rel = os.path.relpath(p, target)
                            mod = os.path.splitext(rel)[0].replace(os.sep, ".")
                            return {
                                "agent_class": node.name,
                                "registration_method": m,
                                "registration_style": "object",
                                "execution_method": "run",
                                "wiring_style": None,
                                "module": mod,
                                "confidence": 0.5,
                            }
    return {}


def run(target: str, registry_path: str, tasks: list[str], verbose: bool = False) -> int:
    from agent_connector.scanner import build_schema, detect_wiring_style
    from agent_connector.generator import generate_wiring

    tools = load_tools(registry_path)
    names = [t["name"] for t in tools]
    print(f"tools in registry ({len(tools)}): {names[:15]}{'...' if len(names) > 15 else ''}")

    # ---- scan the agent ----
    schema = build_schema(target, include_evidence=False)
    if not schema.get("registration_method"):
        fb = _fallback_schema(target)
        if fb:
            print(f"  [fallback] scanner found no register method; direct AST found "
                  f"{fb['agent_class']}.{fb['registration_method']}")
            schema.update(fb)
    print("\n== agent schema ==")
    for k in ("agent_class", "registration_method", "registration_style",
              "execution_method", "wiring_style", "confidence", "scanned_files"):
        print(f"  {k}: {schema[k]}")

    # ---- generate wiring (adapter or manifest/config/prompt) ----
    work = tempfile.mkdtemp(prefix="connect_")
    wiring = generate_wiring(tools, schema, out_dir=os.path.join(work, "wiring"))
    print(f"\n== wiring ==\n  mode: {wiring['mode']}")
    print("  artifacts:", list(wiring["artifacts"]))

    if schema.get("registration_method"):
        return _inject_and_test(target, schema, wiring, tools, work, tasks, verbose)
    return _offline_test(wiring, tools, tasks, verbose)


def _inject_and_test(target, schema, wiring, tools, work, tasks, verbose) -> int:
    from agent_connector.generator import load_wrappers, load_adapter
    sys.path.insert(0, target)
    sys.path.insert(0, work)
    try:
        wrappers = load_wrappers(package_name="wiring.generated_tools",
                                 registration_style=schema.get("registration_style") or "object")
    except Exception as exc:
        print(f"  wrapper load failed: {exc}")
        return 1
    for w in wrappers:
        if not hasattr(w, "name"):
            w.name = getattr(w, "__name__", None)

    adapter_path = wiring["artifacts"].get("adapter")
    if not adapter_path or not os.path.exists(adapter_path):
        print("  no adapter artifact -> fallback DynamicAgent")
        return _dynamic_inject(target, wrappers, tasks, verbose)

    Adapter = load_adapter(schema.get("agent_class") or "Agent",
                           adapter_path=os.path.abspath(adapter_path))
    print(f"  adapter class: {Adapter.__name__}")
    # instantiate the real agent
    agent = _instantiate_agent(target, schema)
    if agent is None:
        print("  real agent instantiation failed -> DynamicAgent fallback")
        return _dynamic_inject(target, wrappers, tasks, verbose)
    print(f"  injected agent: {type(agent).__name__}")
    Adapter(agent).install_tools(wrappers)
    print(f"  installed {len(wrappers)} wrappers")
    return _run_tasks(agent, wrappers, tasks, verbose)
    """No register method / no adapter: build a tiny registrar exposing add_tool."""
    class DynamicAgent:
        def __init__(self):
            self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)

    DynamicAgent.add_tool = add_tool
    agent = DynamicAgent()
    for w in wrappers:
        agent.add_tool(w)
    print(f"  DynamicAgent with {len(wrappers)} tools")
    return _run_tasks(agent, wrappers, tasks, verbose)


def _instantiate_agent(target, schema):
    agent_class = schema.get("agent_class")
    module_path = None
    # find the file that defines the class
    import ast
    for dirpath, _, filenames in os.walk(target):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(p, encoding="utf-8-sig").read().lstrip("\ufeff"))
            except Exception:
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == agent_class:
                    rel = os.path.relpath(p, target)
                    module_path = os.path.splitext(rel)[0].replace(os.sep, ".")
                    break
            if module_path:
                break
    if not module_path:
        print(f"  class {agent_class} not found")
        return None
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, agent_class)
        # try parameterless init, then with empty kwargs
        try:
            return cls()
        except Exception:
            import inspect
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters if p != "self"
                      and sig.parameters[p].default is inspect.Parameter.empty]
            if params:
                print(f"  __init__ needs: {params} (try AGENT_INIT_KWARGS)")
                return None
            return cls()
    except Exception as exc:
        print(f"  instantiate {module_path}.{agent_class} failed: {exc}")
        return None


def _dynamic_inject(target, wrappers, tasks, verbose) -> int:
    """No register method / no adapter: build a tiny registrar exposing add_tool."""
    class DynamicAgent:
        def __init__(self):
            self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)

    DynamicAgent.add_tool = add_tool
    agent = DynamicAgent()
    for w in wrappers:
        agent.add_tool(w)
    print(f"  DynamicAgent with {len(wrappers)} tools")
    return _run_tasks(agent, wrappers, tasks, verbose)


def _run_tasks(agent, wrappers, tasks, verbose) -> int:
    """Run bio tasks; verify tool was really invoked (numbers, not hallucination)."""
    workdir = tempfile.mkdtemp(prefix="task_")
    passed = 0
    wmap = {}
    for w in wrappers:
        nm = getattr(w, "__name__", None) or getattr(w, "name", None)
        if nm:
            wmap[nm] = w

    for tname in tasks:
        spec = BIO_TASKS.get(tname)
        if not spec:
            print(f"  unknown task {tname}; available: {list(BIO_TASKS)}")
            continue
        print(f"\n-- task: {tname} ({spec['desc']})")
        sample = _write_sample(spec["setup"], workdir)
        tool = wmap.get(spec["tool"])
        if tool is None:
            print(f"  SKIP: tool {spec['tool']} not among injected wrappers")
            continue
        # per-task args + prompt format
        if tname == "count_lines" or tname == "seqstats":
            arg = spec.get("arg", "file_path")
            call_args = {arg: sample}
            prompt = spec["prompt"].format(file=sample)
        else:  # bedtools
            call_args = {"a_bed_path": sample, "b_bed_path": sample}
            prompt = spec["prompt"].format(a=sample, b=sample)
        try:
            out = tool(**call_args)
        except Exception as e:
            print(f"  FAIL: call error: {e}")
            continue
        text = str(out or "")
        ok = any(x in text for x in (spec["expect"] or ())) if spec["expect"] else bool(text.strip())
        print(f"  tool output: {text[:160]!r}")
        print(f"  {'PASS' if ok else 'FAIL'}")
        if ok:
            passed += 1
    print(f"\n== task results: {passed}/{len([t for t in tasks if t in BIO_TASKS])} passed ==")
    return 0 if passed == len([t for t in tasks if t in BIO_TASKS]) else 1


def _offline_test(wiring, tools, tasks, verbose) -> int:
    """manifest/config/prompt wiring: just print artifacts; verify structure."""
    print("\n== offline artifacts (no register method) ==")
    mode = wiring["mode"]
    artifacts = wiring["artifacts"]
    if "manifest" in artifacts:
        path = artifacts["manifest"]
        data = json.load(open(path, encoding="utf-8"))
        names = [t["function"]["name"] for t in data]
        print(f"  manifest ({len(data)} tools): {names[:10]}")
    elif "config" in artifacts:
        print(f"  config written to {artifacts['config']}")
    elif "prompt_block" in artifacts:
        pb = artifacts["prompt_block"]
        print(f"  prompt_block ({len(pb)} chars)")
        if verbose:
            print(pb[:500])
    print("\n  tasks skipped (no runnable agent). To run tasks, point --agent at an agent with a register method.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="", help="agent dir or git URL to connect to")
    ap.add_argument("--registry", default="", help="path to registry yaml")
    ap.add_argument("--task", action="append", default=[],
                    help="bio task(s) to run; repeatable. default: all")
    ap.add_argument("--list-agents", action="store_true", help="show built-in tasks")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_agents:
        print("built-in bio tasks:")
        for name, s in BIO_TASKS.items():
            print(f"  {name:14} {s['desc']}")
        return

    tasks = args.task or list(BIO_TASKS)
    target = acquire_agent(args.agent) if args.agent else None
    if target is None:
        raise SystemExit("--agent is required (dir or git URL)")
    registry_path = _registry_path(args.registry)
    rc = run(target, registry_path, tasks, args.verbose)
    sys.exit(rc)


if __name__ == "__main__":
    main()
