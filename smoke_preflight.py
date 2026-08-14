"""Static smoke preflight: verify schema + argv BEFORE executing anything.

For each smoke tool this prints the final OpenAI function schema the LLM would
see, the fixture to use, and the exact argv the executor would build -- WITHOUT
running the tool. If any of these are wrong, we stop before wasting a run.

Usage:
    python smoke_preflight.py            # synthetic representative tools
    python smoke_preflight.py --registry # 4 real tools from the active registry
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

from tool_agent_test import to_function_schemas  # noqa: E402
from agent_connector.tool_runner import _render_command, _render_subcommand  # noqa: E402


def _fixture(tool_name: str, sub: str = "") -> dict:
    """Per-tool minimal legal fixture (NOT a uniform FASTA)."""
    tmp = tempfile.gettempdir()
    out = os.path.join(tmp, "smoke_out")
    os.makedirs(out, exist_ok=True)
    fa = os.path.join(tmp, "smoke_sample.fasta")
    with open(fa, "w") as f:
        f.write(">s1\nACGTACGT\n>s2\nTTTTGGGG\n")
    seq = os.path.join(tmp, "smoke_sample.txt")
    with open(seq, "w") as f:
        f.write("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR")
    fixtures = {
        # subcommand leaf: FASTA in, BINSEQ out
        ("bqtools", "encode"): {"input_file": fa, "output": os.path.join(out, "smoke.binseq")},
        ("bqtools", "decode"): {"input": os.path.join(out, "smoke.binseq"), "output": os.path.join(out, "smoke.fa")},
        ("bqtools", "info"): {"input_file": fa},
        # python CLI: protein sequence, optional pdb/xtc
        ("bioemu", ""): {"sequence": open(seq).read().strip(), "num_samples": 1,
                         "output_dir": os.path.join(out, "bioemu")},
        # a plain CLI with a file input
        ("samtools", ""): {"input_file": fa, "output": os.path.join(out, "out.sam")},
        # a python-API tool (module:function)
        ("gc_tool", ""): {"sequence": open(seq).read().strip()},
    }
    return fixtures.get((tool_name, sub), {"input_file": fa})


def check_tool(tool: dict, sub: str = "", fixture: dict | None = None) -> bool:
    """Static check: schema + argv for one tool/subcommand. Returns True if ok."""
    name = tool["name"]
    label = f"{name}_{sub}" if sub else name
    print(f"\n{'=' * 60}\nFUNCTION: {label}")
    try:
        schemas, fnmap = to_function_schemas(tool)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] to_function_schemas raised: {e}")
        return False
    # pick the schema for THIS subcommand (bqtools returns encode/decode/info)
    target = [s for s in schemas if s["function"]["name"] == label]
    if not target:
        print(f"  [FAIL] no schema for {label} (have: "
              f"{[s['function']['name'] for s in schemas]})")
        return False
    fn = target[0]["function"]
    props = fn["parameters"]["properties"]
    req = fn["parameters"]["required"]
    print(f"  description: {fn['description'][:100]}")
    for k, v in props.items():
        print(f"  {k}: type={v['type']:8} required={'YES' if k in req else 'no '} "
              f"desc={v.get('description','')[:60]}")
    if label not in fnmap:
        print(f"  [FAIL] {label} missing from fnmap")
        return False

    fixture = fixture or _fixture(name, sub)
    print(f"  FIXTURE: {json.dumps(fixture)}")
    try:
        if sub:
            spec = dict(tool, _active_subcommand=sub)
            argv = _render_subcommand(spec, fixture)
        else:
            argv = _render_command(tool.get("command") or "", fixture)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] render raised: {e}")
        return False
    print(f"  EXPECTED ARGV: {argv}")
    return True


def synthetic_tools() -> list[tuple[dict, str]]:
    """Representative tools covering the 4 execution styles, built from the
    same shapes execute_test/discovery_to_registry produce."""
    bqtools = {
        "name": "bqtools", "arg_style": "subcommand", "command": "bqtools {{subcommand}}",
        "description": "BINSEQ utilities", "inputs": {},
        "subcommands": ["encode", "decode", "info"],
        "subcommand_details": {
            "encode": {"params": [
                {"name": "input_file", "type": "path", "positional": True, "required": True,
                 "description": "Input FASTA file to encode"},
                {"name": "--output", "type": "path", "required": True,
                 "description": "Output BINSEQ file"}]},
            "decode": {"params": [
                {"name": "--input", "type": "path", "required": True,
                 "description": "Input BINSEQ file"},
                {"name": "--output", "type": "path", "required": True,
                 "description": "Output FASTA file"}]},
            "info": {"params": [
                {"name": "input_file", "type": "path", "positional": True, "required": True,
                 "description": "Input BINSEQ file"}]},
        },
        "subcommand_discovery_complete": True,
    }
    bioemu = {
        "name": "bioemu", "arg_style": "python", "command":
            "python -m bioemu.sample --sequence {{sequence}} --num_samples {{num_samples}} "
            "--output_dir {{output_dir}} --pdb-path {{pdb_path}} --xtc-path {{xtc_path}}",
        "description": "Protein ensemble emulator", "callable_via": "python -m bioemu.sample",
        "inputs": {
            "sequence": {"type": "string", "description": "Protein amino acid sequence", "required": True},
            "num_samples": {"type": "int", "description": "Number of samples", "required": False},
            "output_dir": {"type": "path", "description": "Output directory", "required": False},
            "pdb_path": {"type": "path", "description": "Optional PDB structure file", "required": False},
            "xtc_path": {"type": "path", "description": "Optional XTC trajectory file", "required": False},
        },
    }
    samtools = {
        "name": "samtools", "arg_style": "named", "command": "samtools sort {{input_file}} -o {{output}}",
        "description": "SAM/BAM tools", "inputs": {
            "input_file": {"type": "path", "description": "Input SAM/BAM file", "required": True},
            "output": {"type": "path", "description": "Output BAM file", "required": True},
        },
    }
    gc_tool = {
        "name": "gc_tool", "arg_style": "python",
        "command": "python -m gc_tool.calc --sequence {{sequence}}",
        "description": "GC content calculator", "callable_via": "python -m gc_tool.calc",
        "inputs": {"sequence": {"type": "string", "description": "DNA sequence", "required": True}},
    }
    return [(bqtools, "encode"), (bqtools, "decode"), (bqtools, "info"),
            (bioemu, ""), (samtools, ""), (gc_tool, "")]


def real_tools(registry: str) -> list[tuple[dict, str]]:
    with open(registry, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = []
    for t in data.get("tools", []):
        name = t.get("name", "")
        if name in ("cooltools", "rellig") and t.get("arg_style") == "subcommand":
            subs = list((t.get("subcommand_details") or {}).keys())[:2]
            for s in subs:
                out.append((t, s))
        elif name in ("bioemu",):
            out.append((t, ""))
    return out


def main() -> int:
    if "--registry" in sys.argv:
        registry = os.environ.get("REGISTRY", "data/mcp_registry.yaml")
        cases = real_tools(registry)
        print(f"[static smoke] using REAL tools from {registry}")
    else:
        cases = synthetic_tools()
        print("[static smoke] using SYNTHETIC representative tools "
              "(bqtools/bioemu/samtools/gc_tool)")
    ok = all(check_tool(t, sub) for t, sub in cases)
    print(f"\n{'=' * 60}\n[static smoke] {'PASS' if ok else 'FAIL'} -- "
          f"{len(cases)} tool schemas checked (no execution performed)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
