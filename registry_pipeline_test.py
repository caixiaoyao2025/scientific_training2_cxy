"""Registry pipeline test: schema -> render -> execute -> output.

For every tool (and for subcommand CLIs, every LEAF) in data/mcp_registry.yaml
this runs the SAME stages an LLM agent call goes through, and reports PASS/FAIL
per stage:

  schema  : validate_tool_schema (pollution + template-var consistency)
  render  : _render_command / _render_subcommand builds a well-formed argv
            from the function-schema inputs (using real, format-correct
            fixtures so `bqtools encode` renders input + --output, not a bare
            subcommand -- the old smoke sampled the BASE tool's inputs and got
            `_args: {}` / argv `['bqtools', 'encode']`, run #31946687579)
  execute : run_tool_spec in-process (auto-installs; real env required)
  output  : the leaf's DECLARED output contract -- file/directory produced at
            the output param's path, or non-empty stdout for stdout-only
            tools (a generic "exit 0 + any stdout/stderr" check would call a
            file-producing tool correct even when it wrote nothing)

`schema` and `render` are runnable anywhere. `execute`/`output` need the
tool's environment (venv_path from the GH execute step, or a system binary),
so on machines without the env they are reported as SKIP, not FAIL.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from tool_agent_test import (  # noqa: E402
    validate_tool_schema, to_function_schemas, _task_output_kind,
    _task_output_param,
)
from agent_connector.tool_runner import (  # noqa: E402
    run_tool_spec, _render_command, _render_subcommand,
)

TEMPLATE_VAR_NAME = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")

_FIXTURES: dict = {}


def _prepare_fixtures() -> dict:
    """Create the smoke fixtures: a real FASTA file and, if the bqtools binary
    is on PATH, a real BINSEQ file (encode the FASTA). Subcommand leaves that
    consume BINSEQ get the binseq path; encode gets the FASTA. Paths are always
    returned so render stays well-formed even when bqtools is absent (execute
    is then SKIP via _env_ready)."""
    tmp = Path(tempfile.gettempdir())
    fasta = tmp / "pipeline_sample.fa"
    fasta.write_text(
        ">seq1\nACGTACGT\n>seq2\nTTTTTT\n>seq3\nCCCGGG\n>seq4\nAAAAT\n>seq5\nGATAC\n",
        encoding="utf-8")
    binseq = tmp / "pipeline_sample.binseq"
    exe = shutil.which("bqtools")
    if exe and not binseq.exists():
        for attempt in (
            [exe, "encode", str(fasta), "--output", str(binseq)],
            [exe, "encode", str(fasta), "--format", "a", "--output", str(binseq)],
        ):
            try:
                r = subprocess.run(attempt, timeout=60, capture_output=True, text=True)
            except Exception:  # noqa: BLE001
                continue
            if r.returncode == 0 and binseq.exists() and binseq.stat().st_size > 0:
                break
    return {"fasta": str(fasta), "binseq": str(binseq)}


def _leaf_fixture(leaf: dict) -> dict:
    """Format-correct fixture values for one subcommand leaf.

    Each leaf gets the input it ACTUALLY consumes and the output it actually
    produces (encode: FASTA -> BINSEQ; decode: BINSEQ -> FASTA; cat/sample/
    revcomp: BINSEQ -> BINSEQ; info/split/pipe/verify: BINSEQ -> stdout). A
    generic `input=<fasta>, output=<tsv>` for every leaf would be a format
    mismatch and would fail the real tools (run #31946687579)."""
    name = (leaf.get("name") or "").rsplit("_", 1)[-1]
    outdir = Path(tempfile.gettempdir()) / "pipeline_out"
    outdir.mkdir(parents=True, exist_ok=True)
    base = {"encode": {"input": _FIXTURES["fasta"],
                       "output": str(outdir / "sample.binseq")},
            "decode": {"input": _FIXTURES["binseq"],
                       "output": str(outdir / "sample.fa")},
            "cat": {"input": _FIXTURES["binseq"],
                    "output": str(outdir / "cat.binseq")},
            "grep": {"input": _FIXTURES["binseq"],
                     "output": str(outdir / "grep.binseq"),
                     "reg": "ACGT"},
            "sample": {"input": _FIXTURES["binseq"],
                       "output": str(outdir / "sample.binseq"),
                       "fraction": "0.5"},
            "revcomp": {"input": _FIXTURES["binseq"],
                        "output": str(outdir / "revcomp.binseq")},
            "info": {"input": _FIXTURES["binseq"]},
            "verify": {"input": _FIXTURES["binseq"]},
            "split": {"input": _FIXTURES["binseq"]},
            "pipe": {"input": _FIXTURES["binseq"]}}
    return {k: v for k, v in base.get(name, {}).items() if v}


def _sample_inputs(tool: dict, leaf: dict | None = None) -> dict:
    """Pick a tiny deterministic value for each template variable so render
    is well-formed even before any LLM runs. For subcommand LEAVES, use the
    real format-correct fixture (input + output) instead of the base tool's
    inputs -- the base has only `subcommand`, which is exactly why the old
    smoke rendered `bqtools encode` with NO args (run #31946687579)."""
    vals = {
        "input_file": "/tmp/sample.fa",
        "sequence": "MKT",
        "num_samples": 1,
        "output_dir": "/tmp/out",
        "pdb_path": "/tmp/a.pdb",
        "xtc_path": "/tmp/a.xtc",
        "output": "/tmp/out.tsv",
    }
    spec = leaf or tool
    out = {}
    cmd = spec.get("command") or ""
    for var in TEMPLATE_VAR_NAME.findall(cmd):
        out[var] = vals.get(var, "x")
    if leaf is not None:
        out.update(_leaf_fixture(leaf))
    for name, meta in (spec.get("inputs") or {}).items():
        # the base `subcommand` selector is the DISPATCHER's job: leaves carry
        # a concrete `_active_subcommand`, so feeding `subcommand` into a leaf
        # call makes the validator reject it as unknown.
        if name == "subcommand":
            continue
        if name not in out and (meta or {}).get("required"):
            out[name] = vals.get(name, "x")
    return out


def _check_output(result: dict, spec: dict, args: dict) -> str:
    """Return '' if the run satisfies the spec's DECLARED output contract.

    File/directory outputs are validated at the output param's path (the
    `outputs[x].input` link); stdout-only tools need real stdout. A generic
    "exit 0 + any stdout/stderr" check would call a file-producing tool
    correct even when it wrote nothing to its declared output file."""
    if result.get("status") == "timeout":
        return f"output: TIMEOUT ({result.get('return_code')})"
    if result.get("return_code") != 0:
        return f"output: exit {result.get('return_code')} (stderr: {(result.get('stderr') or '')[:120]})"
    kind = _task_output_kind(spec)
    if kind in ("file", "directory"):
        param = _task_output_param(spec)
        path = (args or {}).get(param) or ""
        if path:
            if kind == "directory":
                ok = os.path.isdir(path) and bool(os.listdir(path))
            else:
                ok = os.path.isfile(path) and os.path.getsize(path) > 0
            if ok:
                return ""
        return f"output: exit 0 but declared {kind} output missing ({param}={path})"
    stdout = result.get("stdout") or ""
    if stdout.strip():
        return ""
    return "output: exit 0 but no output"


def _env_ready(tool: dict) -> bool:
    """True only if the tool's environment already exists here (no install).
    Never triggers a network install from the pipeline test."""
    install = tool.get("install") or {}
    venv = install.get("venv_path", "")
    if venv and os.path.isdir(venv):
        return True
    cmd = tool.get("command") or ""
    parts = cmd.split()
    if not parts:
        return False
    if parts[0] in ("python", "python3"):
        # module tool: only runnable inside its venv (the isdir check above
        # already returned True if the venv exists -- otherwise SKIP)
        return False
    return shutil.which(parts[0].split("/")[-1]) is not None


def _execute_stage(r: dict, spec: dict, tool: dict, args: dict) -> dict:
    if not _env_ready(tool):
        r["execute"] = "SKIP (env not available on this machine)"
        r["output"] = "SKIP"
        return r
    try:
        # subcommand leaves execute as THEMSELVES (the same object render
        # produced) -- a base spec would be rejected by the runner.
        # tight per-call cap: a tool that HANGS on an unreachable hub (bioemu)
        # must fail in ~2 min, not eat the 600s spec default (run #31941212195).
        result = run_tool_spec(spec, args, timeout_override=120)
        err = _check_output(result, spec, args)
        r["execute"] = "PASS" if not err else f"FAIL ({err})"
        r["output"] = "PASS" if not err else f"FAIL ({err})"
        r["_argv"] = result.get("argv")
        r["_return_code"] = result.get("return_code")
    except FileNotFoundError:
        r["execute"] = "SKIP (env not available on this machine)"
        r["output"] = "SKIP"
    except Exception as e:  # noqa: BLE001
        r["execute"] = f"FAIL ({e})"
        r["output"] = "FAIL"
    return r


def pipeline(tool: dict) -> list[dict]:
    """Run all stages for a tool. Subcommand tools produce ONE report per leaf,
    so each subcommand's input fixture + output contract is exercised (P1)."""
    name = tool.get("name", "?")
    base = {"name": name, "schema": "SKIP", "render": "SKIP",
            "execute": "SKIP", "output": "SKIP"}
    schema_err = validate_tool_schema(tool)
    if schema_err:
        base["schema"] = f"FAIL ({schema_err})"
        return [base]
    base["schema"] = "PASS"
    try:
        schemas, fnmap = to_function_schemas(tool)
    except Exception as e:  # noqa: BLE001
        base["render"] = f"FAIL ({e})"
        return [base]

    if tool.get("arg_style") == "subcommand" and tool.get("subcommand_details"):
        leaves = list(fnmap.values()) or [tool]
        reports = []
        for leaf in leaves:
            r = dict(base)
            r["name"] = leaf.get("name", name)
            r["_function"] = r["name"]
            try:
                args = _sample_inputs(tool, leaf)
                argv = _render_subcommand(leaf, args)
                r["render"] = "PASS"
                r["_argv"] = argv
                r["_args"] = args
            except Exception as e:  # noqa: BLE001
                r["render"] = f"FAIL ({e})"
                reports.append(r)
                continue
            reports.append(_execute_stage(r, leaf, tool, args))
        return reports

    # ---- non-subcommand tool ----
    try:
        args = _sample_inputs(tool)
        argv = _render_command(tool.get("command") or "", args)
        base["render"] = "PASS"
        base["_argv"] = argv
        base["_args"] = args
        base["_functions"] = [s["function"]["name"] for s in schemas]
    except Exception as e:  # noqa: BLE001
        base["render"] = f"FAIL ({e})"
        return [base]
    return [_execute_stage(base, tool, tool, args)]


def main() -> int:
    reg_path = os.environ.get("REGISTRY", str(HERE / "data" / "mcp_registry.yaml"))
    with open(reg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tools = [t for t in data.get("tools", []) if isinstance(t, dict) and t.get("name")]
    _FIXTURES.update(_prepare_fixtures())

    results = []
    for t in tools:
        results.extend(pipeline(t))
    for r in results:
        line = (f"{r['name']:24} schema={r['schema']:6} render={r['render']:6} "
                f"execute={r['execute']} output={r['output']}")
        print(line)
    n_pass = sum(1 for r in results if r["schema"] == "PASS" and r["render"] == "PASS")
    n_fail = sum(1 for r in results if r["schema"].startswith("FAIL")
                 or r["render"].startswith("FAIL"))
    print(f"\nsummary: {n_pass}/{len(results)} schema+render pass, {n_fail} schema/render fail")

    if os.environ.get("PIPELINE_JSON"):
        with open(os.environ["PIPELINE_JSON"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
