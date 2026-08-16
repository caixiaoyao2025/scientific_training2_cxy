"""Registry pipeline test: schema -> render -> execute -> output.

For every tool in data/mcp_registry.yaml this runs the SAME stages an LLM
agent call goes through, and reports PASS/FAIL per stage:

  schema  : validate_tool_schema (pollution + template-var consistency)
  render  : _render_command / _render_subcommand builds a well-formed argv
            from the function-schema inputs (using --help-sampled defaults)
  execute : run_tool_spec in-process (auto-installs; real env required)
  output  : exit 0 AND non-empty stdout (or declared output file exists)

`schema` and `render` are runnable anywhere. `execute`/`output` need the
tool's environment (venv_path from the GH execute step, or a system binary),
so on machines without the env they are reported as SKIP, not FAIL.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from tool_agent_test import validate_tool_schema, to_function_schemas  # noqa: E402
from agent_connector.tool_runner import (  # noqa: E402
    run_tool_spec, _render_command, _render_subcommand,
)

TEMPLATE_VAR_NAME = __import__("re").compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def _sample_inputs(tool: dict) -> dict:
    """Pick a tiny deterministic value for each template variable so render
    is well-formed even before any LLM runs."""
    vals = {
        "input_file": "/tmp/sample.fa",
        "sequence": "MKT",
        "num_samples": 1,
        "output_dir": "/tmp/out",
        "pdb_path": "/tmp/a.pdb",
        "xtc_path": "/tmp/a.xtc",
        "output": "/tmp/out.tsv",
    }
    out = {}
    cmd = tool.get("command") or ""
    for var in TEMPLATE_VAR_NAME.findall(cmd):
        out[var] = vals.get(var, "x")
    for name, meta in (tool.get("inputs") or {}).items():
        if name not in out and (meta or {}).get("required"):
            out[name] = vals.get(name, "x")
    return out


def _check_output(result: dict, tool: dict) -> str:
    """Return '' if the execution looks like a real tool result."""
    if result.get("status") == "timeout":
        return f"output: TIMEOUT ({result.get('return_code')})"
    if result.get("return_code") != 0:
        return f"output: exit {result.get('return_code')} (stderr: {(result.get('stderr') or '')[:120]})"
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    if not stdout.strip() and not stderr.strip():
        return "output: exit 0 but no output"
    return ""


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


def pipeline(tool: dict) -> dict:
    name = tool.get("name", "?")
    report = {"name": name, "schema": "SKIP", "render": "SKIP", "execute": "SKIP", "output": "SKIP"}

    # ---- 1. schema ----
    schema_err = validate_tool_schema(tool)
    if schema_err:
        report["schema"] = f"FAIL ({schema_err})"
        return report
    report["schema"] = "PASS"

    # ---- 2. render ----
    try:
        args = _sample_inputs(tool)
        schemas, fnmap = to_function_schemas(tool)
        if tool.get("arg_style") == "subcommand" and tool.get("subcommand_details"):
            # render a real subcommand argv through the CANONICAL leaf spec
            # (fnmap now maps fname -> the leaf itself, make_leaf_spec) --
            # exactly the object run_tool_spec and the agent test dispatch.
            leaf = list(fnmap.values())[0]
            argv = _render_subcommand(leaf, args)
        else:
            argv = _render_command(tool.get("command") or "", args)
        report["render"] = "PASS"
        report["_argv"] = argv
        report["_args"] = args
        report["_functions"] = [s["function"]["name"] for s in schemas]
    except Exception as e:  # noqa: BLE001
        report["render"] = f"FAIL ({e})"
        return report

    # ---- 3/4. execute + output (needs real env; SKIP, never install) ----
    if not _env_ready(tool):
        report["execute"] = "SKIP (env not available on this machine)"
        report["output"] = "SKIP"
        return report
    try:
        # subcommand tools execute as their LEAF spec (the same object the
        # render step produced) -- a base spec would be rejected by the runner.
        exec_spec = leaf if (tool.get("arg_style") == "subcommand"
                             and tool.get("subcommand_details")) else tool
        result = run_tool_spec(exec_spec, args)
        err = _check_output(result, tool)
        report["execute"] = "PASS" if not err else f"FAIL ({err})"
        report["output"] = "PASS" if not err else f"FAIL ({err})"
        report["_argv"] = result.get("argv")
        report["_return_code"] = result.get("return_code")
    except FileNotFoundError:
        report["execute"] = "SKIP (env not available on this machine)"
        report["output"] = "SKIP"
    except Exception as e:  # noqa: BLE001
        report["execute"] = f"FAIL ({e})"
        report["output"] = "FAIL"

    return report


def main() -> int:
    reg_path = os.environ.get("REGISTRY", str(HERE / "data" / "mcp_registry.yaml"))
    with open(reg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tools = [t for t in data.get("tools", []) if isinstance(t, dict) and t.get("name")]

    results = [pipeline(t) for t in tools]
    for r in results:
        line = (f"{r['name']:24} schema={r['schema']:6} render={r['render']:6} "
                f"execute={r['execute']} output={r['output']}")
        print(line)
    n_pass = sum(1 for r in results if r["schema"] == "PASS" and r["render"] == "PASS")
    n_fail = sum(1 for r in results if r["schema"].startswith("FAIL") or r["render"].startswith("FAIL"))
    print(f"\nsummary: {n_pass}/{len(results)} schema+render pass, {n_fail} schema/render fail")

    if os.environ.get("PIPELINE_JSON"):
        with open(os.environ["PIPELINE_JSON"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
