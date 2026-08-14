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
import re
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


# tokens that indicate a broken/mis-parsed schema. `{{var}}` template
# variables are LEGAL (command placeholders) -- we strip them before checking,
# and only flag real pollution: ANSI escapes, argparse choices `{a,b}` used as
# a name, `[OPTIONS]/[ARGS]` usage text, ellipsis pseudo-tokens.
_TEMPLATE_VAR = re.compile(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}")
# capture the variable NAME so we can cross-check it against the input schema
_TEMPLATE_VAR_NAME = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")
_POLLUTION = re.compile(r"\x1b\[|\.\.\.|\[OPTIONS\]|\[ARGS\]|\bCOMMAND\b|^\.$")
# argparse choices leaked as a parameter key: `{init,check,...}` (not {{var}})
_CHOICES_KEY = re.compile(r"\{[a-zA-Z0-9_,\-\s]+\}")


def _clean_template(command: str) -> str:
    """Replace legal {{var}} placeholders so they don't look like pollution."""
    return _TEMPLATE_VAR.sub("PLACEHOLDER", command)


def validate_tool_schema(tool: dict) -> str:
    """Return '' if the tool's schema is clean enough to give an LLM, else a
    reason (SCHEMA_INVALID / NO_CMD / SUBCOMMAND_INCOMPLETE / R_PACKAGE)."""
    name = tool.get("name", "")
    cmd = tool.get("command") or ""
    as_ = tool.get("arg_style") or "cli"
    inputs = tool.get("inputs") or {}
    # broken command template: check after masking legal {{var}} placeholders
    if _POLLUTION.search(_clean_template(cmd)):
        return f"SCHEMA_INVALID: command polluted ({cmd[:40]})"
    # broken input names: {{var}} is never a legal input KEY; choices `{a,b}`
    # as a key is pollution from argparse subparsers
    for k in inputs.keys():
        if _POLLUTION.search(k) or _TEMPLATE_VAR.search(k) or _CHOICES_KEY.search(k):
            return f"SCHEMA_INVALID: input name polluted ({k!r})"
    if not cmd.strip():
        return "NO_CMD"
    # template variable <-> input schema consistency: every {{var}} in the
    # command must have a matching input, or the rendered argv is undefined.
    vars_used = _TEMPLATE_VAR_NAME.findall(cmd)
    input_names = set(inputs.keys())
    unknown = [v for v in vars_used if v not in input_names]
    if unknown:
        return f"SCHEMA_INVALID: unknown template variable {sorted(set(unknown))} (command {cmd[:60]})"
    # duplicate template variable: e.g. `kaptain {{ONT_IN}} {{ONT_IN}} ...`.
    # repeated flags on the CLI are usually a parse artifact, not intent.
    dupes = sorted({v for v in vars_used if vars_used.count(v) > 1})
    if dupes:
        return f"SCHEMA_INVALID: duplicate template variable {dupes} (command {cmd[:60]})"
    # R package / python_import without -m: not directly callable as a command
    if as_ == "python" and not (tool.get("callable_via") or "").startswith("python -m "):
        return "R_PACKAGE_OR_IMPORT"
    if as_ == "subcommand" and not tool.get("subcommand_discovery_complete"):
        return "SUBCOMMAND_INCOMPLETE"
    return ""


def to_function_schemas(tool: dict) -> tuple[list[dict], dict]:
    """Build OpenAI function schema(s) for a tool.

    For subcommand CLIs we expand each subcommand into its OWN function so the
    agent picks `bqtools_encode(input, output)` instead of guessing a
    `subcommand` argument. Returns (schemas, fnmap) where fnmap maps the
    function name -> (tool_name, subcommand) for execution dispatch.
    """
    fnmap: dict = {}
    if (tool.get("arg_style") == "subcommand") and tool.get("subcommand_details"):
        out = []
        for sub, detail in (tool.get("subcommand_details") or {}).items():
            props = {}
            required = []
            for p in (detail.get("params") or []):
                key = p.get("name", "").lstrip("-").replace("-", "_").lower()
                props[key] = {"type": "string",
                              "description": (p.get("description") or "") or f"Argument {p.get('name')}"}
                # only truly-required params go in `required` (not everything)
                if p.get("required"):
                    required.append(key)
            fname = f"{tool['name']}_{sub.replace('-', '_')}"
            out.append({"type": "function", "function": {
                "name": fname,
                "description": (tool.get("description") or "") + f" -- subcommand {sub}",
                "parameters": {"type": "object", "properties": props, "required": required},
            }})
            fnmap[fname] = (tool["name"], sub)
        return out, fnmap
    # non-subcommand: single function
    props = {}
    required = []
    for name, meta in (tool.get("inputs") or {}).items():
        props[name] = {"type": "string",
                       "description": (meta or {}).get("description", "") or ""}
        # an input with NO `required` field is treated as required by default
        # (auto-discovery only marks the few genuinely-optional flags). An
        # explicit `required: false` stays optional.
        if (meta or {}).get("required") is not False:
            required.append(name)
    fn = {"type": "function", "function": {
        "name": tool["name"],
        "description": (tool.get("description") or "")[:300],
        "parameters": {"type": "object", "properties": props, "required": required},
    }}
    # execution infrastructure (install/venv/command) is NOT tool-call semantics
    # for the LLM; keep only arg_style as a hint.
    if tool.get("arg_style"):
        fn["function"]["arg_style"] = tool["arg_style"]
    fnmap[tool["name"]] = (tool["name"], "")
    return [fn], fnmap


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
    # Only test tools the agent can actually invoke via function-calling:
    # those with a real command (named/positional/subcommand CLI or `python -m`).
    # python_import / python-API tools without a -m entry can't be invoked by a
    # function-calling agent (it can't write import code) - mark and skip them.
    callable_tools = []
    skipped = []
    heavy = ("torch", "tensorflow", "torchvision", "torchaudio", "jax",
             "cupy", "paddle", "triton", "pytorch", "transformers", "diffusers")
    for t in tools:
        cv = t.get("callable_via") or ""
        as_ = t.get("arg_style") or "cli"
        # strict registry validation FIRST: polluted/ambiguous schemas must not
        # reach the LLM (they cause the retry-loop/timeout we saw).
        vres = validate_tool_schema(t)
        if vres:
            skipped.append(t["name"])
            print(f"[skip] {t['name']}: {vres}")
            continue
        # heavy ML deps: installing torch on-demand is too slow/unreliable
        decl = (t.get("install") or {}).get("declared_packages") or []
        if any(h in " ".join(decl).lower() for h in heavy):
            skipped.append(t["name"])
            print(f"[skip] {t['name']}: heavy ML deps (torch/tf) - needs preinstall")
            continue
        callable_tools.append(t)
    schemas = []
    fnmap = {}
    for t in callable_tools:
        sch, fm = to_function_schemas(t)
        schemas.extend(sch)
        fnmap.update(fm)
    spec_map = {t["name"]: t for t in callable_tools}
    if not callable_tools:
        print("no function-callable tools in registry; nothing to test")
        return 0

    # sample input for tools that take a file path
    sample = os.path.join(tempfile.gettempdir(), "agent_test_sample.fasta")
    with open(sample, "w", encoding="utf-8") as f:
        f.write(">seq1\nACGT\nACGT\n>seq2\nTTTTTT\n>seq3\nCCCGGG\n>seq4\nAAAAT\n>seq5\nGATAC\n")

    # --- per-tool concrete tasks with specified params + output validation ---
    # Each task tells the agent the tool, the input file, and a concrete output
    # path to produce. Success = tool exits 0 AND the output file exists. This
    # avoids "invoke it with valid args" (agent guessing) and catches the
    # earlier bqtools "encode but no output" false-positive.
    outdir = os.path.join(tempfile.gettempdir(), "agent_task_out")
    os.makedirs(outdir, exist_ok=True)
    tasks = []
    for t in callable_tools:
        name = t["name"]
        as_ = t.get("arg_style") or "cli"
        out = os.path.join(outdir, f"{name}_out")
        if as_ == "subcommand" and t.get("subcommands"):
            sub = t["subcommands"][0]
            # task for the first subcommand: input file + explicit output path
            label = f"{name}: {sub} on sample -> {os.path.basename(out)}"
            prompt = (f"Call {name}_{sub} (or the {name} tool's {sub} subcommand) to "
                      f"process the input file {sample}. Write the output to {out}. "
                      "Pass the arguments the tool's schema requires (input file and "
                      "output path). After running, the output file should exist.")
            tasks.append((label, prompt, sub, out))
        else:
            label = f"{name}: process sample -> {os.path.basename(out)}"
            prompt = (f"Call the {name} tool to process the input file {sample}. "
                      "Pass the arguments its schema requires. If it has an output "
                      "parameter, write to " + out + ". Aim for exit code 0.")
            tasks.append((label, prompt, "", out))

    print(f"\n== {len(tasks)} concrete per-tool tasks ==")
    stats = {"selected": 0, "started": 0, "succeeded": 0, "output_valid": 0}
    per_tool = {}
    for label, user_prompt, expect_sub, out_path in tasks:
        print(f"\n=== task: {label} ===")
        if os.path.exists(out_path):
            os.remove(out_path)
        messages = [{"role": "user", "content": user_prompt}]
        final = None
        selected = False
        started = False
        succeeded = False
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
                selected = True
                if fn_name not in fnmap:
                    result = f"unknown tool {fn_name}"
                else:
                    tool_name, sub = fnmap[fn_name]
                    if tool_name not in spec_map:
                        result = f"unknown tool {tool_name}"
                    else:
                        tool_spec = dict(spec_map[tool_name])
                        if sub:
                            tool_spec["_active_subcommand"] = sub
                        raw = run_tool_spec(tool_spec, args)
                        result = format_result(raw)
                        if raw.get("argv"):
                            print(f"          argv: {raw['argv']}")
                            print(f"          exit: {raw.get('return_code')}")
                    low = result.lower()
                    started = started or ("command not found" not in low and "exit code: 127" not in low)
                    if "exit code: 0" in low or "status: ok" in low:
                        succeeded = True
                print(f"          result: {result[:150]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        output_exists = succeeded and os.path.exists(out_path)
        output_size = os.path.getsize(out_path) if output_exists else 0
        # output is "valid" only if it's a real artifact: non-empty and not a
        # suspicious tiny/empty shell (e.g. an error message written to the
        # output path, or a 0/1-byte placeholder).
        output_valid = output_exists and output_size > 4
        output_suspicious = output_exists and output_size <= 4
        if succeeded and output_valid:
            status = "TASK_SUCCEEDED"
        elif succeeded and output_exists and output_suspicious:
            status = "OUTPUT_INVALID"
        elif succeeded:
            status = "EXECUTED_NO_OUTPUT"
        elif started:
            status = "TOOL_STARTED"
        elif selected:
            status = "TOOL_SELECTED"
        else:
            status = "NOT_SELECTED"
        per_tool[label.split(":")[0]] = {
            "selected": selected, "started": started, "succeeded": succeeded,
            "output_valid": output_valid, "output_size": output_size, "status": status,
        }
        if selected: stats["selected"] += 1
        if started: stats["started"] += 1
        if succeeded: stats["succeeded"] += 1
        if output_valid: stats["output_valid"] += 1
        print(f"  => {status}")

    print(f"\n== agent tool-usage summary ==")
    n = len(tasks)
    print(f"  tools tested:            {n}")
    print(f"  tool selected:           {stats['selected']}/{n}")
    print(f"  tool started (ran):      {stats['started']}/{n}")
    print(f"  tool exited 0:           {stats['succeeded']}/{n}")
    print(f"  output file valid:       {stats['output_valid']}/{n}")
    print("  (OUTPUT_INVALID/EXECUTED_NO_OUTPUT are NOT successes)")
    # honest: only output_valid is a real end-to-end success
    return 0


if __name__ == "__main__":
    sys.exit(main())
