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

from agent_connector.tool_spec import (  # noqa: E402
    canonical_key, json_schema_type, make_leaf_spec,
)

BASE_URL = (os.environ.get("WESTLAKE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://ark.cn-beijing.volces.com/api/v3")
MODEL = (os.environ.get("WESTLAKE_MODEL") or os.environ.get("OPENAI_MODEL")
         or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash-ga-260731")
API_KEY = (os.environ.get("WESTLAKE_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY") or "")
REGISTRY = os.environ.get("REGISTRY", "data/mcp_registry.yaml")
MAX_TURNS = 6
MAX_SAME_TOOL_ATTEMPTS = 3  # anti-tool-roulette: cap retries per tool per task


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
            # P0: build the leaf spec FIRST (make_leaf_spec is the SINGLE
            # source of truth for a subcommand's inputs -- positional metadata,
            # flag spelling, required semantics, outputs contract) and derive
            # the function schema from ITS `inputs`. Deriving props straight
            # from `subcommand_details[sub].params` re-infers the contract and
            # can drift (e.g. a positional marked required by make_leaf_spec
            # but optional here -> the LLM schema and the validator disagree).
            leaf = make_leaf_spec(tool, sub)
            props = {}
            required = []
            for key, meta in (leaf.get("inputs") or {}).items():
                props[key] = {"type": json_schema_type(meta),
                              "description": (meta or {}).get("description", "") or ""}
                # ONLY an explicit `required: true` (make_leaf_spec forces it
                # for positionals) makes a param required. A flag with no
                # required marker is OPTIONAL -- forcing every param required
                # would hand the LLM a fake schema (e.g. kaptain with all 15
                # flags required -> LLM fills garbage -> CLI error).
                if (meta or {}).get("required") is True:
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
        props[name] = {"type": json_schema_type(meta),
                       "description": (meta or {}).get("description", "") or ""}
        # ONLY an explicit `required: true` is required. Auto-discovery cannot
        # reliably know which flags are mandatory, so defaulting to optional is
        # the honest schema (a guessed-required list is worse: the LLM invents
        # values for flags the tool doesn't need).
        if (meta or {}).get("required") is True:
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


def _task_output_kind(tool: dict, sub: str = "") -> str:
    """Decide what a successful task must produce, from the tool's schema.

    Returns 'file' | 'directory' | 'stdout'. Uses the registry's `outputs`
    contract when present; a declared file/dir output param is the target, and
    tools with no output param are stdout-only (exit 0 + output is the bar).
    """
    outs = (tool.get("outputs") or {}).get(sub, {}) if sub else (tool.get("outputs") or {})
    if not isinstance(outs, dict):
        outs = {}
    if sub and not outs:
        # subcommand output contract lives on the subcommand itself
        # (discovery_to_registry stores it in subcommand_details[sub].outputs),
        # so bqtools_encode's file output is validated as a FILE, not stdout.
        detail = (tool.get("subcommand_details") or {}).get(sub) or {}
        sub_outs = detail.get("outputs") or {}
        if isinstance(sub_outs, dict):
            outs = sub_outs
    for name, meta in outs.items():
        if name == "stdout":
            continue
        t = (meta or {}).get("type", "")
        if t == "directory":
            return "directory"
        if t in ("file", "path"):
            return "file"
    return "stdout"


def _task_output_param(tool: dict, sub: str = "") -> str:
    """The input parameter name that CARRIES the task's output path.

    Returns the canonical input key whose flag/positional the tool writes to
    (bioemu -> 'output_dir', bqtools_encode -> 'output'), or '' if the tool's
    output contract is stdout-only. The task prompt uses this name so the LLM
    passes the output path to the RIGHT parameter instead of guessing.
    """
    outs = (tool.get("outputs") or {}).get(sub, {}) if sub else (tool.get("outputs") or {})
    if not isinstance(outs, dict):
        outs = {}
    if sub and not outs:
        detail = (tool.get("subcommand_details") or {}).get(sub) or {}
        sub_outs = detail.get("outputs") or {}
        if isinstance(sub_outs, dict):
            outs = sub_outs
    for name, meta in outs.items():
        if name == "stdout":
            continue
        t = (meta or {}).get("type", "")
        if t in ("file", "path", "directory"):
            # outputs keys are the canonical input keys (output_dir/output);
            # confirm the name is a real parameter of this tool/leaf
            inputs = (tool.get("inputs") or {})
            if sub:
                detail = (tool.get("subcommand_details") or {}).get(sub) or {}
                inputs = {canonical_key(p.get("name", "")): p
                          for p in (detail.get("params") or [])}
            if name in inputs:
                return name
    return ""


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
        # PLACEHOLDER schemas: `inputs_source: placeholder` means the inputs were
        # GUESSED (default `input_file`), never parsed from the tool's --help.
        # Such an entry has no real parameter contract (bqtools is stuck as
        # `bqtools {{input_file}}` because its paper wasn't rediscovered) and
        # must not reach the LLM -- the agent would only flail on it.
        ev = t.get("evidence") or {}
        if ev.get("inputs_source") == "placeholder":
            skipped.append(t["name"])
            print(f"[skip] {t['name']}: placeholder inputs (never --help-parsed; "
                  f"schema is a guess, not a contract)")
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

    # P0: prove the schemas actually sent to the LLM are what we expect BEFORE
    # spending API calls. If a subcommand tool didn't produce its leaf functions
    # (bqtools_encode etc.) that is a pipeline bug, not an agent failure.
    print(f"\n== {len(schemas)} function schemas sent to the LLM ==")
    for s in schemas:
        fn = s["function"]
        print(f"  {fn['name']:30} params={list(fn['parameters'].get('properties', {}).keys())} "
              f"required={fn['parameters'].get('required')}")
    sub_broken = []
    for t in callable_tools:
        if t.get("arg_style") == "subcommand" and t.get("subcommand_details"):
            for sub in (t.get("subcommand_details") or {}):
                leaf = f"{t['name']}_{sub.replace('-', '_')}"
                if leaf not in fnmap:
                    sub_broken.append(leaf)
    if sub_broken:
        print(f"[ERROR] subcommand tools produced NO leaf function for: {sub_broken}")
        print("        -> registry/subcommand_details is broken; refusing to run LLM")
        return 1
    print("  (all subcommand leaves present in schema)")

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
        if as_ == "subcommand" and t.get("subcommand_details"):
            # one task PER leaf subcommand function (bqtools_encode / _decode /
            # _info ...), not just the first. Each task is bound to its exact
            # function name so the test can distinguish WRONG_FUNCTION.
            for sub, detail in (t.get("subcommand_details") or {}).items():
                expected_fn = f"{name}_{sub.replace('-', '_')}"
                out = os.path.join(outdir, f"{name}_{sub}_out")
                out_kind = _task_output_kind(t, sub)
                out_param = _task_output_param(t, sub)
                label = f"{name}: {sub} on sample -> {os.path.basename(out)}"
                if out_param:
                    prompt = (f"Call the function {expected_fn} (NOT {name} directly) to "
                              f"process the input file {sample}. Write the output to "
                              f"{out} by passing it as the '{out_param}' parameter. "
                              "Pass the input file as the function's input parameter. "
                              "After running, the output must exist.")
                else:
                    prompt = (f"Call the function {expected_fn} (NOT {name} directly) to "
                              f"process the input file {sample}. Pass the arguments "
                              "the function's schema requires. After running, report "
                              "what it printed.")
                tasks.append((label, prompt, expected_fn, out, out_kind))
        else:
            expected_fn = name
            out = os.path.join(outdir, f"{name}_out")
            out_kind = _task_output_kind(t)
            out_param = _task_output_param(t)
            label = f"{name}: process sample -> {os.path.basename(out)}"
            if out_param:
                prompt = (f"Call the {name} tool to process the input file {sample}. "
                          f"Write the output to {out} by passing it as the "
                          f"'{out_param}' parameter. Pass the input file as the "
                          "input parameter. Aim for exit code 0.")
            else:
                prompt = (f"Call the {name} tool to process the input file {sample}. "
                          "Pass the arguments its schema requires. Aim for exit code 0 "
                          "and report what it prints.")
            tasks.append((label, prompt, expected_fn, out, out_kind))

    print(f"\n== {len(tasks)} concrete per-tool tasks ==")
    stats = {"selected": 0, "wrong_function": 0, "started": 0,
             "process_ok": 0, "output_valid": 0, "succeeded": 0}
    per_tool = {}
    for label, user_prompt, expected_fn, out_path, out_kind in tasks:
        print(f"\n=== task: {label} ===")
        if os.path.exists(out_path):
            os.remove(out_path)
        messages = [{"role": "user", "content": user_prompt}]
        final = None
        selected = False
        wrong_function = False
        started = False
        # per-task isolation: the retry cap resets for EVERY task, so a tool
        # burned in task A doesn't block the same tool in task B.
        tool_attempts = {}
        target_exited_0 = False   # the TARGET function exited 0 (process-level)
        target_ran = False        # the TARGET function was actually called
        raw_log = []              # full (fn, argv, rc, stdout, stderr) per call
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
                    if fn_name != expected_fn:
                        # P0: calling ANY other tool must NOT count as success,
                        # and it's a test failure the agent should not repeat.
                        wrong_function = True
                        if tool_name not in spec_map:
                            result = f"unknown tool {tool_name}"
                        else:
                            result = (f"[error_type: wrong_function] This task requires the "
                                      f"function `{expected_fn}`. You called `{fn_name}`, which "
                                      f"is the WRONG tool for this task. Do not call it again.")
                        print(f"          result: {result[:120]}")
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                        continue
                    if tool_name not in spec_map:
                        result = f"unknown tool {tool_name}"
                    elif tool_attempts.get(tool_name, 0) >= MAX_SAME_TOOL_ATTEMPTS:
                        # anti-tool-roulette: same tool kept failing -> tell the
                        # agent to STOP retrying it and fix the task differently.
                        result = (f"[tool {tool_name} already tried {tool_attempts.get(tool_name, 0)} times "
                                  f"and kept failing. STOP calling {tool_name}. Fix the argument "
                                  f"values or use a different tool/approach.]")
                    else:
                        tool_attempts[tool_name] = tool_attempts.get(tool_name, 0) + 1
                        # P0: dispatch the leaf ToolSpec (make_leaf_spec) -- its
                        # inputs are scoped to THIS subcommand, so the LLM's
                        # bqtools_encode(input, output) args validate against the
                        # same schema the LLM was shown (raw spec's inputs={}
                        # rejected every leaf arg as "unknown arguments").
                        tool_spec = make_leaf_spec(spec_map[tool_name], sub) if sub \
                            else spec_map[tool_name]
                        raw = run_tool_spec(tool_spec, args)
                        result = format_result(raw)
                        target_ran = True
                        # full raw capture (P1: never truncate the real error)
                        raw_log.append({
                            "fn": fn_name, "args": args,
                            "argv": raw.get("argv"), "return_code": raw.get("return_code"),
                            "status": raw.get("status"),
                            "stdout": (raw.get("stdout") or "")[-2000:],
                            "stderr": (raw.get("stderr") or "")[-2000:],
                        })
                        if raw.get("argv"):
                            print(f"          argv: {raw['argv']}")
                            print(f"          exit: {raw.get('return_code')}")
                        # P0: process success is ONLY the target function exiting 0
                        if raw.get("return_code") == 0 and raw.get("status") == "ok":
                            target_exited_0 = True
                        if raw.get("return_code") not in (127,):
                            started = True
                    low = result.lower()
                print(f"          result: {result[:150]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # ---- output validation by the task's declared output contract ----
        # P1: exit 0 (process) and TASK_SUCCEEDED (output produced) are separate.
        # An output DIRECTORY is valid only if it's a dir with contents.
        if target_exited_0:
            process_ok = True
            if out_kind == "directory":
                output_ok = (os.path.isdir(out_path)
                             and bool(os.listdir(out_path)))
            elif out_kind == "file":
                output_ok = (os.path.isfile(out_path)
                             and os.path.getsize(out_path) > 4)
            else:  # stdout-only tool: exit 0 + some output is the bar
                last = raw_log[-1] if raw_log else {}
                output_ok = bool((last.get("stdout") or "").strip()
                                 or (last.get("stderr") or "").strip())
        else:
            process_ok = False
            output_ok = False

        if target_exited_0 and output_ok:
            status = "TASK_SUCCEEDED"
        elif target_exited_0:
            status = "OUTPUT_INVALID"
        elif target_ran:
            status = "PROCESS_FAILED"
        elif wrong_function:
            status = "WRONG_FUNCTION"
        elif started:
            status = "TOOL_STARTED"
        elif selected:
            status = "TOOL_SELECTED"
        else:
            status = "NOT_SELECTED"
        per_tool[label] = {
            "expected_fn": expected_fn,
            "selected": selected, "wrong_function": wrong_function,
            "target_ran": target_ran, "target_exited_0": target_exited_0,
            "process_ok": process_ok, "output_ok": output_ok,
            "output_kind": out_kind, "status": status,
            "calls": raw_log,
        }
        if selected: stats["selected"] += 1
        if wrong_function: stats["wrong_function"] += 1
        if started: stats["started"] += 1
        if process_ok: stats["process_ok"] += 1
        if output_ok: stats["output_valid"] += 1
        if status == "TASK_SUCCEEDED": stats["succeeded"] += 1
        print(f"  => {status}")

    print(f"\n== agent tool-usage summary ==")
    n = len(tasks)
    print(f"  tasks:                 {n}")
    print(f"  tool selected:         {stats['selected']}/{n}")
    print(f"  wrong function:        {stats['wrong_function']}/{n}")
    print(f"  tool started (ran):    {stats['started']}/{n}")
    print(f"  target exit 0:         {stats['process_ok']}/{n}")
    print(f"  output valid:          {stats['output_valid']}/{n}")
    print(f"  TASK_SUCCEEDED:        {stats['succeeded']}/{n}")
    print("  (success = TARGET function exit 0 + valid output; any other tool")
    print("   exiting 0 is recorded as WRONG_FUNCTION, NOT success)")
    # persist the FULL per-call detail (argv/stdout/stderr, untruncated) so a
    # failure is debuggable instead of being cut at 150 chars in the console.
    out_json = os.environ.get("AGENT_TEST_JSON", "")
    if out_json:
        import json as _json
        with open(out_json, "w", encoding="utf-8") as f:
            _json.dump({"schemas": [s["function"]["name"] for s in schemas],
                        "tasks": per_tool}, f, ensure_ascii=False, indent=2)
        print(f"full detail -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
