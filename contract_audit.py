"""DEFINITIVE contract audit: for every function the agent sees, prove that
the LLM schema, the leaf ToolSpec the validator+runner use, and the argv
renderer all consume the SAME canonical input names.

This is the acceptance check for "no per-file re-derivation": if any layer
renames a parameter (input -> input_file, subcommand injected, etc.) this
fails loudly instead of drifting.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_connector.tool_spec import (
    get_required_inputs, render_spec, validate_spec,
)
from agent_connector.tool_runner import validate_arguments
from tool_agent_test import load_tools, to_function_schemas

REGISTRY = os.environ.get("REGISTRY", "data/mcp_registry.yaml")
fails = 0
checked = 0


def check(cond, msg):
    global fails, checked
    checked += 1
    if not cond:
        fails += 1
        print("  FAIL", msg)


tools = load_tools(REGISTRY)
print(f"registry: {REGISTRY} ({len(tools)} tools)")

# FULL set of function schemas the LLM is shown
schemas, fnmap = [], {}
for t in tools:
    sch, fm = to_function_schemas(t)
    schemas.extend(sch)
    fnmap.update(fm)

for s in schemas:
    fn = s["function"]
    fname = fn["name"]
    # the EXACT spec the runner receives for this function: fnmap now maps
    # fname -> the leaf ToolSpec itself (make_leaf_spec), so schema and runner
    # provably share one object -- never a base-tool re-parse.
    leaf = fnmap[fname]
    v = validate_spec(leaf)
    check(v == "", f"{fname}: leaf spec invalid: {v}")
    props = set(fn["parameters"]["properties"])
    leaf_inputs = set(leaf.get("inputs") or {})
    # 1. LLM schema param names == runner input names (no input_file invention)
    check(props == leaf_inputs,
          f"{fname}: schema params {sorted(props)} != leaf inputs {sorted(leaf_inputs)} "
          f"(diff {sorted(props ^ leaf_inputs)})")
    # 1b. the LLM schema is STRICT JSON Schema: no custom positional/flag/
    # position/outputs fields (strict function-calling providers reject or
    # drop them). CLI shape is runner metadata that lives ONLY in
    # ToolSpec.inputs; the schema carries it as a description hint instead.
    for key, meta in (leaf.get("inputs") or {}).items():
        prop = fn["parameters"]["properties"][key]
        check("positional" not in prop and "flag" not in prop and "position" not in prop,
              f"{fname}.{key}: CLI metadata leaked into JSON Schema: {sorted(set(prop) - {'type', 'description'})}")
    check("outputs" not in fn,
          f"{fname}: `outputs` leaked into the function object (strict providers reject it)")
    # 1c. leaf metadata is internally consistent: non-positionals carry a flag
    # (or a legal default spelling), positionals have integer positions.
    for key, meta in (leaf.get("inputs") or {}).items():
        meta = meta or {}
        if meta.get("positional"):
            pos = meta.get("position")
            check(isinstance(pos, int),
                  f"{fname}.{key}: positional without integer position: {pos!r}")
        else:
            flag = meta.get("flag")
            if flag:
                check(str(flag).startswith("-"),
                      f"{fname}.{key}: flag {flag!r} must start with '-'")
    # 1d. the description hint tells the LLM the CLI shape (flag / positional)
    for key, meta in (leaf.get("inputs") or {}).items():
        meta = meta or {}
        if meta.get("positional"):
            check("CLI positional" in (fn["parameters"]["properties"][key]["description"] or ""),
                  f"{fname}.{key}: positional hint missing from schema description")
        elif meta.get("flag"):
            check(f"CLI flag: {meta['flag']}" in (fn["parameters"]["properties"][key]["description"] or ""),
                  f"{fname}.{key}: flag hint missing from schema description")
    # 2. required set identical
    req_schema = set(fn["parameters"].get("required") or [])
    req_leaf = set(get_required_inputs(leaf))
    check(req_schema == req_leaf,
          f"{fname}: schema required {sorted(req_schema)} != leaf {sorted(req_leaf)}")
    # 3. type coercion: every int/float/bool is NOT collapsed to string
    for key, meta in (leaf.get("inputs") or {}).items():
        jt = fn["parameters"]["properties"][key]["type"]
        mtype = str((meta or {}).get("type", "")).lower()
        if mtype in ("integer", "int"):
            check(jt == "integer", f"{fname}.{key}: {mtype} -> schema {jt}")
        elif mtype in ("boolean", "bool"):
            check(jt == "boolean", f"{fname}.{key}: {mtype} -> schema {jt}")
    # 4. a minimal legal call validates AND renders (no subcommand leak)
    if req_leaf:
        args = {k: ("/tmp/sample.fasta" if (leaf["inputs"][k].get("type")
                                            in ("path", "file", "string"))
                    else (1 if leaf["inputs"][k].get("type") in ("int", "integer")
                          else (0.5 if leaf["inputs"][k].get("type") in ("float", "number")
                                else True)))
                for k in req_leaf}
        # required positionals must also render; flags render from flag field
        for k in req_leaf:
            if leaf["inputs"][k].get("positional"):
                args[k] = "/tmp/sample.fasta"
        # conditional required: any_of groups need at least one param each
        any_of = (leaf.get("constraints") or {}).get("any_of") or []
        for group in any_of:
            if group and not any(args.get(k) not in (None, "") for k in group):
                # satisfy with the first param in the group
                first = group[0]
                if first not in args:
                    args[first] = "/tmp/sample.fasta"
        cleaned, err = validate_arguments(leaf, args)
        check(err == "", f"{fname}: minimal required args rejected: {err}")
        if not err:
            try:
                argv = render_spec(leaf, cleaned)
                check(bool(argv) and argv[0] not in ("", None),
                      f"{fname}: rendered argv {argv}")
            except Exception as e:  # noqa: BLE001
                check(False, f"{fname}: render raised {e}")
    # 5. `subcommand` is NEVER an LLM-visible parameter
    check("subcommand" not in props,
          f"{fname}: subcommand leaked into LLM schema props!")
    # 6. output contract closure: every declared output file/dir must be a
    # declared input (so the runner can locate its path from the call args).
    for okey, om in (leaf.get("outputs") or {}).items():
        if okey == "stdout":
            continue
        check(okey in leaf_inputs,
              f"{fname}: declared output '{okey}' is not an input param "
              "(runner cannot locate the output path in the call args)")
    # 7. #55: the argv the RUNNER builds must equal the canonical render.
    # run_tool_spec reads the command from execution.command (falling back to
    # spec.command) and dispatches per exec_type, so audit the REAL dispatch
    # path -- render_spec alone would not catch a command/execution drift.
    from agent_connector.argv_renderer import (  # noqa: PLC0415
        render_command as _audit_rc, render_subcommand as _audit_rs,
    )
    execution = leaf.get("execution")
    if not isinstance(execution, dict) or not execution.get("type"):
        execution = {"type": leaf.get("type", "cli"), "command": leaf.get("command", "")}
    etype = execution.get("type", "cli")
    runner_argv = None
    if etype == "python" and execution.get("entry_point"):
        runner_argv = None  # python-import execution path has no argv
    elif etype == "api":
        runner_argv = None  # API path renders a URL, not argv
    elif leaf.get("arg_style") == "subcommand":
        runner_argv = _audit_rs(leaf, cleaned)
    else:
        runner_argv = _audit_rc(execution.get("command", ""), cleaned)
    if runner_argv is not None:
        canon_argv = render_spec(leaf, cleaned)
        check(runner_argv == canon_argv,
              f"{fname}: runner argv {runner_argv} != canonical render_spec "
              f"{canon_argv} (command/execution drift)")

print(f"\n{checked} checks, {fails} failures")
print("CONTRACT AUDIT: " + ("PASS" if fails == 0 else "FAIL"))
sys.exit(1 if fails else 0)
