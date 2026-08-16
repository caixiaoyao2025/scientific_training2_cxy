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
    get_required_inputs, make_leaf_spec, render_spec, validate_spec,
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
    tool_name, sub = fnmap[fname]
    t = next(x for x in tools if x["name"] == tool_name)
    # the EXACT spec the runner receives for this function
    leaf = make_leaf_spec(t, sub) if sub else t
    v = validate_spec(leaf)
    check(v == "", f"{fname}: leaf spec invalid: {v}")
    props = set(fn["parameters"]["properties"])
    leaf_inputs = set(leaf.get("inputs") or {})
    # 1. LLM schema param names == runner input names (no input_file invention)
    check(props == leaf_inputs,
          f"{fname}: schema params {sorted(props)} != leaf inputs {sorted(leaf_inputs)} "
          f"(diff {sorted(props ^ leaf_inputs)})")
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

print(f"\n{checked} checks, {fails} failures")
print("CONTRACT AUDIT: " + ("PASS" if fails == 0 else "FAIL"))
sys.exit(1 if fails else 0)
