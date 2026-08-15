"""Hard preflight gate before any LLM agent test runs.

Verifies the registry contract end-to-end WITHOUT calling the LLM, so a broken
pipeline is caught in seconds, not after 20 minutes of agent flailing:

  gate 1  discovery  : every active tool has REAL schema evidence
                       (inputs were parsed from --help, not a placeholder guess)
  gate 2  subcommand : every subcommand tool has complete subcommand_details
                       (so to_function_schemas can emit leaf functions)
  gate 3  schema     : every tool's function schema is well-formed (no unknown
                       template vars, no polluted input names, leaves exist)
  gate 4  registry   : the ACTIVE registry contains no zombie placeholder
                       entries (authoritative-only; stale tools archived)
  gate 5  roundtrip  : every LLM function schema parameter is accepted by the
                       SAME leaf ToolSpec the runner receives (make_leaf_spec),
                       and renders to argv -- proving bqtools_encode(input,
                       output) never reaches the runner as "unknown arguments".

Any gate that fails prints EXACTLY where the chain broke and exits 1, so the
workflow stops BEFORE wasting API calls / 20 minutes.
"""
from __future__ import annotations

import os
import sys

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tool_agent_test import validate_tool_schema, to_function_schemas  # noqa: E402
from agent_connector.tool_spec import make_leaf_spec, validate_spec  # noqa: E402

REGISTRY = os.environ.get("REGISTRY", "data/mcp_registry.yaml")


def main() -> int:
    if not os.path.exists(REGISTRY):
        print(f"[PREFLIGHT FAIL] registry not found: {REGISTRY}")
        return 1
    with open(REGISTRY, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tools = [t for t in data.get("tools", []) if isinstance(t, dict) and t.get("name")]
    if not tools:
        print(f"[PREFLIGHT FAIL] registry {REGISTRY} has 0 tools -- nothing to test")
        return 1

    failures: list[str] = []
    n_ok = 0

    for t in tools:
        name = t["name"]
        # ---- gate 1: real schema evidence (not a placeholder guess) ----
        ev = t.get("evidence") or {}
        if ev.get("inputs_source") == "placeholder":
            failures.append(f"gate1 discovery: {name} has PLACEHOLDER inputs "
                            "(never --help-parsed) -- schema is a guess, not a contract")
            continue
        # gate 1b: argv completeness -- every positional the REAL --help
        # declared must be in the registry inputs. A schema can be internally
        # consistent (every {var} declared) yet semantically incomplete
        # (run #36 bioemu: only --filter_samples, the 3 required SEQUENCE/
        # NUM_SAMPLES/OUTPUT_DIR slots lost) -> preflight PASS -> LLM fails
        # at runtime with "no value for the required argument: sequence".
        if ev.get("exec_positional_args"):
            inp = t.get("inputs") or {}
            for pa in ev["exec_positional_args"]:
                pa_name = str(pa.get("name", "")).lstrip("-<>[]").replace("-", "_").lower()
                if pa_name and pa_name not in inp:
                    failures.append(f"gate1 argv-completeness: {name}: help declared "
                                    f"positional '{pa_name}' missing from registry inputs "
                                    "(schema incomplete; tool cannot be invoked correctly)")
        # ---- gate 2: subcommand tools must have complete details ----
        as_ = t.get("arg_style") or "cli"
        if as_ == "subcommand":
            if not t.get("subcommand_details"):
                failures.append(f"gate2 subcommand: {name} is arg_style=subcommand "
                                "but has empty subcommand_details")
                continue
            if not t.get("subcommand_discovery_complete"):
                failures.append(f"gate2 subcommand: {name} subcommand discovery INCOMPLETE")
                continue
        # ---- gate 3: schema well-formed + leaves exist ----
        vres = validate_tool_schema(t)
        if vres:
            failures.append(f"gate3 schema: {name}: {vres}")
            continue
        try:
            schemas, fnmap = to_function_schemas(t)
        except Exception as e:  # noqa: BLE001
            failures.append(f"gate3 schema: {name}: to_function_schemas raised {e}")
            continue
        if not schemas:
            failures.append(f"gate3 schema: {name}: produced 0 function schemas")
            continue
        fn_by_name = {s["function"]["name"]: s["function"] for s in schemas}
        if as_ == "subcommand":
            subs = t.get("subcommands") or list((t.get("subcommand_details") or {}).keys())
            missing = [f"{name}_{s.replace('-', '_')}" for s in subs
                       if f"{name}_{s.replace('-', '_')}" not in fnmap]
            if missing:
                failures.append(f"gate3 schema: {name} missing leaf functions: {missing}")
                continue
            # ---- gate 5: round-trip -- leaf LLM schema == leaf runner spec ----
            # The failure we are killing here: LLM sees bqtools_encode(input,
            # output) but the runner gets the RAW spec (inputs={}) and answers
            # "unknown arguments: input, output". Validate that the leaf spec
            # built by make_leaf_spec (which tool_agent_test now dispatches)
            # accepts EVERY parameter of the function schema the LLM was shown.
            for s in subs:
                fname = f"{name}_{s.replace('-', '_')}"
                leaf = make_leaf_spec(t, s)
                if validate_spec(leaf):
                    failures.append(f"gate5 roundtrip: {fname}: leaf spec invalid: "
                                    f"{validate_spec(leaf)}")
                    continue
                fn = fn_by_name[fname]
                # every schema property must be a runner-valid input of the leaf
                props = set(fn["parameters"]["properties"])
                leaf_inputs = set(leaf["inputs"])
                if not props.issubset(leaf_inputs):
                    failures.append(f"gate5 roundtrip: {fname}: function schema params "
                                    f"{sorted(props - leaf_inputs)} NOT in leaf runner "
                                    f"inputs (LLM would get 'unknown arguments')")
                    continue
                if not leaf_inputs.issubset(props):
                    failures.append(f"gate5 roundtrip: {fname}: leaf runner inputs "
                                    f"{sorted(leaf_inputs - props)} NOT exposed to LLM "
                                    f"(agent cannot pass them)")
                    continue
                # every required schema param must be required in the leaf spec,
                # so a legal call passes validate_arguments
                req = set(fn["parameters"].get("required") or [])
                leaf_req = {k for k, m in leaf["inputs"].items()
                            if (m or {}).get("required") is True}
                if req != leaf_req:
                    failures.append(f"gate5 roundtrip: {fname}: required mismatch "
                                    f"(schema {sorted(req)} vs leaf {sorted(leaf_req)})")
                    continue
                # positional metadata must survive into the leaf (bqtools encode
                # INPUT is a positional, not a flag)
                for p in (t.get("subcommand_details") or {}).get(s, {}).get("params") or []:
                    k = p.get("name", "").lstrip("-").replace("-", "_").lower()
                    want_pos = bool(p.get("positional"))
                    got_pos = bool((leaf["inputs"].get(k) or {}).get("positional"))
                    if want_pos != got_pos:
                        failures.append(f"gate5 roundtrip: {fname}: param '{k}' positional "
                                        f"mismatch (schema {want_pos} vs leaf {got_pos})")
                        break
        else:
            # non-subcommand: single function vs its own inputs
            fn = fn_by_name[name]
            props = set(fn["parameters"]["properties"])
            leaf_inputs = set((t.get("inputs") or {}))
            if not props.issubset(leaf_inputs):
                failures.append(f"gate5 roundtrip: {name}: function schema params "
                                f"{sorted(props - leaf_inputs)} NOT in tool inputs")
                continue
            req = set(fn["parameters"].get("required") or [])
            leaf_req = {k for k, m in (t.get("inputs") or {}).items()
                        if (m or {}).get("required") is True}
            if req != leaf_req:
                failures.append(f"gate5 roundtrip: {name}: required mismatch "
                                f"(schema {sorted(req)} vs tool {sorted(leaf_req)})")
                continue
        n_ok += 1

    # ---- gate 4: authoritative-only registry (no zombies) ----
    n_placeholder = sum(1 for t in tools
                        if (t.get("evidence") or {}).get("inputs_source") == "placeholder")
    if n_placeholder:
        failures.append(f"gate4 registry: {n_placeholder} placeholder entries are STILL in "
                        "the active registry (should have been archived by authoritative merge)")

    print(f"\n[preflight] {REGISTRY}: {len(tools)} tools, {n_ok} pass all gates")
    if failures:
        print("[PREFLIGHT FAIL] -- LLM test will NOT run. Broken at:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[PREFLIGHT PASS] -- all tools have real schema contracts; LLM test may run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
