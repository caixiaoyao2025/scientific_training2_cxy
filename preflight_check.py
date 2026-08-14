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
        if as_ == "subcommand":
            subs = t.get("subcommands") or list((t.get("subcommand_details") or {}).keys())
            missing = [f"{name}_{s.replace('-', '_')}" for s in subs
                       if f"{name}_{s.replace('-', '_')}" not in fnmap]
            if missing:
                failures.append(f"gate3 schema: {name} missing leaf functions: {missing}")
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
