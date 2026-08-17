"""Canonical ToolSpec / leaf-spec layer -- SINGLE source of truth.

Every consumer (generator wrappers, tool_agent_test, the runner, preflight,
smoke) derives its input schema and required-set from the SAME functions
here, so a subcommand leaf can never drift between "LLM schema says
input/output", "registry inputs says nothing", and "runner rejects unknown
arguments".

Central pieces:
  make_leaf_spec(tool, sub)  : build the leaf ToolSpec for a subcommand from
                               registry.subcommand_details (inputs scoped to
                               that sub, positional metadata preserved, its
                               OWN outputs contract).
  get_input_schema(spec)     : canonical {canonical_key -> {type, required,
                               positional, position, description}}.
  get_required_inputs(spec)  : ONLY explicit `required: true` (matching the
                               LLM function schema; a guessed-required flag is
                               a fake contract).
  is_required(meta)          : single required-semantics used by everyone.
  json_schema_type(meta)     : registry type -> OpenAI JSON-schema type
                               (integer/float/boolean passed through, not
                               collapsed to string).
  validate_spec(spec)        : registry-contract validation of a leaf.
  render_spec(spec, args)    : argv via _render_command/_render_subcommand,
                               both of which read the SAME canonical inputs.
"""
from __future__ import annotations

import copy
import re
from typing import Any

TEMPLATE_VAR_NAME = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def canonicalize_param_name(name: str) -> str:
    """THE single parameter-name canonicalizer for the whole pipeline.

    One key per CLI token: --ont-in / <ONT_IN> / ont_in -> 'ont_in'.
    Every layer (discovery -> registry -> generator -> LLM schema -> runner
    argv) uses THIS function, so a parameter can NEVER appear as `input`,
    `input_file` and `infile` in different stages. Mirrors the historical
    execute_test._canonical_param_name exactly."""
    s = str(name or "").strip().strip("<>[]{}")
    s = s.lstrip("-")
    if not s:
        return ""
    return s.lower().replace("-", "_")


# backward-compatible alias used by earlier layers
canonical_key = canonicalize_param_name


def is_required(meta: Any) -> bool:
    """ONLY an explicit `required: true` means required.

    Auto-discovery cannot reliably know which flags are mandatory, so a
    missing/None/false marker must mean OPTIONAL (defaulting to required
    hands the LLM a fake schema and forces `validate_arguments` to reject
    otherwise-legal calls)."""
    return bool(meta) and (meta.get("required") is True)


def _param_input(p: dict[str, Any]) -> dict[str, Any]:
    """Registry subcommand param -> canonical input meta (keeps every piece
    the runner + function schema need: type, required, positional order,
    takes_value so store-flags render bare, and the EXACT flag spelling so the
    argv renderer emits `--output` not a guessed name)."""
    meta: dict[str, Any] = {
        "type": p.get("type", "string"),
        "description": p.get("description") or f"Argument {p.get('name')}",
        "required": p.get("required") is True,
        "source": "help_parsed",
    }
    if p.get("positional"):
        meta["positional"] = True
        # required semantics from the PARSER, not forced: a bare metavar
        # (SEQUENCE, <INPUT>) is a mandatory argv slot, but a bracketed
        # `[INPUT]` / `[FILES ...]` is OPTIONAL (reads stdin / zero-or-more).
        # Forcing every positional required would demand an arg the CLI usage
        # itself marks optional.
        meta["required"] = p.get("required") is True
        if p.get("position") is not None:
            meta["position"] = p["position"]
    else:
        # flag spelling as the tool declared it (--format), used by the
        # renderer to build argv; canonical_key keeps the input key clean.
        meta["flag"] = p.get("name", "")
    if p.get("takes_value") is not None:
        meta["takes_value"] = p["takes_value"]
    if p.get("aliases"):
        meta["aliases"] = p["aliases"]
    if p.get("choices"):
        meta["choices"] = p["choices"]
    return meta


def make_leaf_spec(tool: dict[str, Any], sub: str) -> dict[str, Any]:
    """Canonical leaf ToolSpec for a subcommand-CLI leaf function.

    The leaf is a COMPLETE, self-contained ToolSpec -- NOT a base tool plus a
    `_active_subcommand` hint. Its `command` is the concrete invocation
    (`bqtools encode`), its `inputs` are scoped to THIS subcommand (positional
    metadata + flag spelling preserved), and it carries the sub's OWN outputs
    contract. Every consumer (generator wrappers, agent test, runner,
    preflight) runs on THIS spec; none of them re-derive a subcommand or
    re-read `subcommand_details` -- there is exactly one schema source.
    """
    details = (tool.get("subcommand_details") or {}).get(sub) or {}
    params = details.get("params") or []
    # DEEP copy: the leaf is a fresh, immutable contract. `dict(tool)` would
    # keep sharing subcommand_details/install/execution with the BASE registry
    # object, so a later `spec["execution"][...] = ...` would mutate another
    # tool's (or the base's) spec. The leaf must never alias the base.
    leaf = copy.deepcopy(tool)
    leaf["name"] = f"{tool.get('name', '')}_{sub.replace('-', '_')}"
    leaf["_active_subcommand"] = sub
    leaf["description"] = (tool.get("description") or "") + f" -- {sub}"
    # concrete command: the tool's command with {{subcommand}} replaced by the
    # chosen sub -- NEVER just the first token (split()[0] would turn
    # `python -m nano_signal_simulator {{subcommand}}` into `python simulate`).
    # Accept BOTH brace conventions: discovery emits Jinja-style {{subcommand}},
    # the merged MCP registry rewrites it to server-native {subcommand}
    # (merge_to_mcp.clean_tool_entry). Either form MUST become the concrete sub
    # -- a literal `{subcommand}` leaking into leaf argv is a hard error.
    # Base tools with a bare command (older registries) get the sub appended.
    cmd_tmpl = (tool.get("command") or "").strip().replace("{{subcommand}}", "{subcommand}")
    if "{subcommand}" in cmd_tmpl:
        leaf_cmd = cmd_tmpl.replace("{subcommand}", sub)
    elif cmd_tmpl:
        leaf_cmd = f"{cmd_tmpl} {sub}"
    else:
        leaf_cmd = f"{tool.get('name') or ''} {sub}"
    leaf["command"] = leaf_cmd.strip()
    if "{subcommand}" in leaf["command"] or "{{subcommand}}" in leaf["command"]:
        raise ValueError(
            f"leaf {leaf['name']} command still contains subcommand placeholder: "
            f"{leaf['command']!r}")
    # execution must carry the SAME concrete command: run_tool_spec reads
    # execution.command for the cli/docker fallbacks, and a stale template here
    # is exactly the command/execution drift that put `{subcommand}` back into
    # argv. Fresh dict so we never mutate the BASE object's execution.
    leaf["execution"] = dict(tool.get("execution") or {})
    leaf["execution"].setdefault("type", tool.get("type", "cli"))
    leaf["execution"]["command"] = leaf["command"]
    leaf["inputs"] = {canonical_key(p.get("name", "")): _param_input(p)
                      for p in params if canonical_key(p.get("name", ""))}
    leaf["outputs"] = details.get("outputs") or tool.get("outputs") or {}
    leaf["resources"] = tool.get("resources") or {}
    # per-subcommand constraints (e.g. split requires one of file/sfile/xfile)
    sub_constraints = details.get("constraints")
    if sub_constraints:
        leaf["constraints"] = sub_constraints
    return leaf


def get_input_schema(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("inputs") or {}


def get_required_inputs(spec: dict[str, Any]) -> list[str]:
    return sorted(k for k, m in get_input_schema(spec).items() if is_required(m))


def json_schema_type(meta: Any) -> str:
    """Registry input type -> OpenAI JSON-schema type.

    Auto-discovery records `integer`/`float`/`path` etc.; passing those
    through (instead of collapsing everything to `string`) lets the LLM emit
    real numbers (num_samples: 1) instead of strings. Unknown/`path`/`file`
    stay `string` (paths are strings in JSON)."""
    t = (meta or {}).get("type", "string") if isinstance(meta, dict) else "string"
    t = str(t).lower()
    if t in ("integer", "int"):
        return "integer"
    if t in ("float", "double", "number"):
        return "number"
    if t in ("boolean", "bool"):
        return "boolean"
    return "string"


def get_resources(spec: dict[str, Any]) -> dict[str, Any]:
    """Runtime resources (paths to pre-existing DBs/indexes) declared on the
    spec, keyed by canonical_key. These are NOT LLM-inventable: they are
    injected by the runner from the environment, never part of the function
    schema."""
    return spec.get("resources") or {}


def function_property(meta: Any) -> dict[str, Any]:
    """OpenAI function-schema property for one ToolSpec input.

    STRICT JSON Schema only: `type` + `description`. The CLI shape
    (positional slot / flag spelling) is runner metadata that lives EXCLUSIVELY
    in ToolSpec.inputs (and argv_renderer reads it from there) -- it is never
    emitted as a custom JSON-Schema field, because strict providers reject or
    drop non-standard properties, and the LLM should not be deciding how argv
    is rendered. The shape is surfaced as a plain-text hint in the description
    instead, so the model still knows `input` is an argv slot and `output`
    takes the `--output` flag."""
    prop: dict[str, Any] = {"type": json_schema_type(meta),
                            "description": (meta or {}).get("description", "") or ""}
    if (meta or {}).get("choices"):
        prop["enum"] = [str(x) for x in meta["choices"]]
    hint: list[str] = []
    if (meta or {}).get("positional"):
        hint.append("CLI positional argument")
        if (meta or {}).get("position") is not None:
            hint.append(f"position {meta['position']}")
    flag = (meta or {}).get("flag")
    if flag:
        hint.append(f"CLI flag: {flag}")
    if hint:
        prop["description"] = ((prop["description"] + " ").strip()
                                + f"[{', '.join(hint)}]").strip()
    return prop


def function_anyof(spec: dict[str, Any]) -> dict[str, list[list[str]]] | None:
    """Extract ``constraints.any_of`` from a leaf ToolSpec.

    The split subcommand requires at least one of --file/--sfile/--xfile.
    This is expressed as a conditional required (anyOf) that cannot be
    represented by individual param ``required: true`` fields. Returns
    e.g. ``{"any_of": [["file"], ["sfile"], ["xfile"]]}`` or None."""
    return (spec.get("constraints") or {}).get("any_of") or None


def validate_spec(spec: dict[str, Any]) -> str:
    """Contract validation of a (leaf) ToolSpec; '' means valid.

    Covers the same gates as tool_agent_test.validate_tool_schema plus the
    subcommand-leaf shape, so preflight can run it on the EXACT spec the
    runner receives. For subcommand leaves the `{{subcommand}}` placeholder is
    injected by the dispatcher (fnmap -> _active_subcommand), NOT an input.
    A command placeholder is satisfied by an `inputs` key OR a declared
    `resources` key (resources are injected by the runner, not LLM args).
    """
    inputs = get_input_schema(spec)
    if not isinstance(inputs, dict):
        return "input schema is not a dict"
    for k in inputs:
        if not k or k != k.strip() or " " in k or "\t" in k:
            return f"input name polluted: {k!r}"
        t = (inputs[k] or {}).get("type", "string")
        if t not in ("string", "str", "int", "integer", "float", "number",
                     "bool", "boolean", "path", "file", "list", "array", "json"):
            return f"input {k!r}: unknown type {t!r}"
    declared = set(inputs) | set(get_resources(spec))
    cmd = spec.get("command") or ""
    if spec.get("_active_subcommand") and ("{subcommand}" in cmd or "{{subcommand}}" in cmd):
        return "leaf command still contains subcommand placeholder"
    used = TEMPLATE_VAR_NAME.findall(cmd)
    if spec.get("arg_style") == "subcommand":
        used = [v for v in used if v != "subcommand"]
    missing = sorted({v for v in used if v not in declared})
    if missing:
        return f"command references undeclared inputs: {missing} (command {cmd[:60]})"
    return ""


def render_spec(spec: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Build the argv for a leaf/non-subcommand spec from the SAME canonical
    inputs every other stage reads. Pure renderer lives in argv_renderer (the
    runner delegates to it too), so the schema/contract layer has no dependency
    on the executor."""
    from agent_connector.argv_renderer import (  # noqa: PLC0415
        render_command, render_subcommand,
    )

    if spec.get("arg_style") == "subcommand":
        return render_subcommand(spec, args)
    return render_command(spec.get("command") or "", args)
