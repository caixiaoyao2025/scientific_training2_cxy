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

import re
from typing import Any

TEMPLATE_VAR_NAME = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def canonical_key(name: str) -> str:
    """One input key per CLI token: --ont-in / <ONT_IN> / ont_in -> 'ont_in'.

    Mirrors execute_test._canonical_param_name so the command template
    placeholder, the `inputs` dict key and the function schema parameter
    always agree."""
    s = str(name or "").strip().strip("<>[]{}")
    s = s.lstrip("-")
    if not s:
        return ""
    return s.lower().replace("-", "_")


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
        meta["required"] = True  # a positional argv slot is mandatory
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
    leaf = dict(tool)
    leaf["name"] = f"{tool.get('name', '')}_{sub.replace('-', '_')}"
    leaf["_active_subcommand"] = sub
    leaf["description"] = (tool.get("description") or "") + f" -- {sub}"
    # concrete command: base executable + the chosen subcommand, NOT a
    # `{{subcommand}}` template the runner must interpret later.
    exe = (tool.get("command") or "").split()[0] if (tool.get("command") or "").strip() \
        else (tool.get("name") or "")
    leaf["command"] = f"{exe} {sub}".strip()
    leaf["inputs"] = {canonical_key(p.get("name", "")): _param_input(p)
                      for p in params if canonical_key(p.get("name", ""))}
    leaf["outputs"] = details.get("outputs") or tool.get("outputs") or {}
    leaf["resources"] = tool.get("resources") or {}
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
    used = TEMPLATE_VAR_NAME.findall(cmd)
    if spec.get("arg_style") == "subcommand":
        used = [v for v in used if v != "subcommand"]
    missing = sorted({v for v in used if v not in declared})
    if missing:
        return f"command references undeclared inputs: {missing} (command {cmd[:60]})"
    return ""


def render_spec(spec: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Build the argv for a leaf/non-subcommand spec from the SAME canonical
    inputs every other stage reads."""
    from agent_connector.tool_runner import _render_command, _render_subcommand  # noqa: PLC0415

    if spec.get("arg_style") == "subcommand":
        return _render_subcommand(spec, args)
    return _render_command(spec.get("command") or "", args)
