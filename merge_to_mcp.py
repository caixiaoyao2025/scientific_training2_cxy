import yaml
import os
import re

# server.py's native convention (str.format placeholders + type whitelist)
SERVER_TOOL_TYPES = {"cli", "java", "script"}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
USER_REGISTRY = os.path.join(DATA_DIR, "mcp_registry.yaml")
ARCHIVE_REGISTRY = os.path.join(DATA_DIR, "registry_archive.yaml")
DISCOVERED = os.path.join(os.path.dirname(__file__), "discovered_registry.yaml")

def clean_tool_entry(tool):
    """Keep user-facing schema fields; promote the verification evidence from
    _discovery_metadata to top-level `evidence.*` so downstream agents and
    reviewers can see how the tool was verified. Pure debugging fields are
    dropped.

    Also normalizes the output contract: server.py renders ONLY
    `expected_outputs` (list of {name, render_as}) while auto-discovered
    entries carry `outputs` (dict). Convert file outputs into
    expected_outputs so both layers work off one entry.
    """
    entry = {k: v for k, v in tool.items() if not k.startswith("_")}
    md = tool.get("_discovery_metadata")
    if isinstance(md, dict):
        evidence = {
            "exec_status": md.get("exec_status", ""),
            "exec_reason": md.get("exec_reason", ""),
            "exec_executable": md.get("exec_executable", ""),
            "exec_retries": md.get("exec_retries", 0),
            "exec_heal_evidence": md.get("exec_heal_evidence", ""),
            "verified_license": md.get("verified_license", False),
            "verified_license_path": md.get("verified_license_path", ""),
            "verified_status": md.get("verified_status", ""),
            "inputs_source": md.get("inputs_source", ""),
            "params_schema": md.get("exec_params_schema", []),
            "installed_versions": md.get("exec_installed_versions", []),
        }
        evidence = {k: v for k, v in evidence.items() if v not in ("", None, [], False) or k in ("verified_license",)}
        if evidence:
            entry["evidence"] = evidence
    # outputs(dict) -> expected_outputs(list) for server.py's renderer.
    # Only outputs that reference a declared input path make sense as
    # expected_outputs (the renderer reads the path from the call args).
    if not entry.get("expected_outputs"):
        inputs = entry.get("inputs") or {}
        converted = []
        for name, spec in (entry.get("outputs") or {}).items():
            if name == "stdout":
                continue  # console output, not a renderable file parameter
            if name in inputs and (spec or {}).get("type") in ("file", "path"):
                converted.append({"name": name, "render_as": "text"})
        if converted:
            entry["expected_outputs"] = converted

    # ---- server.py compatibility boundary ----
    # The MCP server renders commands with str.format(), where `{{x}}` is an
    # ESCAPED literal (renders as the text "{x}", no substitution). The
    # discovery pipeline uses Jinja-style {{x}} (tool_runner handles both);
    # the merged registry must be server-native {x}.
    cmd = entry.get("command") or ""
    if cmd:
        entry["command"] = re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}",
                                  r"{\1}", cmd)
    # subcommand CLIs: the {subcommand} placeholder must be a declared input
    # or server.validate_tool_spec rejects the whole registry load. Leaf
    # function expansion (to_function_schemas) is unaffected: it builds leaf
    # inputs from subcommand_details, not top-level inputs.
    if entry.get("arg_style") == "subcommand":
        subs = entry.get("subcommands") or list((entry.get("subcommand_details") or {}).keys())
        if subs and "subcommand" not in (entry.get("inputs") or {}):
            entry.setdefault("inputs", {})
            entry["inputs"]["subcommand"] = {
                "type": "string",
                "required": True,
                "description": "Subcommand to run. One of: " + ", ".join(map(str, subs)),
            }
    # server's legacy type whitelist is {cli, java, script}; entries backed by
    # an execution block skip the type check, but `python`-typed entries with
    # a plain command template (python -m ...) are just CLIs to the server.
    if (not isinstance(entry.get("execution"), dict)
            and entry.get("type") not in SERVER_TOOL_TYPES):
        if entry.get("command"):
            entry["type"] = "cli"
    return entry

def _atomic_yaml_dump(path, data):
    """Write via temp file + os.replace so a crash mid-write can never leave
    a truncated/empty registry behind."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    os.replace(tmp, path)

def merge_registries():
    with open(DISCOVERED, "r", encoding="utf-8") as f:
        discovered = yaml.safe_load(f) or {}

    new_tools = discovered.get("tools", [])

    # QUALITY GATE: an empty discovered registry means the discovery/contract
    # pipeline failed THIS run -- it does NOT mean the old tools are invalid.
    # Merging an empty set would archive every existing active entry and
    # clear the MCP registry (one upstream bug = whole registry gone).
    if not new_tools:
        raise RuntimeError(
            "discovered_registry.yaml contains 0 tools; refusing to merge "
            "into / archive from the existing MCP registry. Fix the "
            "discovery/contract pipeline (check pending_tools.json and "
            "excluded_tools.json) before merging."
        )

    cleaned = [clean_tool_entry(t) for t in new_tools]

    if os.path.exists(USER_REGISTRY):
        with open(USER_REGISTRY, "r", encoding="utf-8") as f:
            user_reg = yaml.safe_load(f) or {}
    else:
        user_reg = {"tools": []}

    if os.path.exists(ARCHIVE_REGISTRY):
        with open(ARCHIVE_REGISTRY, "r", encoding="utf-8") as f:
            archive_reg = yaml.safe_load(f) or {}
    else:
        archive_reg = {"tools": []}

    existing = user_reg.get("tools", [])
    existing_names = {t.get("name", "") for t in existing}

    # New discoveries override same-name entries so re-runs refresh the schema
    # (e.g. newly parsed --help params, install contract, verification results).
    fresh = [t for t in cleaned if t["name"] not in existing_names]
    updated = [t for t in cleaned if t["name"] in existing_names]
    updated_names = {u["name"] for u in updated}

    # AUTHORITATIVE-ONLY MERGE: the active registry must reflect ONLY what THIS
    # run actually discovered + verified. An existing entry that was NOT
    # re-discovered this round is NOT carried forward -- "not found this run"
    # does not mean "still valid" (bqtools was a zombie: old placeholder schema
    # kept forever because its paper wasn't rediscovered). Preserve it in
    # registry_archive.yaml for history, but keep it OUT of the active registry.
    dropped = []
    archived = []
    kept_existing = []
    for t in existing:
        name = t.get("name", "")
        if name in updated_names:
            continue  # refreshed this run -> replaced by the new entry below
        dropped.append(name)
        archived.append(t)

    user_reg["tools"] = kept_existing
    user_reg["tools"].extend(fresh)
    user_reg["tools"].extend(updated)

    # archive: keep accumulated history (don't overwrite what's already there)
    archive_names = {a.get("name", "") for a in archive_reg["tools"]}
    archive_reg["tools"].extend([a for a in archived if a.get("name", "") not in archive_names])

    os.makedirs(DATA_DIR, exist_ok=True)
    _atomic_yaml_dump(USER_REGISTRY, user_reg)
    _atomic_yaml_dump(ARCHIVE_REGISTRY, archive_reg)

    print(f"Merged {len(fresh)} new + {len(updated)} updated tools into {USER_REGISTRY}")
    if dropped:
        print(f"  archived {len(dropped)} stale entries (not in this run's discovery): {dropped}")
    print(f"Total active tools: {len(user_reg['tools'])} (archive: {len(archive_reg['tools'])})")

if __name__ == "__main__":
    merge_registries()
