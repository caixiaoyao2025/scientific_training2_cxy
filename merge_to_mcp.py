import yaml
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
USER_REGISTRY = os.path.join(DATA_DIR, "mcp_registry.yaml")
ARCHIVE_REGISTRY = os.path.join(DATA_DIR, "registry_archive.yaml")
DISCOVERED = os.path.join(os.path.dirname(__file__), "discovered_registry.yaml")

def clean_tool_entry(tool):
    """Keep user-facing schema fields; promote the verification evidence from
    _discovery_metadata to top-level `evidence.*` so downstream agents and
    reviewers can see how the tool was verified. Pure debugging fields are
    dropped.
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
    return entry

def merge_registries():
    with open(DISCOVERED, "r", encoding="utf-8") as f:
        discovered = yaml.safe_load(f) or {}

    new_tools = discovered.get("tools", [])
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
    with open(USER_REGISTRY, "w", encoding="utf-8") as f:
        yaml.dump(user_reg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    with open(ARCHIVE_REGISTRY, "w", encoding="utf-8") as f:
        yaml.dump(archive_reg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Merged {len(fresh)} new + {len(updated)} updated tools into {USER_REGISTRY}")
    if dropped:
        print(f"  archived {len(dropped)} stale entries (not in this run's discovery): {dropped}")
    print(f"Total active tools: {len(user_reg['tools'])} (archive: {len(archive_reg['tools'])})")

if __name__ == "__main__":
    merge_registries()
