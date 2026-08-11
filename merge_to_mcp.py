import yaml
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
USER_REGISTRY = os.path.join(DATA_DIR, "mcp_registry.yaml")
DISCOVERED = os.path.join(os.path.dirname(__file__), "discovered_registry.yaml")

def clean_tool_entry(tool):
    entry = {k: v for k, v in tool.items() if not k.startswith("_")}
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

    existing = user_reg.get("tools", [])
    existing_names = {t.get("name", "") for t in existing}

    # New discoveries override same-name entries so re-runs refresh the schema
    # (e.g. newly parsed --help params, install contract, verification results).
    fresh = [t for t in cleaned if t["name"] not in existing_names]
    updated = [t for t in cleaned if t["name"] in existing_names]
    user_reg["tools"] = [t for t in existing if t.get("name", "") not in {u["name"] for u in updated}]
    user_reg["tools"].extend(fresh)
    user_reg["tools"].extend(updated)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USER_REGISTRY, "w", encoding="utf-8") as f:
        yaml.dump(user_reg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Merged {len(fresh)} new + {len(updated)} updated tools into {USER_REGISTRY}")
    print(f"Total user tools: {len(user_reg['tools'])}")

if __name__ == "__main__":
    merge_registries()
