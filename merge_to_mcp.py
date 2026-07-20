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

    existing_names = set()
    if os.path.exists(USER_REGISTRY):
        with open(USER_REGISTRY, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        for t in existing.get("tools", []):
            existing_names.add(t.get("name", ""))

    fresh = [t for t in cleaned if t["name"] not in existing_names]

    if os.path.exists(USER_REGISTRY):
        with open(USER_REGISTRY, "r", encoding="utf-8") as f:
            user_reg = yaml.safe_load(f) or {}
    else:
        user_reg = {"tools": []}

    user_reg.setdefault("tools", []).extend(fresh)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USER_REGISTRY, "w", encoding="utf-8") as f:
        yaml.dump(user_reg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Merged {len(fresh)} new tools into {USER_REGISTRY}")
    print(f"Total user tools: {len(user_reg['tools'])}")

if __name__ == "__main__":
    merge_registries()
