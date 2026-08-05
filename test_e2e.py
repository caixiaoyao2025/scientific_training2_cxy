"""Local end-to-end replication of the notebook pipeline using Biomni as target."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = r"C:\Users\123456\Desktop\st2\SCI1003_TEAM11"
TARGET = r"C:\Users\123456\AppData\Local\Temp\opencode\Biomni"

sys.path.insert(0, REPO)

import yaml  # noqa: E402
from agent_connector.scanner import build_schema  # noqa: E402
from agent_connector.generator import (  # noqa: E402
    generate_wiring,
    load_wrappers,
    load_adapter,
)

# --- step 2: scan ---
schema = build_schema(TARGET, include_evidence=False)
print("== agent_schema ==")
for k in ("agent_class", "registration_method", "registration_argument",
          "registration_style", "registration_via_decorator",
          "execution_method", "wiring_style", "confidence", "scanned_files"):
    print(f"  {k}: {schema[k]}")
assert schema["registration_method"] == "add_tool", schema
assert schema["registration_style"] == "function", schema
assert schema["wiring_style"] is None, "Biomni has a register method; wiring_style must be None"

# --- step 4: auto wiring ---
registry = yaml.safe_load(open(os.path.join(REPO, "registry.yaml"), encoding="utf-8"))
tools = registry["tools"]

work = tempfile.mkdtemp(prefix="e2e_")
wiring = generate_wiring(tools, schema, out_dir=os.path.join(work, "wiring"))
assert wiring["mode"] == "adapter", wiring["mode"]
print("\n== wiring ==")
print("  mode:", wiring["mode"])
print("  artifacts:", list(wiring["artifacts"]))

# --- step 5: load + inject into dummy agent, then call a tool ---
sys.path.insert(0, work)
wrappers = load_wrappers(package_name="wiring.generated_tools", registration_style="function")
print("  loaded wrappers:", len(wrappers))

Adapter = load_adapter("react", adapter_path=os.path.join(work, "wiring", "adapter.py"))


class DummyAgent:
    def __init__(self):
        self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)


agent = DummyAgent()
Adapter(agent).install_tools(wrappers)
print("  injected:", len(agent.tools), "tools")

sample = os.path.join(work, "sample.txt")
Path(sample).write_text("line1\nline2\nline3\n", encoding="utf-8")
target = next(t for t in agent.tools if getattr(t, "__name__", None) == "count_lines_python")
print("\n-- call count_lines_python(file_path=sample) --")
print(target(file_path=sample))
print("\nE2E OK")
