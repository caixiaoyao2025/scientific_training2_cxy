"""Code generation: turn ToolSpec entries into wrapper classes and produce
an Adapter that injects them into an arbitrary agent framework.

Wrapper: one class per tool, exposing a method named after the agent's
detected execution interface (default: run). Execution is delegated to
tool_runner.run_tool_spec, so wrappers never reimplement tool logic.

Adapter: calls the agent's detected registration method for each tool.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_EXECUTION_METHOD = "run"

WRAPPER_TEMPLATE = '''"""Auto-generated wrapper for tool: {name}"""
import json as _json
from pathlib import Path as _Path

from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {spec_repr}


class {class_name}:
    """Tool wrapper backed by a ToolSpec (registry.yaml entry)."""

    def __init__(self):
        self.name = {name_json}
        self.description = {desc_json}

    def __repr__(self):
        return f"<{{self.__class__.__name__}} name={{self.name}}>"

    def {method}(self, **kwargs):
        return format_result(run_tool_spec(_TOOL_SPEC, kwargs))

    def call_with_args(self, tool_input):
        if isinstance(tool_input, str):
            tool_input = _json.loads(tool_input)
        return self.{method}(**tool_input)

    __call__ = {method}
    func = {method}
'''

FUNCTION_WRAPPER_TEMPLATE = '''"""Auto-generated function wrapper for tool: {name}"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {spec_repr}


def {func_name}({signature}) -> str:
    """{desc}"""
    return format_result(run_tool_spec(_TOOL_SPEC, {kwargs_dict}))
'''


def make_class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_")) + "Wrapper"


def generate_wrappers(
    tools: list[dict[str, Any]],
    out_dir: str = "generated_tools",
    execution_method: str = DEFAULT_EXECUTION_METHOD,
    registration_style: str = "object",
) -> list[str]:
    """Write one _wrapper.py per tool; return the list of module paths.

    registration_style='function' emits a plain Python function (for
    frameworks that introspect functions, e.g. Biomni's add_tool).
    Otherwise a class exposing run/invoke/__call__/func is emitted.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    init_path = Path(out_dir) / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    written: list[str] = []
    for tool in tools:
        class_name = make_class_name(tool["name"])
        filename = Path(out_dir) / f"{tool['name']}_wrapper.py"

        if registration_style == "function":
            inputs = tool.get("inputs", {})
            func_name = _safe_identifier(tool["name"])
            signature = ", ".join(f"{name}: str" for name in inputs) or "**kwargs"
            if inputs:
                kwargs_dict = "{" + ", ".join(f"{name!r}: {name}" for name in inputs) + "}"
            else:
                kwargs_dict = "dict(kwargs)"
            code = FUNCTION_WRAPPER_TEMPLATE.format(
                name=tool["name"],
                func_name=func_name,
                signature=signature,
                desc=tool.get("description", ""),
                spec_repr=repr(tool),
                kwargs_dict=kwargs_dict,
            )
        else:
            code = WRAPPER_TEMPLATE.format(
                class_name=class_name,
                name=tool["name"],
                name_json=json.dumps(tool["name"], ensure_ascii=False),
                desc_json=json.dumps(tool.get("description", ""), ensure_ascii=False),
                spec_repr=repr(tool),
                method=_safe_identifier(execution_method),
            )
        filename.write_text(code, encoding="utf-8")
        written.append(str(filename))
    return written


def _safe_identifier(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not cleaned:
        cleaned = DEFAULT_EXECUTION_METHOD
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


ADAPTER_TEMPLATE = '''"""Auto-generated adapter for agent class: {agent_class}"""


class {adapter_class}:
    """Binds a list of tool wrappers into a {agent_class} agent."""

    def __init__(self, agent):
        self.agent = agent

    def install_tools(self, tools):
        for tool in tools:
            self.agent.{registration_method}(tool)
        return len(tools)
'''


def make_adapter_class_name(agent_class: str) -> str:
    return agent_class + "Adapter"


def generate_adapter(
    agent_class: str,
    registration_method: str,
    out_path: str = "adapter.py",
) -> str:
    """Write the adapter module and return its path."""
    code = ADAPTER_TEMPLATE.format(
        agent_class=agent_class,
        adapter_class=make_adapter_class_name(agent_class),
        registration_method=_safe_identifier(registration_method),
    )
    Path(out_path).write_text(code, encoding="utf-8")
    return out_path


def load_wrappers(package_name: str = "generated_tools", registration_style: str = "object") -> list[Any]:
    """Import every *_wrapper.py module and collect its wrapper callables.

    For registration_style='function', plain functions defined in the
    wrapper modules are collected. Otherwise Wrapper class instances are
    instantiated.
    """
    instances: list[Any] = []
    package = importlib.import_module(package_name)
    package_path = Path(package.__file__).parent
    for filename in sorted(os.listdir(package_path)):
        if not filename.endswith("_wrapper.py"):
            continue
        module = importlib.import_module(f"{package_name}.{filename[:-3]}")
        if registration_style == "function":
            for _name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module.__name__ and not _name.startswith("_"):
                    instances.append(obj)
        else:
            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if _name.endswith("Wrapper") and obj.__module__ == module.__name__:
                    instances.append(obj())
    return instances


def load_adapter(agent_class: str, adapter_path: str = "adapter.py") -> Any:
    """Import the generated adapter module and return the Adapter class.

    Loads from the file path directly, so it works regardless of sys.path
    or whether adapter.py lives inside a package directory.
    """
    module_name = Path(adapter_path).stem
    spec = importlib.util.spec_from_file_location(module_name, Path(adapter_path).resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, make_adapter_class_name(agent_class))


# --- fallback wiring for agents with no register method -------------------


def _tool_to_function_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-style function schema entry for a ToolSpec."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, meta in (tool.get("inputs") or {}).items():
        properties[name] = {
            "type": "string",
            "description": meta.get("description", "") or "",
        }
        required.append(name)
    # surface the install / environment contract so the caller knows what to
    # set up before invoking the tool (see discovery_to_registry.py 'install')
    install = tool.get("install") or {}
    fn = {
        "name": tool["name"],
        "description": tool.get("description", "") or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    if tool.get("arg_style"):
        fn["arg_style"] = tool["arg_style"]
    if install:
        fn["install"] = install
    return {"type": "function", "function": fn}


def generate_tools_manifest(tools: list[dict[str, Any]], out_path: str = "tools_manifest.json") -> str:
    """Wiring style 'manifest': write a `tools=[...]` list (OpenAI function
    calling / LangChain compatible) and return the output path.

    Consume it with:
      with open(out_path) as f:
          tools_list = json.load(f)
      llm_kwargs = {"tools": tools_list, "tool_choice": "auto"}
    """
    manifest = [_tool_to_function_schema(t) for t in tools]
    Path(out_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def generate_config_fragment(tools: list[dict[str, Any]], out_path: str = "tools_config.yaml") -> str:
    """Wiring style 'config': write a canonical tool-catalog config snippet that
    can be merged into an agent that reads tools from a config file.

    YAML by default; pass a path ending in .json to emit JSON instead. The
    shape is deliberately generic -- adjust key names to match the target
    agent's own config schema.
    """
    entries: dict[str, Any] = {}
    for tool in tools:
        inputs = {
            name: {"type": meta.get("type", "string"), "description": meta.get("description", "")}
            for name, meta in (tool.get("inputs") or {}).items()
        }
        entries[tool["name"]] = {
            "description": tool.get("description", "") or "",
            "execution": tool.get("execution") or {"type": tool.get("type"), "command": tool.get("command")},
            "inputs": inputs,
        }
    if str(out_path).endswith(".json"):
        Path(out_path).write_text(
            json.dumps({"tools": entries}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        import yaml

        Path(out_path).write_text(
            yaml.safe_dump({"tools": entries}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return out_path


def generate_prompt_block(tools: list[dict[str, Any]]) -> str:
    """Wiring style 'prompt': return a text block describing every tool that
    can be appended to the agent's system prompt.

    The agent reads the block and emits calls like:
      <tool name="samtools_flagstat" args='{"bam_path": "/data/x.bam"}'/>
    """
    lines = ["## Available tools", "Call a tool exactly as: "
             '<tool name="TOOL_NAME" args=\'{"param": "value"}\'/>', ""]
    for tool in tools:
        lines.append(f"### {tool['name']}")
        if tool.get("description"):
            lines.append(f"Description: {tool['description']}")
        inputs = tool.get("inputs") or {}
        if inputs:
            lines.append("Parameters:")
            for name, meta in inputs.items():
                lines.append(f"- {name} ({meta.get('type', 'string')}): {meta.get('description', '')}")
        lines.append("")
    return "\n".join(lines)


WIRING_INSTRUCTIONS = {
    "manifest": (
        "No register method found. Pass the manifest to each LLM invocation: "
        "tools_list = json.load(open('{manifest}')); llm(..., tools=tools_list). "
        "Each entry is an OpenAI function-calling schema."
    ),
    "config": (
        "No register method found. Merge the emitted tool catalog into the "
        "agent's config file (key names may need renaming to match its schema)."
    ),
    "prompt": (
        "No register method found. Append the prompt block to the agent's "
        "system prompt; the agent calls tools by emitting <tool name=... args=.../>."
    ),
}


def generate_wiring(
    tools: list[dict[str, Any]],
    schema: dict[str, Any],
    out_dir: str = "wiring",
    registration_style: str | None = None,
) -> dict[str, Any]:
    """Top-level wiring dispatch based on the agent_schema.

    - registration_method present      -> wrappers + adapter (inject path)
    - wiring_style 'manifest'/'config'/'prompt' -> emit the corresponding artifact
    Returns {'mode': ..., 'artifacts': {name: path|text}, 'instructions': ...}.
    """
    artifacts: dict[str, Any] = {}
    mode: str

    if schema.get("registration_method"):
        mode = "adapter"
        generated = generate_wrappers(
            tools,
            out_dir=os.path.join(out_dir, "generated_tools"),
            execution_method=schema.get("execution_method") or DEFAULT_EXECUTION_METHOD,
            registration_style=registration_style or schema.get("registration_style") or "object",
        )
        artifacts["wrappers"] = generated
        adapter_path = generate_adapter(
            schema.get("agent_class") or "Agent",
            schema["registration_method"],
            out_path=os.path.join(out_dir, "adapter.py"),
        )
        artifacts["adapter"] = adapter_path
        instructions = (
            "Inject with: "
            f"Adapter = load_adapter('{schema.get('agent_class') or 'Agent'}', adapter_path=...); "
            "Adapter(agent).install_tools(load_wrappers('generated_tools', registration_style=...))"
        )
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        style = schema.get("wiring_style") or "prompt"
        mode = style
        if style == "manifest":
            artifacts["manifest"] = generate_tools_manifest(tools, out_path=os.path.join(out_dir, "tools_manifest.json"))
        elif style == "config":
            artifacts["config"] = generate_config_fragment(tools, out_path=os.path.join(out_dir, "tools_config.yaml"))
        else:
            artifacts["prompt_block"] = generate_prompt_block(tools)
        instructions = WIRING_INSTRUCTIONS[style].format(**artifacts)

    return {"mode": mode, "artifacts": artifacts, "instructions": instructions}
