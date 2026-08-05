"""AST-based discovery of how an arbitrary agent framework registers and
executes tools. Rule-based by default; optionally refined by an LLM
(see resolver.py).

The scanner does not need the target framework installed: it only reads
source code and produces a structured agent_schema.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent-connector.scanner")

# --- keyword heuristics -------------------------------------------------
REGISTRATION_HINTS = ("register", "add_tool", "add_mcp", "install", "attach", "append")
TOOL_NAME_HINTS = ("tool", "function", "action", "skill", "api", "mcp", "plugin")
EXECUTION_METHOD_HINTS = ("run", "execute", "call", "invoke", "use", "apply", "__call__")
TOOL_SCHEMA_FIELDS = ("name", "description", "command", "parameters", "input_schema", "schema")
SKIP_DIR_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    "__pycache__",
    ".tox",
    ".eggs",
    ".idea",
}


class FunctionInfo:
    """Lightweight wrapper around an AST function/class-node with extra info."""

    __slots__ = ("node", "class_name", "file", "line")

    def __init__(self, node: ast.AST, class_name: str, file: str, line: int) -> None:
        self.node = node
        self.class_name = class_name
        self.file = file
        self.line = line

    @property
    def name(self) -> str:
        return getattr(self.node, "name", "")

    def body_text(self) -> str:
        try:
            return "\n".join(ast.unparse(stmt) for stmt in self.node.body)
        except Exception:
            return ""

    @property
    def params(self) -> list[str]:
        args = self.node.args.args  # type: ignore[attr-defined]
        return [a.arg for a in args]

    @property
    def decorators(self) -> list[str]:
        out = []
        for dec in getattr(self.node, "decorator_list", []):
            try:
                out.append(ast.unparse(dec))
            except Exception:
                pass
        return out


class ClassInfo:
    __slots__ = ("node", "file", "line")

    def __init__(self, node: ast.ClassDef, file: str, line: int) -> None:
        self.node = node
        self.file = file
        self.line = line

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def bases(self) -> list[str]:
        out = []
        for base in self.node.bases:
            try:
                out.append(ast.unparse(base))
            except Exception:
                pass
        return out

    def attributes(self) -> list[str]:
        out = []
        for stmt in self.node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        out.append(target.id)
        return out


class RepoModel:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self.files: list[str] = []
        self.functions: list[FunctionInfo] = []
        self.methods: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []

    def parse_errors(self) -> list[dict[str, Any]]:
        return getattr(self, "_parse_errors", [])


def _skip_dir(parts: tuple[str, ...]) -> bool:
    return any(part in SKIP_DIR_PARTS or part.startswith(".") for part in parts)


def scan_python_files(repo_path: str) -> list[str]:
    root = Path(repo_path).resolve()
    files: list[str] = []
    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if not _skip_dir((Path(current_root).name, d))]
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(str(Path(current_root) / filename))
    return files


def _collect(repo_path: str) -> RepoModel:
    model = RepoModel(repo_path)
    model.files = scan_python_files(repo_path)
    model._parse_errors = []  # type: ignore[attr-defined]

    for file in model.files:
        try:
            tree = ast.parse(Path(file).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            model._parse_errors.append({"file": file, "error": str(exc)})  # type: ignore[attr-defined]
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, ast.AsyncFunctionDef):
                    continue
                info = FunctionInfo(node, "", file, getattr(node, "lineno", 0))
                model.functions.append(info)
            elif isinstance(node, ast.ClassDef):
                model.classes.append(ClassInfo(node, file, getattr(node, "lineno", 0)))

    # Map methods into their class.
    for cls in model.classes:
        for stmt in cls.node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(stmt, ast.AsyncFunctionDef):
                    continue
                model.methods.append(FunctionInfo(stmt, cls.name, cls.file, stmt.lineno))

    return model


# --- classification rules ------------------------------------------------
def _decorator_is_tool_registration(decorators: list[str]) -> bool:
    for dec in decorators:
        cleaned = dec.replace(" ", "")
        if cleaned in {"@tool", "@tool_use", "@register_tool", "@mcp.tool()", "@mcp.tool"}:
            return True
        if "@tool(" in cleaned or "mcp.tool(" in cleaned:
            return True
    return False


def _detect_registration_style(name: str, body_lower: str, params: list[str]) -> str:
    """Classify how the framework expects the 'tool' argument to look:
    function (a callable it introspects), dict (schema mapping), or object
    (a tool class instance)."""
    if "inspect.getsource" in body_lower or "function_to_api_schema" in body_lower or "function_to_schema" in body_lower:
        return "function"
    if "required_parameters" in body_lower or "required_keys" in body_lower or "validate_tool" in body_lower:
        return "dict"
    tool_params = [p for p in params if p not in {"self", "cls", "this"}]
    if tool_params and any("tool" in p.lower() or "api" in p.lower() for p in tool_params):
        if ".append(" in body_lower:
            return "object"
    return "object"


def find_registration_candidates(model: RepoModel) -> list[dict[str, Any]]:
    """Return functions/methods that look like 'register a tool into the agent'."""
    candidates: list[dict[str, Any]] = []
    for info in model.functions + model.methods:
        name_lower = info.name.lower()
        body_lower = info.body_text().lower()
        decs = info.decorators

        is_decorator_reg = _decorator_is_tool_registration(decs)
        has_reg_hint = any(h in name_lower for h in REGISTRATION_HINTS)
        has_tool_hint = any(h in name_lower for h in TOOL_NAME_HINTS)
        body_manages_tools = any(
            token in body_lower for token in ("tools", "tool", "functions", "mcp")
        )

        if not (is_decorator_reg or (has_reg_hint and (has_tool_hint or body_manages_tools))):
            continue

        params = info.params
        tool_params = [p for p in params if p not in {"self", "cls", "this"}]
        storage = None
        if ".tools.append" in body_lower:
            storage = "<obj>.tools"
        elif "self.tools" in body_lower:
            storage = "self.tools"

        candidates.append(
            {
                "name": info.name,
                "class_name": info.class_name,
                "file": info.file,
                "line": info.line,
                "params": params,
                "tool_param_candidates": tool_params,
                "registration_style": _detect_registration_style(info.name, body_lower, params),
                "storage": storage,
                "decorator": is_decorator_reg,
                "decorators": decs,
                "evidence_snippet": info.body_text()[:300],
            }
        )
    return candidates


def find_execution_candidates(model: RepoModel) -> list[dict[str, Any]]:
    """Return methods that execute a tool object (name in EXECUTION_METHOD_HINTS)."""
    candidates: list[dict[str, Any]] = []
    for info in model.methods:
        name_lower = info.name.lower()
        if not any(name_lower.startswith(h) or name_lower == h for h in EXECUTION_METHOD_HINTS):
            continue
        body_lower = info.body_text().lower()
        # A tool-execution method usually touches tool inputs/outputs or dispatches.
        dispatches = any(token in body_lower for token in ("dispatch", "kwargs", "arguments", "inputs"))
        candidates.append(
            {
                "name": info.name,
                "class_name": info.class_name,
                "file": info.file,
                "line": info.line,
                "params": info.params,
                "dispatches_args": dispatches,
                "evidence_snippet": info.body_text()[:300],
            }
        )
    return candidates


def find_tool_class_candidates(model: RepoModel) -> list[dict[str, Any]]:
    """Return classes that look like the 'tool object' (has name/description/command fields)."""
    candidates: list[dict[str, Any]] = []
    for cls in model.classes:
        attributes = cls.attributes()
        name_lower = cls.name.lower()
        has_tool_hint = any(h in name_lower for h in TOOL_NAME_HINTS)
        has_schema_field = any(field in attributes for field in TOOL_SCHEMA_FIELDS)
        subclass_of_tool = any("tool" in base.lower() or "base" in base.lower() for base in cls.bases)
        if (has_tool_hint or has_schema_field or subclass_of_tool) and not name_lower.startswith("_"):
            candidates.append(
                {
                    "name": cls.name,
                    "file": cls.file,
                    "line": cls.line,
                    "bases": cls.bases,
                    "attributes": attributes,
                }
            )
    return candidates


_INVOCATION_RE = re.compile(r"\.(invoke|run|execute|__call__|call)\s*\(")


def detect_execution_method(files: list[str]) -> tuple[str | None, int, str | None]:
    """Scan source for `.<method>(` calls on tool-like objects and return the
    most common one. Useful when the framework relies on an external tool
    protocol (e.g. LangChain `.invoke`) not defined inside its own source."""
    counts: dict[str, int] = {}
    example_file: dict[str, str] = {}
    for file in files:
        try:
            text = Path(file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _INVOCATION_RE.finditer(text):
            method = match.group(1)
            counts[method] = counts.get(method, 0) + 1
            example_file.setdefault(method, file)
    if not counts:
        return None, 0, None
    method, count = max(counts.items(), key=lambda item: item[1])
    return method, count, example_file.get(method)


def build_schema(
    repo_path: str,
    *,
    include_evidence: bool = False,
) -> dict[str, Any]:
    """Run the full rule-based analysis and produce an agent_schema dict.

    Fields filled by heuristics; 'confidence' tells the caller how sure we
    are. An LLM pass (resolver.py) can confirm/override these.
    """
    model = _collect(repo_path)
    registrations = find_registration_candidates(model)
    executions = find_execution_candidates(model)
    tool_classes = find_tool_class_candidates(model)

    # Prefer methods/classes defined on a class whose name contains Agent/Executor,
    # then standalone functions.
    def rank_score(entry: dict[str, Any]) -> int:
        score = 0
        cls = entry.get("class_name") or ""
        name = entry.get("name") or ""
        if any(token in cls.lower() for token in ("agent", "executor", "react", "runner")):
            score += 3
        if entry.get("storage"):
            score += 2
        if entry.get("decorator"):
            score += 2
        if name in {"register_tool", "add_tool", "add_mcp"}:
            score += 2
        if name in {"execute", "run", "_run"}:
            score += 2
        return score

    best_reg = max(registrations, key=rank_score, default=None)
    best_exec = max(executions, key=rank_score, default=None)

    invoked_method, invoked_count, invoked_file = detect_execution_method(model.files)
    if best_exec is None and invoked_method:
        best_exec = {
            "name": invoked_method,
            "class_name": None,
            "file": invoked_file or "",
            "line": 0,
            "params": [],
            "dispatches_args": False,
            "evidence_snippet": "",
        }
    if best_exec and best_exec.get("file"):
        best_exec["execution_evidence"] = f"{best_exec['file']}:{best_exec['line']}"
        best_exec["invocation_count"] = invoked_count

    # Determine the agent/executor class name (where best candidates live).
    agent_class = best_reg.get("class_name") if best_reg else (best_exec.get("class_name") if best_exec else None)

    schema: dict[str, Any] = {
        "repo_path": str(Path(repo_path).resolve()),
        "wiring_style": detect_wiring_style(repo_path, best_reg["name"] if best_reg else None),
        "agent_class": agent_class,
        "registration_method": best_reg["name"] if best_reg else None,
        "registration_argument": (
            best_reg["tool_param_candidates"][0] if best_reg and best_reg["tool_param_candidates"] else None
        ),
        "registration_style": best_reg.get("registration_style") if best_reg else None,
        "registration_storage": best_reg.get("storage") if best_reg else None,
        "registration_via_decorator": bool(best_reg and best_reg.get("decorator")),
        "execution_method": best_exec["name"] if best_exec else None,
        "execution_class": best_exec.get("class_name") if best_exec else None,
        "tool_class": tool_classes[0]["name"] if tool_classes else None,
        "tool_class_fields": tool_classes[0]["attributes"] if tool_classes else [],
        "confidence": 0.7 if (best_reg and best_exec) else 0.3,
        "scanned_files": len(model.files),
        "parse_errors": model.parse_errors()[:10],
        "registrations": registrations[:10] if include_evidence else None,
        "executions": executions[:10] if include_evidence else None,
        "tool_classes": tool_classes[:10] if include_evidence else None,
    }
    return schema


CONFIG_FILE_HINTS = ("tool", "registry", "config", "mcp", "plugin")
WIRING_TOOLS_KWARG_RE = re.compile(r"(?<!\.)\btools\s*=\s*(\[|\w)", re.MULTILINE)


def detect_wiring_style(repo_path: str, registration_method: str | None = None) -> str | None:
    """Choose how tools can be wired into an agent that has no register method.

    Returns one of:
      None     -- a normal registration method exists (adapter path)
      manifest -- tools are passed as a `tools=[...]` list to each LLM call
      config   -- tools are defined in a config file the agent reads
      prompt   -- tools must be described as text in the system prompt
    """
    if registration_method:
        return None

    root = Path(repo_path).resolve()

    for file in scan_python_files(repo_path):
        try:
            text = Path(file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if WIRING_TOOLS_KWARG_RE.search(text):
            return "manifest"

    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if not _skip_dir((Path(current_root).name, d))]
        for filename in filenames:
            if filename.endswith((".yaml", ".yml", ".json", ".toml")):
                stem = Path(filename).stem.lower()
                if any(hint in stem for hint in CONFIG_FILE_HINTS):
                    return "config"

    return "prompt"
