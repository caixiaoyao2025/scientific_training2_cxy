"""Standalone ToolSpec executor used by generated wrappers.

Dispatches a ToolSpec's `execution` block to the right runner without
depending on the MCP server package. Mirrors the Execution Engine in
server.py (cli / python / api / docker).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from typing import Any


def _render_command(command_template: str, arguments: dict[str, Any]) -> list[str]:
    quoted = {key: shlex.quote(str(value)) for key, value in arguments.items()}
    rendered = command_template.format(**quoted)
    argv = shlex.split(rendered, posix=True)
    if not argv:
        raise ValueError("Rendered command is empty.")
    return argv


def _run_cli(command: str, arguments: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    argv = _render_command(command, arguments)
    completed = subprocess.run(
        argv, capture_output=True, text=True, check=False, timeout=timeout
    )
    return {
        "status": "ok" if completed.returncode == 0 else "command_error",
        "return_code": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "argv": argv,
    }


def _run_python(entry_point: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import contextlib
    import importlib
    import io

    module_name, _, function_name = entry_point.partition(":")
    if not function_name:
        raise ValueError(f"entry_point must be 'module:function', got {entry_point!r}")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            result = function(**arguments)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "command_error",
            "return_code": 1,
            "stdout": buffer.getvalue(),
            "stderr": f"{type(exc).__name__}: {exc}",
            "argv": [entry_point],
        }
    output = buffer.getvalue()
    if result is not None and not isinstance(result, str):
        output += json.dumps(result, ensure_ascii=False, default=str) + "\n"
    elif isinstance(result, str):
        output += result + "\n"
    return {"status": "ok", "return_code": 0, "stdout": output, "stderr": "", "argv": [entry_point]}


def _run_api(execution: dict[str, Any], arguments: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    import urllib.error
    import urllib.parse
    import urllib.request

    endpoint = execution["endpoint"]
    method = str(execution.get("method", "POST")).upper()
    quoted = {key: urllib.parse.quote(str(value)) for key, value in arguments.items()}
    rendered_url = endpoint.format(**quoted)
    try:
        if method == "GET":
            query_args = {k: v for k, v in arguments.items() if "{" + k + "}" not in endpoint}
            if query_args:
                rendered_url += ("&" if "?" in rendered_url else "?") + urllib.parse.urlencode(query_args)
            request = urllib.request.Request(rendered_url, method="GET")
        else:
            request = urllib.request.Request(
                rendered_url,
                data=json.dumps(arguments).encode("utf-8"),
                method=method,
                headers={"Content-Type": "application/json"},
            )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "status": "ok",
                "return_code": response.status,
                "stdout": response.read().decode("utf-8", errors="replace"),
                "stderr": "",
                "argv": [rendered_url],
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": "command_error",
            "return_code": exc.code,
            "stdout": "",
            "stderr": exc.read().decode("utf-8", errors="replace"),
            "argv": [rendered_url],
        }
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "status": "command_error",
            "return_code": None,
            "stdout": "",
            "stderr": str(exc),
            "argv": [rendered_url],
        }


def _run_docker(execution: dict[str, Any], arguments: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    command_argv: list[str] = []
    if execution.get("command"):
        command_argv = _render_command(execution["command"], arguments)
    argv = ["docker", "run", "--rm", execution["image"], *command_argv]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)
    return {
        "status": "ok" if completed.returncode == 0 else "command_error",
        "return_code": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "argv": argv,
    }


def run_tool_spec(spec: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a ToolSpec (registry.yaml entry) with the given arguments."""
    execution = spec.get("execution")
    if not isinstance(execution, dict) or not execution.get("type"):
        execution = {"type": spec.get("type", "cli"), "command": spec.get("command", "")}
    exec_type = execution.get("type", "cli")
    timeout = int(spec.get("timeout_seconds", 600))

    if exec_type == "python":
        return _run_python(execution["entry_point"], arguments)
    if exec_type == "api":
        return _run_api(execution, arguments, timeout=timeout)
    if exec_type == "docker":
        return _run_docker(execution, arguments, timeout=timeout)
    return _run_cli(execution.get("command", ""), arguments, timeout=timeout)


def format_result(result: dict[str, Any]) -> str:
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    lines = [line for line in stdout.splitlines() if line.strip()]
    preview = "\n".join(lines[:50])
    if len(lines) > 50:
        preview += f"\n... ({len(lines) - 50} more lines truncated)"
    if stderr.strip():
        preview = (preview + "\n[stderr]\n" + stderr) if preview else "[stderr]\n" + stderr
    status = result.get("status", "ok")
    return f"[tool status: {status}, exit code: {result.get('return_code')}]\n{preview}".strip()
