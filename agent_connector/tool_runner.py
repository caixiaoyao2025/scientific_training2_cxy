"""Standalone ToolSpec executor used by generated wrappers.

Dispatches a ToolSpec's `execution` block to the right runner without
depending on the MCP server package. Mirrors the Execution Engine in
server.py (cli / python / api / docker).

Isolation rules:
  - cli / docker: already run in a separate subprocess.
  - python: now also runs in a separate subprocess (dedicated interpreter),
    so tool code can never crash, pollute (sys.path / os.environ) or
    conflict (shared packages) with the host agent process.
  - api: pure HTTP request, no subprocess needed.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any


def _render_command(command_template: str, arguments: dict[str, Any]) -> list[str]:
    quoted = {key: shlex.quote(str(value)) for key, value in arguments.items()}
    rendered = command_template.format(**quoted)
    argv = shlex.split(rendered, posix=True)
    if not argv:
        raise ValueError("Rendered command is empty.")
    return argv


def _coerce_arguments(spec: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Cast str args to the types declared in ToolSpec.inputs.

    Wrappers receive everything as str (LLM output); a declared `type:
    int` argument should reach the tool as an int, not "42".
    """
    coerced = dict(arguments)
    for name, meta in (spec.get("inputs") or {}).items():
        if name not in coerced:
            continue
        v = coerced[name]
        t = (meta or {}).get("type", "string")
        try:
            if t in ("int", "integer") and not isinstance(v, bool):
                coerced[name] = int(v)
            elif t in ("float", "number") and not isinstance(v, bool):
                coerced[name] = float(v)
            elif t in ("bool", "boolean"):
                if isinstance(v, str):
                    coerced[name] = v.strip().lower() in ("1", "true", "yes")
                else:
                    coerced[name] = bool(v)
        except (TypeError, ValueError):
            pass  # keep original value; the tool may still handle it
    return coerced


def _run_cli(command: str, arguments: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    argv = _render_command(command, arguments)
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=False,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return {
            "status": "command_error",
            "return_code": 127,
            "stdout": "",
            "stderr": f"command not found: {argv[0]}",
            "argv": argv,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "command_error",
            "return_code": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": f"timed out after {timeout}s",
            "argv": argv,
        }
    return {
        "status": "ok" if completed.returncode == 0 else "command_error",
        "return_code": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "argv": argv,
    }


# Runs in a dedicated interpreter: import module, call function, return JSON
# on stdout. Never runs in the host agent process.
_PYTHON_RUNNER_SOURCE = r'''
import importlib
import json
import sys

entry_point = sys.argv[1]
arguments = json.loads(sys.argv[2])
module_name, _, function_name = entry_point.partition(":")
try:
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    result = function(**arguments)
except BaseException as exc:  # noqa: BLE001 - report any failure, incl. sys.exit
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(1)
if result is not None and not isinstance(result, str):
    output = json.dumps(result, ensure_ascii=False, default=str)
elif isinstance(result, str):
    output = result
else:
    output = ""
print(json.dumps({"output": output}))
'''


def _run_python(entry_point: str, arguments: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    module_name, _, function_name = entry_point.partition(":")
    if not function_name:
        raise ValueError(f"entry_point must be 'module:function', got {entry_point!r}")
    env = os.environ.copy()
    # Let the child interpreter resolve the same modules as the host process
    # (e.g. tool_helpers living in the tools repo, added to sys.path at setup).
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    argv = [sys.executable, "-c", _PYTHON_RUNNER_SOURCE, entry_point,
            json.dumps(arguments, ensure_ascii=False)]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=False,
            timeout=timeout, encoding="utf-8", errors="replace", env=env,
        )
    except FileNotFoundError:
        return {
            "status": "command_error",
            "return_code": 127,
            "stdout": "",
            "stderr": f"python executable not found: {argv[0]}",
            "argv": argv,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "command_error",
            "return_code": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": f"timed out after {timeout}s",
            "argv": argv,
        }
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if stdout.strip():
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None  # not our protocol; surface raw output
        if payload is not None and "error" in payload:
            # child reported a failure -> move the message to stderr
            stderr = (payload["error"] + ("\n" + stderr if stderr else "")).strip()
            stdout = ""
        elif payload is not None and completed.returncode == 0:
            stdout = payload.get("output", "")
    return {
        "status": "ok" if completed.returncode == 0 else "command_error",
        "return_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "argv": argv,
    }


def _run_api(execution: dict[str, Any], arguments: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    import urllib.error
    import urllib.parse
    import urllib.request

    endpoint = execution["endpoint"]
    method = str(execution.get("method", "POST")).upper()
    placeholders = re.findall(r"\{(\w+)\}", endpoint)
    missing = [k for k in placeholders if k not in arguments]
    if missing:
        return {
            "status": "command_error",
            "return_code": None,
            "stdout": "",
            "stderr": f"missing args for URL template: {missing}",
            "argv": [endpoint],
        }
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
    argv = ["docker", "run", "--rm"]
    volumes = execution.get("volumes") or []
    if volumes:
        for vol in volumes:
            argv += ["-v", str(vol)]
    elif os.path.isdir("data"):
        # ToolSpec paths are described as "/data/..."; bind the repo data dir
        # so the container can actually see them.
        argv += ["-v", f"{os.path.abspath('data')}:/data"]
    argv += [execution["image"], *command_argv]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=False,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return {
            "status": "command_error",
            "return_code": 127,
            "stdout": "",
            "stderr": "docker not found on PATH; install Docker to use this tool",
            "argv": argv,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "command_error",
            "return_code": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": f"timed out after {timeout}s",
            "argv": argv,
        }
    return {
        "status": "ok" if completed.returncode == 0 else "command_error",
        "return_code": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "argv": argv,
    }


def _ensure_installed(spec: dict[str, Any]) -> list[str]:
    """Auto-install a tool's environment if its command/module is missing.

    Reads the tool's `install` contract (from discovery_to_registry.py) and
    installs what's needed so a downstream agent can invoke the tool without
    manual setup. Returns a list of install actions performed ([] if none).
    """
    import shutil as _sh
    actions: list[str] = []
    install = spec.get("install") or {}
    method = install.get("method", "")
    command = (spec.get("command") or "")
    cmd_name = command.split()[0] if command else ""

    def _try_install(argv: list[str], timeout: int = 600) -> bool:
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, check=False,
                                timeout=timeout, encoding="utf-8", errors="replace")
            return cp.returncode == 0
        except Exception:
            return False

    exe_name = cmd_name.split("/")[-1] if cmd_name else ""
    if method in ("pip_pkg", "pip_url") and exe_name and _sh.which(exe_name) is None:
        target = install.get("command", "")
        # command may be "pip install <url>" (full shell) or just "<url>"
        if target.startswith("pip "):
            parts = target.split()
            target = parts[2] if len(parts) >= 3 else ""
        if target and not target.startswith("pip "):
            if _try_install([sys.executable, "-m", "pip", "install", "-q", target]):
                actions.append(f"pip install {target}")
    elif method == "cargo" and exe_name and _sh.which(exe_name) is None:
        url = install.get("command", "")
        if url:
            argv = url.replace("cargo install --git", "cargo install --git").split()
            if _try_install(argv, 1800):
                actions.append(f"cargo install {url}")
    elif method == "npm" and exe_name and _sh.which(exe_name) is None:
        pkg = install.get("command", "") or cmd_name
        if _try_install(["npm", "install", "-g", pkg], 1800):
            actions.append(f"npm install -g {pkg}")
    return actions


def run_tool_spec(spec: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a ToolSpec (registry.yaml entry) with the given arguments."""
    execution = spec.get("execution")
    if not isinstance(execution, dict) or not execution.get("type"):
        execution = {"type": spec.get("type", "cli"), "command": spec.get("command", "")}
    exec_type = execution.get("type", "cli")
    timeout = int(spec.get("timeout_seconds", 600))
    arguments = _coerce_arguments(spec, arguments)

    # auto-install the tool's environment on first use (agent self-provisioning)
    installed = _ensure_installed(spec)
    if installed:
        print(f"[tool-runner] auto-installed: {installed}")

    if exec_type == "python":
        return _run_python(execution["entry_point"], arguments, timeout=timeout)
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
