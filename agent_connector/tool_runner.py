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
    # drop empty-string / None args so `--flag ""` never renders
    filtered = {k: v for k, v in arguments.items() if v not in (None, "", False)}
    # values are NOT pre-quoted: we tokenize the template, then substitute
    # each value verbatim into its own argv slot, so spaces stay inside a
    # single token and shell metachars are never re-interpreted.
    values = {key: str(value) for key, value in filtered.items()}
    # templates may use either Jinja-style {{x}} or Python-style {x} placeholders
    template = re.sub(r"\{\{(\w+)\}\}", r"{\1}", command_template)
    # tokenize the template (shlex keeps {x} placeholders intact since they
    # have no spaces), then drop any token that references a MISSING arg --
    # including the flag that precedes it, so `--pdb-path {{pdb_path}}` with
    # pdb_path unset renders to NOTHING, not a bare `--pdb-path`.
    tokens = shlex.split(template, posix=True)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("{") and tok.endswith("}"):
            var = tok[1:-1]
            if var in values:
                out.append(values[var])
                i += 1
            else:
                # missing value: also drop the preceding flag token if any
                if out and out[-1].startswith("-") and " " not in out[-1]:
                    out.pop()
                i += 1
        else:
            # flag token: peek ahead - if the next token is a missing
            # placeholder, drop BOTH (flag + its unset value)
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if (tok.startswith("-") and nxt and nxt.startswith("{") and nxt.endswith("}")
                    and nxt[1:-1] not in values):
                i += 2
            else:
                out.append(tok)
                i += 1
    argv = [a for a in out if a != ""]
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


_VALID_INPUT_TYPES = {"string", "str", "int", "integer", "float", "number",
                      "bool", "boolean", "path", "file", "list", "array", "json"}


def validate_arguments(spec: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Strict argument validation (the Pydantic layer, without the dep).

    Enforces: required inputs present, no unknown inputs, no pollution in the
    input schema, and every input has a legal type. Returns (cleaned_arguments,
    '') on success or ({}, reason) on failure so a bad tool_call is rejected
    before any subprocess runs.
    """
    inputs = spec.get("inputs") or {}
    if not isinstance(inputs, dict):
        return {}, f"input schema is not a dict: {inputs!r}"
    for name, meta in inputs.items():
        if not isinstance(name, str):
            return {}, f"input name not a string: {name!r}"
        if name != name.strip() or " " in name or "\t" in name:
            return {}, f"input name polluted: {name!r}"
        t = (meta or {}).get("type", "string")
        if t not in _VALID_INPUT_TYPES:
            return {}, f"input {name!r}: unknown type {t!r}"
    known = set(inputs.keys())
    unknown = set(arguments) - known
    if unknown:
        return {}, f"unknown arguments: {sorted(unknown)}"
    # an input with NO `required` field defaults to required (consistent with
    # to_function_schemas); only `required: false` is genuinely optional.
    missing = [k for k, m in inputs.items() if (m or {}).get("required") is not False
               and (arguments.get(k) in (None, ""))]
    if missing:
        return {}, f"missing required inputs: {missing}"
    # unknown template vars in the command would render to garbage argv.
    # EXCEPT subcommand CLIs: `{{subcommand}}` is injected by the dispatcher
    # (fnmap -> _active_subcommand), it is NOT a user-supplied input.
    import re as _re
    cmd = spec.get("command") or ""
    used = _re.findall(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}", cmd)
    if spec.get("arg_style") == "subcommand":
        used = [v for v in used if v != "subcommand"]
    unbound = sorted({v for v in used if v not in known})
    if unbound:
        return {}, f"command template references undeclared inputs: {unbound}"
    cleaned = {k: v for k, v in arguments.items() if k in known}
    return _coerce_arguments(spec, cleaned), ""


def _run_cli(command: str, arguments: dict[str, Any], timeout: int = 600,
             env: dict | None = None) -> dict[str, Any]:
    argv = _render_command(command, arguments)
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


def _run_python(entry_point: str, arguments: dict[str, Any], timeout: int = 600,
                venv_py: str | None = None, env: dict | None = None) -> dict[str, Any]:
    module_name, _, function_name = entry_point.partition(":")
    if not function_name:
        raise ValueError(f"entry_point must be 'module:function', got {entry_point!r}")
    if env is None:
        env = os.environ.copy()
        # Let the child interpreter resolve the same modules as the host process
        # (e.g. tool_helpers living in the tools repo, added to sys.path at setup).
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
    interpreter = venv_py or sys.executable
    argv = [interpreter, "-c", _PYTHON_RUNNER_SOURCE, entry_point,
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


def _try_import(pkg_name: str) -> tuple[int, str, str]:
    """Check whether a python package can be imported (python-API tools)."""
    if not pkg_name:
        return 1, "", "no package name"
    try:
        cp = subprocess.run(
            [sys.executable, "-c", f"import {pkg_name}"],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except Exception as exc:
        return 1, "", str(exc)


def _ensure_installed(spec: dict[str, Any], exec_type: str = "cli") -> tuple[list[str], list[str]]:
    """Auto-install a tool's environment if its command/module is missing.

    Returns (actions_performed, errors). Errors are surfaced so the caller
    (and the LLM) can tell 'not installed' from 'install failed'.
    Tools with heavy deps (torch/tensorflow etc.) are NOT auto-installed here:
    their install is too slow/flaky to do on-demand; they must be preinstalled
    in the environment (e.g. by the pipeline's execute step) or skipped.
    """
    import shutil as _sh
    heavy = ("torch", "tensorflow", "torchvision", "torchaudio", "jax",
             "cupy", "paddle", "triton", "pytorch", "transformers", "diffusers")
    declared = (spec.get("install") or {}).get("declared_packages") or []
    installed_pkgs = (spec.get("install") or {}).get("python_packages") or []
    declared_txt = " ".join(declared).lower()
    if any(h in declared_txt for h in heavy) or any(h in str(installed_pkgs).lower() for h in heavy):
        return [], ["heavy ML deps (torch/tf) detected - install is not auto-run here; preinstall or skip"]
    actions: list[str] = []
    errors: list[str] = []
    install = spec.get("install") or {}
    method = install.get("method", "")
    command = (spec.get("command") or "")
    cmd_name = command.split()[0] if command else ""
    exe_name = cmd_name.split("/")[-1] if cmd_name else ""

    def _try_install(argv: list[str], timeout: int = 600) -> bool:
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, check=False,
                                timeout=timeout, encoding="utf-8", errors="replace")
            return cp.returncode == 0
        except Exception:
            return False

    if method in ("pip_pkg", "pip_url"):
        # if the pipeline's execute step already installed this tool into a kept
        # venv, reuse it (skip on-demand install - heavy deps like torch).
        venv_path = (spec.get("install") or {}).get("venv_path", "")
        if venv_path and os.path.isdir(venv_path):
            return [], []
        # python-API tools (arg_style=python) are verified by import, not which
        is_python = spec.get("arg_style") == "python" or exec_type == "python"
        # determine the importable package name. For `python -m pkg.module` the
        # package is pkg, NOT the literal "python". Prefer install.command's
        # PyPI name (e.g. "bioemu") or execution.entry_point's module.
        pkg_for_import = (install.get("declared_packages") or [""])[0] or ""
        if not pkg_for_import:
            target0 = install.get("command", "")
            if target0.startswith("pip "):
                parts = target0.split()
                pkg_for_import = parts[2] if len(parts) >= 3 else ""
            elif target0:
                pkg_for_import = target0.split("==")[0].split(">=")[0].strip()
        if not pkg_for_import:
            cmd0 = (spec.get("command") or "").split()
            if len(cmd0) >= 3 and cmd0[0] == "python" and cmd0[1] == "-m":
                pkg_for_import = cmd0[2].split(".")[0]  # python -m bioemu.sample -> bioemu
            else:
                pkg_for_import = exe_name or ""
        need_install = False
        if is_python:
            # check importability
            rc_imp, _, _ = _try_import(pkg_for_import)
            if rc_imp != 0:
                need_install = True
        elif exe_name and _sh.which(exe_name) is None:
            need_install = True
        if need_install:
            target = install.get("command", "")
            if target.startswith("pip "):
                parts = target.split()
                target = parts[2] if len(parts) >= 3 else ""
            candidates = []
            if target and not target.startswith("pip "):
                candidates.append(target)
            if not candidates:
                candidates.append(pkg_for_import or exe_name)
            installed = False
            for cand in candidates:
                if _try_install([sys.executable, "-m", "pip", "install", "-q", cand]):
                    actions.append(f"pip install {cand}")
                    installed = True
                    break
                errors.append(f"pip install {cand} failed")
            if not installed and errors:
                errors = [errors[0]]
            elif installed and is_python:
                # verify import after install
                rc2, _, _ = _try_import(pkg_for_import)
                if rc2 != 0:
                    errors.append(f"installed but 'import {pkg_for_import}' still fails")
    elif method == "cargo" and exe_name and _sh.which(exe_name) is None:
        url = install.get("command", "")
        if url:
            argv = url.replace("cargo install --git", "cargo install --git").split()
            if _try_install(argv, 1800):
                actions.append(f"cargo install {url}")
            else:
                errors.append(f"cargo install {url} failed")
    elif method == "npm" and exe_name and _sh.which(exe_name) is None:
        pkg = install.get("command", "") or cmd_name
        if _try_install(["npm", "install", "-g", pkg], 1800):
            actions.append(f"npm install -g {pkg}")
        else:
            errors.append(f"npm install -g {pkg} failed")
    return actions, errors


def _render_subcommand(spec: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Render a subcommand-CLI invocation by dispatching to the chosen subcommand.

    spec.subcommand_details[sub] = {params: [{name:'--input',...}], usage}.
    We build `<cmd> <sub> --input <val> --output <val>` using ONLY the params
    that subcommand declares, from the agent-passed arguments. The agent just
    passes all params; we dispatch by the `subcommand` argument.
    """
    command = (spec.get("command") or "").split()[0]
    # subcommand can come from the `subcommand` argument OR from an
    # explicitly-dispatched call (fnmap set _active_subcommand)
    sub = spec.get("_active_subcommand") or arguments.get("subcommand", "")
    details = (spec.get("subcommand_details") or {}).get(sub) or {}
    params = details.get("params") or []
    if not params:
        # no per-sub detail: fall back to generic template
        return _render_command(spec.get("command") or "", arguments)
    argv = [command, sub]
    for p in params:
        name = p.get("name", "")
        key = name.lstrip("-").replace("-", "_").lower()
        val = arguments.get(key)
        # omit empty/None optional values entirely (don't render --flag "")
        if val in (None, "", False):
            continue
        if p.get("positional"):
            # positional arg: just the value in order, no flag
            argv.append(shlex.quote(str(val)))
        elif name.startswith("--"):
            argv.append(name)
            argv.append(shlex.quote(str(val)))
        else:
            argv.append(name)
            argv.append(shlex.quote(str(val)))
    return argv


def run_tool_spec(spec: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a ToolSpec (registry.yaml entry) with the given arguments."""
    execution = spec.get("execution")
    if not isinstance(execution, dict) or not execution.get("type"):
        execution = {"type": spec.get("type", "cli"), "command": spec.get("command", "")}
    exec_type = execution.get("type", "cli")
    timeout = int(spec.get("timeout_seconds", 600))
    arguments, arg_err = validate_arguments(spec, arguments)
    if arg_err:
        return {"status": "validation_error", "return_code": None,
                "stdout": "", "stderr": f"[validation] {arg_err}", "argv": []}

    # auto-install the tool's environment on first use (agent self-provisioning)
    installed, install_errors = _ensure_installed(spec, exec_type)
    if installed:
        print(f"[tool-runner] auto-installed: {installed}")
    if install_errors:
        print(f"[tool-runner] auto-install failed: {install_errors}")

    # if a kept venv exists (from execute step), run the tool inside it
    venv_path = (spec.get("install") or {}).get("venv_path", "")
    venv_py = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv_path, "bin", "python")
    if venv_path and os.path.isdir(venv_path) and os.path.exists(venv_py):
        bin_dir = os.path.join(venv_path, "Scripts") if os.name == "nt" else os.path.join(venv_path, "bin")
        env_run = dict(os.environ)
        env_run["PATH"] = bin_dir + os.pathsep + env_run.get("PATH", "")
    else:
        env_run = None

    if exec_type == "python":
        ep = execution.get("entry_point")
        if ep:
            return _run_python(ep, arguments, timeout=timeout, venv_py=venv_py if env_run else None,
                               env=env_run)
        return _run_cli(execution.get("command", ""), arguments, timeout=timeout,
                        env=env_run)
    if exec_type == "api":
        return _run_api(execution, arguments, timeout=timeout)
    if exec_type == "docker":
        return _run_docker(execution, arguments, timeout=timeout)
    # subcommand CLIs: dispatch by the `subcommand` argument to that sub's params
    if spec.get("arg_style") == "subcommand":
        try:
            argv = _render_subcommand(spec, arguments)
            try:
                completed = subprocess.run(
                    argv, capture_output=True, text=True, check=False,
                    timeout=timeout, encoding="utf-8", errors="replace", env=env_run)
                return {
                    "status": "ok" if completed.returncode == 0 else "command_error",
                    "return_code": completed.returncode,
                    "stdout": completed.stdout or "",
                    "stderr": completed.stderr or "",
                    "argv": argv,
                }
            except FileNotFoundError:
                return {"status": "command_error", "return_code": 127,
                        "stdout": "", "stderr": f"command not found: {argv[0]}", "argv": argv}
            except subprocess.TimeoutExpired:
                return {"status": "command_error", "return_code": None,
                        "stdout": "", "stderr": f"timed out after {timeout}s", "argv": argv}
        except Exception:
            pass  # fall through to generic template below
    result = _run_cli(execution.get("command", ""), arguments, timeout=timeout,
                      env=env_run)
    # if the command still can't be found after auto-install, tell the caller
    if result.get("return_code") == 127 and install_errors:
        result["stderr"] = (result.get("stderr", "") +
                            f"\n[auto-install failed] {'; '.join(install_errors)}")
    return result


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
    msg = f"[tool status: {status}, exit code: {result.get('return_code')}]\n{preview}".strip()
    # structured hints so the agent FIXES parameters instead of tool-hopping:
    # validation errors -> which inputs are wrong; command not found -> tell it
    # the tool environment is broken so it should not keep retrying that tool.
    if status == "validation_error":
        msg += "\n[error_type: invalid_arguments] Fix the argument names/values " \
               "per the tool's schema (required/type), then retry THIS tool."
    elif status == "command_error" and result.get("return_code") == 127:
        msg += ("\n[error_type: tool_not_installed] The tool executable is missing "
                "and could not be auto-installed. Do NOT retry this tool; choose "
                "another tool or report it as unavailable.")
    elif status == "command_error" and result.get("return_code") is None:
        msg += "\n[error_type: timeout] The tool timed out. Do NOT retry with the " \
               "same arguments."
    return msg
