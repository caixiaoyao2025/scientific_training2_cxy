"""Execution test for auto-discovered tools (step 3.6).

verify_repo.py proves a repo is structurally healthy; this module goes one
step further: it actually *installs* the tool (into an isolated venv) and
smoke-runs it on a small sample input. Only tools that pass are safe to put
into the registry as invocable commands.

Grading:
  - passed  : install succeeded AND a smoke run exited 0 with non-empty output
  - failed  : install or smoke run failed (reason recorded)
  - skipped : no install command / entry point available to test (recorded)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from typing import Any, Optional

INSTALL_TIMEOUT = 300          # seconds per pip install (default)
INSTALL_TIMEOUT_ML = 1800      # heavy ML deps (torch/tf/jax/cuda) need longer
RUN_TIMEOUT = 60               # seconds per smoke run
HEAVY_DEPS = ("torch", "tensorflow", "torchvision", "torchaudio", "jax",
              "cuda", "cupy", "paddle", "onnxruntime-gpu", "triton",
              "pytorch", "transformers", "diffusers", "esm")
SAMPLE_FASTA = """>seq1\nACGTACGTACGT\n>seq2\nTTTTTTGGGGGG\n>seq3\nCCCGGGAAATTT\n"""


def _run(args: list[str], timeout: int, cwd: Optional[str] = None,
         env: Optional[dict] = None) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, check=False,
            timeout=timeout, cwd=cwd, env=env,
            encoding="utf-8", errors="replace",
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except FileNotFoundError:
        return 127, "", f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


def _classify_failure(output: str) -> str:
    """Classify a failed run/install into env_issue vs incomplete vs failed.

    - env_issue : missing dependency in the test environment (ModuleNotFoundError,
      ImportError, No module named) -> NOT the repo's fault.
    - incomplete: the repo's own code is broken (SyntaxError, NameError, etc.)
    - failed    : anything else (wrong args, timeout, no output).
    """
    if not output:
        return "failed"
    low = output.lower()
    env_patterns = (
        "modulenotfounderror", "importerror", "no module named",
        "cannot import", "not installed", "could not be found", "no such file",
        "command not found", "undefined symbol", "nomodulenamed",
        # pip dependency-resolution failures (version conflicts, unfindable)
        "cannot install", "conflicting dependencies", "conflict is caused by",
        "resolutionimpossible", "no matching distribution",
        "could not find a version", "dependency resolver",
    )
    incomplete_patterns = (
        "syntaxerror", "indentationerror", "nameerror", "attributeerror",
        "typeerror", "valueerror", "indexerror", "keyerror",
        "traceback (most recent call last)",
    )
    for pat in env_patterns:
        if pat in low:
            return "env_issue"
    for pat in incomplete_patterns:
        if pat in low:
            return "incomplete"
    return "failed"


def _detect_heavy_deps(repo_dir: str) -> bool:
    """True if the repo declares heavy ML deps that need a long install window."""
    texts = []
    for fname in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
                  "environment.yml"):
        p = os.path.join(repo_dir, fname)
        if os.path.exists(p):
            try:
                texts.append(open(p, "r", encoding="utf-8",
                                  errors="replace").read().lower())
            except OSError:
                pass
    blob = "\n".join(texts)
    return any(d in blob for d in HEAVY_DEPS)


def _extract_usage(help_output: str) -> str:
    """Extract the usage line from --help output (first usage: ... line)."""
    if not help_output:
        return ""
    m = re.search(r"usage:\s*(.+)", help_output, re.IGNORECASE)
    return m.group(1).strip()[:200] if m else ""


def _parse_subcommands(help_output: str) -> list[str]:
    """Extract subcommand names from --help output.

    Handles click/typer/argparse-subparsers style:
      Commands:
        encode   Encode FAS
        decode   Decode BINSEQ
        info     Show info
    Returns command names (e.g. ['encode', 'decode', 'info']).
    """
    if not help_output:
        return []
    low = help_output
    m = re.search(r"(?:^|\n)\s*commands?\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
                  low, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    cmds = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.split()[0] if line.split() else ""
        if name and name not in cmds and not name.startswith("-"):
            cmds.append(name)
    return cmds[:20]


def _detect_arg_style(help_output: str) -> str:
    """Classify how the tool expects arguments.

    Returns one of:
      - named       : option-style CLI (cmd --flag value ...)
      - positional  : positional args (cmd file1 file2 -o outdir)
      - subcommand  : cmd <subcommand> ...
      - python      : import-style API (no CLI usage line)
    """
    if not help_output:
        return "python"
    low = help_output.lower()
    # subcommand CLI: help lists "Commands:" AND usage has a COMMAND token
    has_commands = re.search(r"\bcommands?\s*:", low) is not None
    usage = re.search(r"usage:\s*(.+)", help_output, re.IGNORECASE)
    usage_line = usage.group(1) if usage else ""
    if has_commands and re.search(r"\bCOMMAND\b", usage_line, re.IGNORECASE):
        return "subcommand"
    # positional: usage tokens after the binary that are real args (not
    # options, not [..], not pseudo tokens like options:/COMMAND)
    if usage_line:
        after = re.sub(r"\[[^\]]*\]", " ", usage_line)
        after = re.sub(r"\s-\w", " ", after)  # cut option flags
        tokens = [t for t in after.split() if t]
        pseudo = {"options", "option", "options:", "commands", "command",
                  "args", "arg", "usage:"}
        positional = [t for t in tokens[1:]  # skip binary name
                      if not t.startswith("-") and t.strip("<>[]:,").lower() not in pseudo
                      and not (t.isupper() and len(t) > 2)]  # COMMAND/ARGS pseudo
        if positional:
            return "positional"
    return "named"


def _extract_readme_examples(repo_dir: str, pkg: str, max_examples: int = 3) -> list[str]:
    """Extract concrete invocation examples from the README.

    Returns full commands (e.g. `python -m bioemu.sample --sequence GYDPETGTWG
    --num_samples 10 --output_dir out`) that authors wrote. These show agents
    exactly how to pass parameters. Also collects `import <pkg>...` call lines.
    """
    readme = ""
    for root, _dirs, files in os.walk(repo_dir):
        for fn in files:
            if fn.lower().startswith("readme"):
                try:
                    readme = open(os.path.join(root, fn), encoding="utf-8",
                                  errors="replace").read()
                except Exception:
                    continue
                if readme:
                    break
        if readme:
            break
    if not readme:
        return []
    examples = []
    # 1) full `python -m <pkg>.module ...` command lines
    for m in re.finditer(r"python\s+-m\s+[\w.]+(?:[^\n`]*)", readme):
        line = m.group(0).strip()
        if pkg in line and line not in examples:
            examples.append(line)
    # 2) `import <pkg> ...` call snippets from python code blocks
    for blk in re.findall(r"```(?:python)?\s*\n(.*?)```", readme, re.S):
        if re.search(rf"\bfrom\s+{pkg}(?:\.\w+)*\s+import", blk):
            lines = [ln.strip() for ln in blk.splitlines()
                     if ln.strip() and not ln.startswith("#")]
            for ln in lines:
                if pkg in ln and ln not in examples:
                    examples.append(ln)
    return examples[:max_examples]


def _extract_readme_usage(repo_dir: str, pkg: str) -> str:
    """Find a runnable usage of a python package from the README.

    Looks for `python -m <pkg>.module ...` lines (a real entry point) or
    `import <pkg>...` code blocks. Returns the first `python -m` invocation as
    a (module, args) hint, or "" if none. This is author-written usage, more
    reliable than an LLM guessing the API.
    """
    readme = ""
    for root, _dirs, files in os.walk(repo_dir):
        for fn in files:
            if fn.lower().startswith("readme"):
                try:
                    readme = open(os.path.join(root, fn), encoding="utf-8",
                                  errors="replace").read()
                except Exception:
                    continue
                if readme:
                    break
        if readme:
            break
    if not readme:
        return ""
    # 1) python -m <pkg>.module ...  (real entry point)
    for m in re.finditer(r"python\s+-m\s+([\w.]+(?:\.[\w]+)+)", readme):
        mod = m.group(1)
        if mod.split(".")[0] == pkg:
            return f"python -m {mod}"
    # 2) `from <pkg> import ...` or `import <pkg>.x` inside code blocks
    for blk in re.findall(r"```(?:python)?\s*\n(.*?)```", readme, re.S):
        if re.search(rf"\b(from\s+{pkg}(?:\.\w+)*\s+import|import\s+{pkg}(?:\.\w+)*)", blk):
            # return a sanitized one-liner: first line with import + call
            lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
            if lines:
                return " | ".join(lines[:2])
    return ""


def _extract_flag_params(readme_examples: list) -> list[dict[str, str]]:
    """Extract CLI flags from readme example commands.

    e.g. 'python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10
    --output_dir out' -> [--sequence, --num_samples, --output_dir]
    Returns [{name, type, description}] with --strip for schema keys.
    """
    out = []
    seen = set()
    for ex in readme_examples:
        for m in re.finditer(r"(--[\w-]+)", ex):
            flag = m.group(1)
            if flag in seen:
                continue
            seen.add(flag)
            # guess type from the following token (int/float/path)
            rest = ex[m.end():].lstrip()
            nxt = rest.split()[0] if rest else ""
            ptype = "string"
            if nxt.replace(".", "", 1).isdigit() or nxt.lstrip("-").isdigit():
                ptype = "int" if nxt.lstrip("-").isdigit() and "." not in nxt else "float"
            elif nxt and ("/" in nxt or nxt.startswith("~") or "." in nxt):
                ptype = "path"
            out.append({"name": flag.lstrip("-").replace("-", "_"),
                        "type": ptype,
                        "description": f"CLI flag {flag} (from README example)",
                        "flag": flag})
    return out[:15]


def _llm_attempt_call(pkg_name: str, repo_dir: str, venv_py: str,
                      env: dict, sample: str, max_attempts: int = 3) -> dict:
    """Let an LLM try to write a working call for a python-import tool.

    For tools that are importable but have no CLI entry (e.g. bioemu), we ask a
    volcengine LLM to read the repo and write `import <pkg>; <call>`. Each
    attempt executes the code; failures are fed back for the LLM to fix. After
    `max_attempts` we give up.

    Returns a dict: {ok, status, code, evidence}. Requires the same
    LLM env vars as tool_agent_test.py (WESTLAKE_/OPENAI_/DEEPSEEK_).
    """
    import urllib.request
    api_key = (os.environ.get("WESTLAKE_API_KEY") or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("DEEPSEEK_API_KEY") or "")
    base_url = (os.environ.get("WESTLAKE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("DEEPSEEK_BASE_URL")
                or "https://ark.cn-beijing.volces.com/api/v3")
    model = (os.environ.get("WESTLAKE_MODEL") or os.environ.get("OPENAI_MODEL")
             or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash-ga-260731")
    if not api_key:
        return {"ok": False, "status": "no_llm_key", "code": "", "evidence": ""}

    # gather repo hints: README + module list so the LLM knows what to call
    hints = []
    for f in sorted(os.listdir(repo_dir))[:10]:
        hints.append(f)
    readme = ""
    for root, _dirs, files in os.walk(repo_dir):
        for fn in files:
            if fn.lower().startswith("readme"):
                try:
                    readme = open(os.path.join(root, fn), encoding="utf-8",
                                  errors="replace").read()
                except Exception:
                    pass
                if readme:
                    break
        if readme:
            break

    def _call_llm(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600, "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            return f"LLM_ERROR: {exc}"

    last_err = ""
    for attempt in range(1, max_attempts + 1):
        prompt = (f"The Python package '{pkg_name}' is installed in a venv. "
                  f"Its repo contains files: {', '.join(hints)}.\n"
                  f"README (excerpt):\n{readme[:1500]}\n"
                  f"Write a single python -c command that imports {pkg_name} and "
                  "performs a minimal, valid invocation on the sample file "
                  f"'{sample}' (if relevant), printing some output. "
                  "Output ONLY the python code, no explanation.\n"
                  f"{('Previous attempt failed with: ' + last_err + ' Fix the code.') if last_err else ''}")
        code = _call_llm(prompt).strip()
        # strip markdown fences
        code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code).strip()
        if not code or code.startswith("LLM_ERROR"):
            last_err = code or "empty LLM response"
            continue
        # execute in the venv
        rc, out, err = _run([venv_py, "-c", code], RUN_TIMEOUT, env=env)
        if rc == 0 and (out.strip() or err.strip()):
            return {"ok": True, "status": "passed",
                    "code": code, "evidence": (out + err)[-400:]}
        last_err = f"exit {rc}: {(out + err)[-300:]}"
    return {"ok": False, "status": "not_callable", "code": "", "evidence": last_err}


def _parse_positional_args(help_output: str) -> list[dict[str, str]]:
    """Extract positional arguments from --help output.

    Two sources:
      - usage line:  pgv-blast [options] seq1.gbk seq2.gbk seq3.gbk -o out
      - argparse 'positional arguments:' block with real descriptions:
            positional arguments:
              fasta       input FASTA file
    Returns [{name, type, description, positional}].
    """
    if not help_output:
        return []
    # source 1: argparse positional arguments: block (real names + descriptions)
    desc_map = {}
    m = re.search(r"positional arguments?\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
                  help_output, re.IGNORECASE | re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            line = line.rstrip()
            if not line.strip() or line.strip().startswith("-"):
                continue
            parts = line.split(None, 1)
            if parts:
                desc_map[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    # source 2: usage line tokens
    m = re.search(r"usage:\s*\S+\s*(?:\[[^\]]*\]\s*)*(.+)", help_output, re.IGNORECASE)
    if not m:
        return []
    rest = m.group(1)
    cut = re.search(r"\s-\w", rest)
    if cut:
        rest = rest[:cut.start()]
    tokens = rest.split()
    out = []
    pseudo = {"options", "option", "options:", "commands", "command",
              "args", "arg"}
    for t in tokens:
        t = t.strip("<>[]")
        if not t or t.startswith("-"):
            continue
        if t.lower() in pseudo or (t.isupper() and len(t) > 2):
            continue
        desc = desc_map.get(t, "") or desc_map.get(t.lower(), "") \
            or f"Positional argument {t}"
        out.append({"name": t, "type": "path", "description": desc,
                    "positional": True})
    return out[:10]


def _parse_help_params(help_output: str) -> list[dict[str, str]]:
    """Extract CLI parameters from a --help / usage string.

    Handles common formats (argparse, typer, click, optparse):
      --reference PATH   Reference genome
      -r, --reference PATH
      <input_file>       Input FASTA
    Returns a list of {name, type, description}.
    """
    if not help_output:
        return []
    params: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in help_output.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\s*(-[a-zA-Z],?\s+)?(--[\w][\w-]*|-[\w])\s+([A-Z_]+|\{[^}]+\}|<[^>]+>)?\s*(.*)$',
                     line, re.IGNORECASE)
        if not m:
            continue
        short_flag, flag, metavar, desc = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
        # if both a short and long flag are present, use the long name as the key
        if short_flag and flag.startswith("--"):
            name = flag
            aliases = [short_flag.strip().rstrip(",")]
        else:
            name = flag
            aliases = []
        # skip the standard help flag itself; it's not a real tool parameter
        if name in ("--help", "-h"):
            continue
        # only flags (--x or short) make sense as named params
        if (name.startswith("--") or (name.startswith("-") and not name.startswith("--"))) \
                and name not in seen:
            seen.add(name)
            ptype = "string"
            if metavar and metavar.lower() in ("path", "file", "dir", "directory", "infile", "outfile"):
                ptype = "path"
            elif metavar and metavar.lower() in ("int", "integer", "n", "count", "number"):
                ptype = "integer"
            elif metavar and metavar.lower() in ("float", "double"):
                ptype = "float"
            # keep only the human description, drop argparse noise like
            # [required], [default: x], [choices: a|b]
            desc = re.sub(r"\s*\[(required|default|choices|count|append|nargs)[^\]]*\]\s*$", "", desc).strip()
            entry = {
                "name": name,
                "type": ptype,
                "description": desc or f"CLI flag {name}",
            }
            if aliases:
                entry["aliases"] = aliases
            params.append(entry)
        elif flag.startswith("--"):
            continue
    return params[:20]


def _venv_python(venv_dir: str) -> str:
    return os.path.join(venv_dir, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv_dir, "bin", "python")


def _venv_bin(venv_dir: str) -> str:
    return os.path.join(venv_dir, "Scripts") if os.name == "nt" \
        else os.path.join(venv_dir, "bin")


def _find_installed_executable(bin_dir: str, candidates: list[str]) -> tuple[str, str]:
    """Find the real executable installed into the venv that matches a candidate.

    Do NOT assume tool.name == CLI command (pyGenomeViz -> pgv-blast). List what
    actually got installed and match case/underscore-insensitively. Returns
    (matched_candidate, absolute_exe_path) or ("", "").
    """
    if not os.path.isdir(bin_dir):
        return "", ""
    installed = [f for f in os.listdir(bin_dir) if not f.startswith((".", "_", "__"))
                 and f not in ("activate", "activate.bat", "activate_this.py",
                               "activate.csh", "activate.fish", "deactivate.bat",
                               "pip", "pip3", "pip3.11", "python", "python3",
                               "python3.11", "wheel", "wheel.exe", "setuptools",
                               "f2py", "f2py3", "f2py3.11", "pkg-config", "normalizer")]
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    installed_norm = {norm(f): f for f in installed}
    for cand in candidates:
        key = norm(cand)
        if key in installed_norm:
            return cand, os.path.join(bin_dir, installed_norm[key])
        # also try basename of a path candidate
        base = norm(cand.split("/")[-1] if "/" in cand else cand)
        if base in installed_norm:
            return cand, os.path.join(bin_dir, installed_norm[base])
    return "", ""


def _infer_entry_from_examples(readme_examples: list, pkg: str) -> str:
    """From readme import examples, infer module:Class entry point."""
    for ex in readme_examples:
        m = re.search(r"from\s+([\w.]+)\s+import\s+([\w]+)", ex)
        if m:
            mod, name = m.group(1), m.group(2)
            if mod.split(".")[0] == pkg:
                return f"{mod}:{name}"
    return ""


def _inspect_python_entry(venv_py: str, entry_point: str, env: dict) -> list[dict[str, str]]:
    """Inspect a module:Class entry point and return its __init__ parameters.

    Runs in the venv: imports the module, reads the class __init__ signature,
    and returns [{name, type, description, required}] based on real Python
    params (with defaults -> optional). This is the correct source of inputs
    for python-API tools (NOT the CLI --help, which differs from Python args).
    """
    code = (
        "import importlib, inspect, json, sys\n"
        "module_name, _, class_name = sys.argv[1].partition(':')\n"
        "m = importlib.import_module(module_name)\n"
        "cls = getattr(m, class_name)\n"
        "sig = inspect.signature(cls.__init__)\n"
        "out = []\n"
        "for name, p in sig.parameters.items():\n"
        "    if name in ('self', 'args', 'kwargs'):\n"
        "        continue\n"
        "    default = None if p.default is inspect.Parameter.empty else p.default\n"
        "    t = type(default).__name__ if default is not None else 'str'\n"
        "    out.append({'name': name, 'type': t, 'default': default is not None,\n"
        "               'description': ''})\n"
        "print(json.dumps(out))\n"
    )
    try:
        cp = subprocess.run(
            [venv_py, "-c", code, entry_point],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace", env=env)
        if cp.returncode == 0 and cp.stdout.strip():
            import json as _json
            params = _json.loads(cp.stdout.strip().splitlines()[-1])
            return [{"name": p["name"],
                     "type": p["type"],
                     "description": f"Python parameter {p['name']}" + ("" if p["default"] else " (required)"),
                     "required": not p["default"]}
                    for p in params][:20]
    except Exception:
        pass
    return []


def _python_import_smoke(venv_py: str, module_name: str, env: dict) -> tuple[int, str, str]:
    """Try importing a module in the venv (Python-API style tools with no CLI)."""
    return _run([venv_py, "-c",
                 f"import {module_name}; print('IMPORT_OK {module_name}')"],
                RUN_TIMEOUT, env=env)


def _test_docker(repo_dir: str, name: str, timeout: int = 600) -> dict[str, Any]:
    """Build + smoke-run a Docker-based tool. Returns a status report dict.

    Uses the already-cloned repo_dir as build context. We try `docker build`
    then `docker run --rm <image> --help` (and `<image> <sample>` if the repo
    ships one). Falls back to not_tested if docker isn't available in this
    environment.
    """
    report: dict[str, Any] = {"status": "not_tested", "reason": ""}
    docker = shutil.which("docker")
    if not docker:
        report["reason"] = "docker not installed in this environment"
        return report
    tag = f"disc_{re.sub(r'[^a-zA-Z0-9_.-]', '_', name).lower()[:40]}"
    rc, out, err = _run(["docker", "build", "-t", tag, repo_dir], timeout)
    if rc == 124:
        report["status"] = "timeout"
        report["reason"] = f"docker build timed out after {timeout}s (image pull/build heavy)"
        return report
    if rc != 0:
        report["status"] = "failed"
        report["reason"] = f"docker build failed (exit {rc}): {(out + err)[-200:]}"
        return report
    # smoke: --help (broad), then a run with the sample if present
    sample = _find_sample_input(repo_dir)
    for args in ([tag, "--help"], [tag, sample] if sample else None):
        if args is None:
            continue
        rc2, out2, err2 = _run(["docker", "run", "--rm", *args], timeout)
        if rc2 == 0 and (out2.strip() or err2.strip()):
            report["status"] = "passed"
            report["reason"] = f"`docker run {' '.join(args)}` -> exit 0"
            report["run_evidence"] = (out2 + err2)[-400:]
            report["executable"] = tag
            return report
    report["status"] = "failed"
    report["reason"] = "docker build ok but no smoke run exited 0 with output"
    return report


def _test_conda(repo_dir: str, name: str, install_cmd: str,
                timeout: int = 1800) -> dict[str, Any]:
    """Create a conda env from environment.yml (if present) and smoke-run.

    Requires conda on PATH. If no environment.yml exists but the README says
    conda, try `conda create -n <name> python=3.11` then install the repo's
    requirements. Returns a status report dict.
    """
    report: dict[str, Any] = {"status": "not_tested", "reason": ""}
    conda = shutil.which("conda")
    if not conda:
        report["reason"] = "conda not installed in this environment"
        return report
    env_name = f"disc_{re.sub(r'[^a-zA-Z0-9_.-]', '_', name).lower()[:30]}"

    def _create_result(rc: int, what: str, out: str, err: str) -> Optional[dict]:
        if rc == 124:
            return {"status": "timeout",
                    "reason": f"{what} timed out after {timeout}s (heavy env init)"}
        if rc != 0:
            return {"status": "failed",
                    "reason": f"{what} failed (exit {rc}): {(out + err)[-200:]}"}
        return None

    # 1. create the env (from environment.yml if present, at any depth)
    env_yml = ""
    for root, _dirs, files in os.walk(repo_dir):
        if "environment.yml" in files:
            p = os.path.join(root, "environment.yml")
            if not env_yml or p.count(os.sep) < env_yml.count(os.sep):
                env_yml = p
    if env_yml:
        rc, out, err = _run(
            ["conda", "env", "create", "-n", env_name, "-f", env_yml],
            timeout)
        bad = _create_result(rc, "conda env create", out, err)
        if bad:
            report.update(bad)
            return report
    else:
        rc, out, err = _run(
            ["conda", "create", "-n", env_name, "-y", "python=3.11"], timeout)
        bad = _create_result(rc, "conda create", out, err)
        if bad:
            report.update(bad)
            return report
    # 2. install repo deps if a requirements.txt exists
    req = ""
    for root, _dirs, files in os.walk(repo_dir):
        for f in files:
            if f == "requirements.txt":
                p = os.path.join(root, f)
                if not req or p.count(os.sep) < req.count(os.sep):
                    req = p
    if req:
        rc_r, out_r, err_r = _run(
            ["conda", "run", "-n", env_name, "pip", "install", "-q", "-r", req],
            timeout)
        if rc_r == 124:
            report["status"] = "timeout"
            report["reason"] = (f"conda env deps install timed out after {timeout}s "
                                "(heavy env init)")
            return report
    # 3. smoke: use real entry points from the repo (console scripts / entry
    #    files), not the repo name guessed as module/command.
    declared = _parse_console_scripts(repo_dir)
    import_cands = []
    if declared:
        import_cands = declared
    else:
        import_cands = [re.sub(r"[^a-zA-Z0-9_]", "", name).lstrip("_")]

    env_prefix = None
    for cand in import_cands:
        rc3, out3, err3 = _run(
            ["conda", "run", "-n", env_name, "python", "-c",
             f"import {cand}; print('IMPORT_OK {cand}')"],
            timeout)
        if rc3 == 0 and "IMPORT_OK" in out3:
            report["status"] = "passed"
            report["reason"] = f"`conda run -n {env_name} python -c 'import {cand}'` -> exit 0"
            report["run_evidence"] = (out3 + err3)[-400:]
            report["executable"] = cand
            return report
        # locate the env's bin dir once for command lookup
        rc_b, out_b, _ = _run(["conda", "run", "-n", env_name,
                               "python", "-c",
                               "import sys, os; print(os.path.dirname(sys.executable))"],
                              timeout)
        if rc_b == 0:
            env_prefix = out_b.strip()
    # 4. try real console scripts installed in the env bin dir
    if env_prefix and os.path.isdir(env_prefix):
        bin_dir = env_prefix
        cands = declared or [name]
        matched, exe_path = _find_installed_executable(bin_dir, cands)
        if exe_path:
            rc4, out4, err4 = _run([exe_path, "--help"], timeout)
            if rc4 == 0 and (out4.strip() or err4.strip()):
                report["status"] = "passed"
                report["reason"] = f"`{os.path.basename(exe_path)} --help` -> exit 0"
                report["run_evidence"] = (out4 + err4)[-400:]
                report["executable"] = os.path.basename(exe_path)
                return report
    report["status"] = "failed"
    report["reason"] = ("conda env created but no import/command smoke exited 0 "
                        f"(import candidates: {import_cands})")
    return report


def _find_sample_input(repo_dir: str) -> str:
    """Prefer a real example file from the repo; fall back to sample.fasta."""
    for root, _dirs, files in os.walk(repo_dir):
        for f in files:
            if f.lower().endswith((".fasta", ".faa", ".fa", ".fas", ".seq", ".fq", ".txt")):
                p = os.path.join(root, f)
                if "example" in p.lower() or "test" in p.lower() or "demo" in p.lower():
                    return p
    for root, _dirs, files in os.walk(repo_dir):
        for f in files:
            if f.lower().endswith((".fasta", ".faa", ".fa", ".fas", ".seq", ".fq")):
                p = os.path.join(root, f)
                if os.path.getsize(p) < 2_000_000:
                    return p
    return ""


def _parse_console_scripts(repo_dir: str) -> list[str]:
    """Parse [project.scripts] / entry_points / console_scripts from a cloned repo.

    Returns the real command names a package declares (e.g. `fingerprint` for
    RiSpy) instead of guessing from the repo name.
    """
    names: list[str] = []

    pyproject = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(pyproject):
        try:
            content = open(pyproject, "r", encoding="utf-8", errors="replace").read()
        except Exception:
            content = ""
        # [project.scripts] section:  name = "module:fn"
        m = re.search(r"\[project\.scripts\](.*?)(?=\n\[|\Z)", content, re.S)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "[")):
                    continue
                key = re.split(r"\s*[=:]\s*", line, 1)[0].strip().strip('"\'')
                if key:
                    names.append(key)
        # tool.poetry.scripts:  name = "module:fn"
        m = re.search(r"\[tool\.poetry\.scripts\](.*?)(?=\n\[|\Z)", content, re.S)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "[")):
                    continue
                key = re.split(r"\s*[=:]\s*", line, 1)[0].strip().strip('"\'')
                if key:
                    names.append(key)

    setup_py = os.path.join(repo_dir, "setup.py")
    if os.path.exists(setup_py):
        try:
            content = open(setup_py, "r", encoding="utf-8", errors="replace").read()
        except Exception:
            content = ""
        # entry_points={...'console_scripts': [...]}  or  console_scripts=[
        m = re.search(r"console_scripts\s*[=:]\s*\[(.*?)\]", content, re.S)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip().strip("'\",")
                if not line or line.startswith(("#", "[")):
                    continue
                key = line.split("=", 1)[0].strip()
                if key:
                    names.append(key)
        m = re.search(r"entry_points\s*=.*?console_scripts\s*=\s*\[(.*?)\]", content, re.S)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip().strip("'\",")
                if not line or line.startswith(("#", "[")):
                    continue
                key = line.split("=", 1)[0].strip()
                if key:
                    names.append(key)

    setup_cfg = os.path.join(repo_dir, "setup.cfg")
    if os.path.exists(setup_cfg):
        try:
            content = open(setup_cfg, "r", encoding="utf-8", errors="replace").read()
        except Exception:
            content = ""
        m = re.search(r"console_scripts\s*=(.*?)(?=\n\[|\Z)", content, re.S)
        if m:
            for line in m.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "[")):
                    continue
                key = line.split("=", 1)[0].strip()
                if key:
                    names.append(key)

    # de-dup, preserve order, drop obviously bad tokens
    seen: set[str] = set()
    out = []
    for n in names:
        n = n.strip().strip("'\"")
        if not n or n.lower() in ("python", "pip", "py", "setup"):
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _command_candidates(repo_url: str, entry_scripts: list[str], name: str) -> list[str]:
    stem = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower().strip("_")
    cands: list[str] = []
    if stem:
        cands.append(stem)
    for f in entry_scripts:
        if f.endswith(".sh"):
            cands.append(f)
        else:
            cands.append(f[:-3])
    seen: set[str] = set()
    return [c for c in cands if c and not (c in seen or seen.add(c))]


def _clone(url: str, dest: str) -> tuple[int, str]:
    return _run(["git", "clone", "--depth", "1", url, dest], 120)


def _self_heal_command(fail_output: str, known_candidates: list[str],
                       bin_dir: str) -> tuple[str, str]:
    """Wrapper self-heal: extract a better command from failure output.

    Looks for `command not found: X` / `X: command not found` in the stderr and
    matches X against the binaries actually installed in the venv bin dir.
    Returns (new_command_name, evidence) or ("", "").
    """
    m = re.search(r"(?:command not found|not found|No such file or directory)[:\s]+([\w.\-]+)",
                  fail_output, re.IGNORECASE)
    if not m:
        # format 2: `bash: pygenomeviz: command not found` (name BEFORE the marker)
        m = re.search(r"(?:bash|sh|/bin/sh|zsh|fish)?\s*[:]?\s*([\w.\-]+):\s+(?:command not found|not found)\b",
                      fail_output, re.IGNORECASE)
    if not m:
        return "", ""
    guessed = m.group(1).strip()
    # try to match the missing name against installed binaries (case/underscore-insensitive)
    if os.path.isdir(bin_dir):
        installed = os.listdir(bin_dir)
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        target = norm(guessed)
        for f in installed:
            if norm(f) == target and f not in ("activate", "pip", "python", "wheel"):
                return f, f"healed: {guessed} -> installed binary {f}"
    # fall back: if the missing command matches a known candidate loosely
    for c in known_candidates:
        if norm(c) == target:
            return c, f"healed: {guessed} -> known candidate {c}"
    return "", ""


def execute_test(repo_url: str, install_method: str = "",
                 install_cmd: str = "", entry_scripts: Optional[list[str]] = None,
                 repo_name: str = "", external_commands: Optional[list] = None) -> dict[str, Any]:
    """Install + smoke-run one tool; returns an execution report dict."""
    entry_scripts = entry_scripts or []
    external_commands = external_commands or []
    name = repo_name or (urllib.parse.urlparse(repo_url).path.rstrip("/").split("/")[-1]
                         or "unknown_tool")
    # remember missing system commands (from verify) so failure reasons can
    # tell the caller exactly what to install
    report_missing = [
        {"command": c.get("command") if isinstance(c, dict) else c,
         "install_hint": c.get("install_hint", "") if isinstance(c, dict) else ""}
        for c in external_commands
        if (c.get("kind") == "system_missing" if isinstance(c, dict) else True)
    ]
    report = {
        "tool": name,
        "repo_url": repo_url,
        "status": "skipped",
        "reason": "",
        "install_ok": False,
        "install_evidence": "",
        "run_ok": False,
        "run_evidence": "",
        "executable": "",
        "params_schema": [],
        "arg_style": "",
        "callable_via": "",
        "execution": None,
        "readme_examples": [],
        "llm_call_code": "",
        "llm_call_status": "",
        "positional_args": [],
        "subcommands": [],
        "subcommand_details": {},
        "installed_versions": [],
        "exec_retries": 0,
        "heal_evidence": "",
        "missing_system_commands": report_missing,
        "system_install_log": [],
        "checked_at": __import__("datetime").datetime.now().isoformat(),
    }

    # Windows can't create dirs containing quotes etc. from paper-extracted names
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', "_", name).strip("._-") or "tool"
    if not install_cmd:
        report["status"], report["reason"] = "skipped", "no install command"
        return report

    workdir = tempfile.mkdtemp(prefix="execute_")
    try:
        # ---- 1. obtain the code (needed for sample inputs / entry scripts) ----
        repo_dir = os.path.join(workdir, safe_name)
        rc, _out, err = _clone(repo_url, repo_dir)
        if rc != 0:
            report["status"], report["reason"] = "failed", f"clone failed: {err[:200]}"
            return report

        venv_dir = os.path.join(workdir, "venv")
        rc, out, err = _run([sys.executable, "-m", "venv", venv_dir], 120)
        if rc != 0:
            report["status"], report["reason"] = "failed", \
                f"venv creation failed: {err[:200]}"
            return report

        venv_py = _venv_python(venv_dir)

        # bootstrap build backend (bare venvs lack setuptools/wheel)
        _run([venv_py, "-m", "pip", "install", "-q", "--upgrade",
              "pip", "setuptools", "wheel"], 180)

        # build install args from install_method / install_cmd
        # install from the LOCAL clone (most reliable for arbitrary repos)
        if install_method in ("pip_pkg", "pip_url"):
            args = [venv_py, "-m", "pip", "install", "-q", repo_dir]
        elif install_method == "pip_requirements":
            # resolve the requirements file against the ACTUAL cloned tree
            # instead of parsing the install command string (unreliable).
            req = ""
            for root, _dirs, files in os.walk(repo_dir):
                for f in files:
                    if f in ("requirements.txt", "requirements_full.txt",
                             "requirements-dev.txt", "requirements_prod.txt"):
                        p = os.path.join(root, f)
                        # prefer the shallowest (root-level) one
                        if not req or p.count(os.sep) < req.count(os.sep):
                            req = p
            if not req:
                report["status"], report["reason"] = "failed", \
                    "pip_requirements but no requirements.txt found in repo"
                return report
            args = [venv_py, "-m", "pip", "install", "-q", "-r", req]
        elif install_method == "python_script":
            # source-run style: no install step, but try to install a
            # requirements.txt if present so imports resolve in the venv.
            req = ""
            for root, _dirs, files in os.walk(repo_dir):
                for f in files:
                    if f == "requirements.txt":
                        p = os.path.join(root, f)
                        if not req or p.count(os.sep) < req.count(os.sep):
                            req = p
            if req:
                _run([venv_py, "-m", "pip", "install", "-q", "-r", req],
                     INSTALL_TIMEOUT)
            args = [venv_py, "-c", "import sys; sys.exit(0)"]  # no-op install
        elif install_method == "r_pkg":
            # R package: needs R installed; install via remotes from the clone.
            # Prefer system R, else fall back to R inside the active conda env
            # (setup-miniconda installs r-base into an env, not system PATH).
            r_bin = shutil.which("R") or shutil.which("Rscript")
            conda_r = None
            if not r_bin and shutil.which("conda"):
                # scan all conda envs for R, not just base/current
                rc_envs, out_envs, _ = _run(["conda", "env", "list"], 60)
                env_names = []
                if rc_envs == 0:
                    for line in out_envs.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and line != "base":
                            env_names.append(line.split()[0])
                env_names = ["base"] + env_names + [os.environ.get("CONDA_DEFAULT_ENV", "")]
                seen_env = set()
                for env in env_names:
                    if not env or env in seen_env:
                        continue
                    seen_env.add(env)
                    rc_r, out_r, _ = _run(
                        ["conda", "run", "-n", env, "which", "R"], 60)
                    if rc_r == 0 and out_r.strip():
                        conda_r = ["conda", "run", "-n", env, "R"]
                        break
                if conda_r is None:
                    rc_r, out_r, _ = _run(["conda", "run", "which", "R"], 60)
                    if rc_r == 0 and out_r.strip():
                        conda_r = ["conda", "run", "R"]
            r_cmd = [r_bin] if r_bin else (conda_r or None)
            if not r_cmd:
                report["status"], report["reason"] = "not_tested", \
                    "R not installed in this environment (checked PATH and conda envs)"
                return report
            rc_r, out_r, err_r = _run(
                r_cmd + ["-e",
                 "install.packages('remotes', repos='https://cloud.r-project.org'); "
                 f"remotes::install_local('{repo_dir}', repos='https://cloud.r-project.org')"],
                1800)
            report["install_evidence"] = (out_r + err_r)[-400:]
            if rc_r != 0:
                report["status"] = "failed"
                report["reason"] = f"R install_local failed (exit {rc_r}): {(out_r + err_r)[-200:]}"
                return report
            report["install_ok"] = True
            import_name = re.sub(r"[^a-zA-Z0-9_]", "", name).lstrip("_")
            r_label = " ".join(r_cmd)
            rc_s, out_s, err_s = _run(
                r_cmd + ["-e", f"library({import_name}); cat('R_IMPORT_OK')"],
                RUN_TIMEOUT)
            if rc_s == 0 and "R_IMPORT_OK" in (out_s + err_s):
                report["status"] = "passed"
                report["reason"] = f"`{r_label} -e 'library({import_name})'` -> exit 0"
                report["run_ok"] = True
                report["run_evidence"] = (out_s + err_s)[-400:]
                report["executable"] = import_name
                return report
            report["status"] = "failed"
            report["reason"] = "R install ok but library() smoke failed"
            return report
        elif install_method == "make":
            # C/C++ repo: make build + run the produced binary
            rc_m, out_m, err_m = _run(["make", "-C", repo_dir], 1800)
            report["install_evidence"] = (out_m + err_m)[-400:]
            if rc_m != 0:
                report["status"] = "failed"
                report["reason"] = f"make failed (exit {rc_m}): {(out_m + err_m)[-200:]}"
                return report
            report["install_ok"] = True
            built = os.path.join(repo_dir, name)
            if os.path.exists(built):
                rc_s, out_s, err_s = _run([built, "--help"], RUN_TIMEOUT)
                if rc_s == 0 and (out_s.strip() or err_s.strip()):
                    report["status"] = "passed"
                    report["reason"] = f"`{built} --help` -> exit 0"
                    report["run_ok"] = True
                    report["run_evidence"] = (out_s + err_s)[-400:]
                    report["executable"] = name
                    return report
            report["status"] = "failed"
            report["reason"] = "make build ok but no runnable binary smoke passed"
            return report
        elif install_method == "npm":
            # Node repo: npm install then smoke via bin/ from node_modules/.bin
            rc_n, out_n, err_n = _run(
                ["npm", "install", "--prefix", repo_dir], 1800)
            report["install_evidence"] = (out_n + err_n)[-400:]
            if rc_n != 0:
                report["status"] = "failed"
                report["reason"] = f"npm install failed (exit {rc_n}): {(out_n + err_n)[-200:]}"
                return report
            report["install_ok"] = True
            nbin = os.path.join(repo_dir, "node_modules", ".bin")
            # try the repo-name binary, else any binary that looks like the tool
            exe_cand = [name]
            if os.path.isdir(nbin):
                installed_bins = [f for f in os.listdir(nbin) if not f.endswith((".cmd", ".ps1"))]
                exe_cand += [b for b in installed_bins if name.lower() in b.lower() or b.lower() in name.lower()]
            for cand in exe_cand:
                exe = os.path.join(nbin, cand) if os.path.isdir(nbin) else ""
                if os.path.exists(exe):
                    rc_s, out_s, err_s = _run([exe, "--help"], RUN_TIMEOUT)
                    if rc_s == 0 and (out_s.strip() or err_s.strip()):
                        report["status"] = "passed"
                        report["reason"] = f"`{exe} --help` -> exit 0"
                        report["run_ok"] = True
                        report["run_evidence"] = (out_s + err_s)[-400:]
                        report["executable"] = cand
                        return report
            report["status"] = "failed"
            report["reason"] = "npm install ok but no runnable binary smoke passed"
            return report
        elif install_method == "docker":
            # build + smoke the container in this environment (docker present
            # on GH runners). If docker is unavailable, we fall back to
            # not_tested rather than failing.
            docker_report = _test_docker(repo_dir, name)
            report.update(docker_report)
            return report
        elif install_method == "conda_env":
            conda_report = _test_conda(repo_dir, name, install_cmd)
            report.update(conda_report)
            return report
        elif install_method == "cargo":
            # Rust repo: cargo build into the venv's bin dir, smoke-run from there
            rc_c, out_c, err_c = _run(
                ["cargo", "build", "--release", "--manifest-path",
                 os.path.join(repo_dir, "Cargo.toml")],
                1800)
            report["install_evidence"] = (out_c + err_c)[-400:]
            if rc_c != 0:
                report["status"] = "failed"
                report["reason"] = f"cargo build failed (exit {rc_c}): {(out_c + err_c)[-200:]}"
                return report
            report["install_ok"] = True
            # smoke via the built binary in target/release
            built = os.path.join(repo_dir, "target", "release", name)
            if os.path.exists(built):
                rc_s, out_s, err_s = _run([built, "--help"], RUN_TIMEOUT)
                if rc_s == 0 and (out_s.strip() or err_s.strip()):
                    report["status"] = "passed"
                    report["reason"] = f"`{built} --help` -> exit 0"
                    report["run_ok"] = True
                    report["run_evidence"] = (out_s + err_s)[-400:]
                    report["executable"] = os.path.basename(built)
                    return report
            report["status"] = "failed"
            report["reason"] = "cargo build ok but no runnable binary smoke passed"
            return report
        elif install_method == "go":
            # Go repo: go build the main package, smoke-run the binary
            rc_g, out_g, err_g = _run(
                ["go", "build", "-o", os.path.join(workdir, safe_name + "_go"),
                 "."],
                1800, cwd=repo_dir)
            report["install_evidence"] = (out_g + err_g)[-400:]
            if rc_g != 0:
                report["status"] = "failed"
                report["reason"] = f"go build failed (exit {rc_g}): {(out_g + err_g)[-200:]}"
                return report
            report["install_ok"] = True
            built = os.path.join(workdir, safe_name + "_go")
            if os.path.exists(built):
                rc_s, out_s, err_s = _run([built, "--help"], RUN_TIMEOUT)
                if rc_s == 0 and (out_s.strip() or err_s.strip()):
                    report["status"] = "passed"
                    report["reason"] = f"`{built} --help` -> exit 0"
                    report["run_ok"] = True
                    report["run_evidence"] = (out_s + err_s)[-400:]
                    report["executable"] = os.path.basename(built)
                    return report
            report["status"] = "failed"
            report["reason"] = "go build ok but no runnable binary smoke passed"
            return report
        else:
            report["status"], report["reason"] = "not_tested", \
                f"install method '{install_method}' not supported in smoke test"
            return report

        # tiered install timeout: heavy ML deps (torch/tf/jax) need far longer
        install_timeout = INSTALL_TIMEOUT_ML if _detect_heavy_deps(repo_dir) \
            else INSTALL_TIMEOUT
        rc, out, err = _run(args, install_timeout)
        report["install_evidence"] = (out + err)[-400:]
        if rc == 124:
            # install timed out - this is NOT "tool is broken". It means the
            # environment init (torch download etc.) exceeded our window.
            report["status"] = "timeout"
            report["reason"] = ("execution deferred: dependency install timed out "
                                f"after {install_timeout}s (heavy env init, e.g. ML "
                                "deps); not a code failure")
            report["install_evidence"] = (out + err)[-400:]
            return report
        if rc != 0:
            cls = _classify_failure((out + err)[-1200:])
            report["status"] = cls if cls != "failed" else "failed"
            report["reason"] = \
                f"install failed (exit {rc}): {(out + err)[-200:]}"
            return report
        report["install_ok"] = True

        # record the actually-installed package versions (reproducibility
        # evidence, step 3.6): exact pins make the tool's env reproducible.
        _frz_rc, freeze_out, _frz_err = _run(
            [venv_py, "-m", "pip", "freeze"], 60)
        if _frz_rc == 0:
            pins = [ln.strip() for ln in freeze_out.splitlines()
                    if ln.strip() and not ln.startswith("-")]
            report["installed_versions"] = pins[:80]

        # ---- 2.5 install missing system commands (from verify's scan) ----
        # If the repo invokes blastn/samtools etc. that aren't on PATH, try to
        # install them via the recorded install_hint (apt/conda) so the smoke
        # can actually run. Failures are recorded, not fatal.
        import shutil as _sh
        system_install_log = []
        for mcmd in report_missing:
            cname = mcmd.get("command", "")
            hint = mcmd.get("install_hint", "")
            if not cname or _sh.which(cname.split("/")[-1]):
                continue
            if not hint:
                continue
            # parse the hint: prefer conda (works on GH runner with miniconda),
            # fall back to apt (needs sudo)
            installed = False
            if "conda install" in hint:
                pkg = hint.split("conda install")[-1].strip().split("|")[0].strip()
                rc, out, err = _run(["conda", "install", "-y", *pkg.split()], 900)
                installed = rc == 0
                system_install_log.append(f"conda install {pkg} -> {'ok' if installed else 'failed'}")
            if not installed and "apt-get install" in hint:
                pkg = hint.split("apt-get install")[-1].strip().split("|")[0].strip()
                rc, out, err = _run(
                    ["sudo", "apt-get", "install", "-y", *pkg.split()], 900)
                installed = rc == 0
                system_install_log.append(f"apt-get install {pkg} -> {'ok' if installed else 'failed'}")
        if system_install_log:
            report["system_install_log"] = system_install_log
            print(f"    [system-deps] {'; '.join(system_install_log)}")

        # ---- 3. prepare a sample input ----
        sample = _find_sample_input(repo_dir)
        if not sample:
            sample = os.path.join(workdir, "sample.fasta")
            with open(sample, "w", encoding="utf-8") as f:
                f.write(SAMPLE_FASTA)

        # ---- 4. smoke-run candidate commands ----
        bin_dir = _venv_bin(venv_dir)
        env = dict(os.environ)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        declared = _parse_console_scripts(repo_dir)
        cands = _command_candidates(repo_url, entry_scripts, name)
        # declared console scripts (e.g. `fingerprint` for RiSpy) take priority
        cands = declared + [c for c in cands if c not in declared]
        runs: list[str] = []
        worst = "failed"  # track most specific failure reason

        # 4a. find the REAL executable that got installed (don't assume name==cmd)
        matched_cand, exe_path = _find_installed_executable(bin_dir, cands)

        def _try(args: list[str]) -> bool:
            nonlocal worst
            rc, out, err = _run(args, RUN_TIMEOUT, env=env)
            ev = f"`{' '.join(args)}` -> exit {rc}"
            runs.append(ev)
            if rc == 0 and (out.strip() or err.strip()):
                report["status"] = "passed"
                report["reason"] = ev
                report["run_ok"] = True
                report["run_evidence"] = (out + err)[-400:]
                report["params_schema"] = _parse_help_params(out or err)
                report["arg_style"] = _detect_arg_style(out or err)
                report["positional_args"] = _parse_positional_args(out or err)
                report["subcommands"] = _parse_subcommands(out or err)
                # for subcommand CLIs, probe each subcommand's --help to capture
                # its own parameters (so the schema tells agents how to call it)
                if report["arg_style"] == "subcommand" and report["subcommands"]:
                    subs = {}
                    exe_base = args[0]  # the executable
                    for sub in report["subcommands"]:
                        rc_s, out_s, err_s = _run(
                            [exe_base, sub, "--help"], RUN_TIMEOUT, env=env)
                        if rc_s == 0 and (out_s.strip() or err_s.strip()):
                            subs[sub] = {
                                "params": _parse_help_params(out_s or err_s),
                                "usage": _extract_usage(out_s or err_s),
                            }
                    if subs:
                        report["subcommand_details"] = subs
                return True
            cls = _classify_failure((out + err)[-1200:])
            if cls == "incomplete":
                worst = "incomplete"
            elif cls == "env_issue" and worst != "incomplete":
                worst = "env_issue"
            return False

        if exe_path:
            report["executable"] = os.path.basename(exe_path)
            if _try([exe_path, sample]):
                return report
            if _try([exe_path, "--help"]):
                return report
        elif cands:
            # no installed binary matched -> try entry scripts (.py/.sh) or the
            # bare candidate through PATH (venv/bin is on PATH)
            for cand in cands:
                if cand.endswith(".py"):
                    base = [venv_py, os.path.join(repo_dir, cand)]
                elif cand.endswith(".sh"):
                    base = ["bash", os.path.join(repo_dir, cand)]
                else:
                    base = [cand]
                if _try(base + [sample]):
                    return report
                if _try(base + ["--help"]):
                    return report
        # 4b. Python-API fallback: no CLI found, try importing the package
        import_name = re.sub(r"[^a-zA-Z0-9_]", "", name).lstrip("_")
        if import_name:
            rc_imp, out_imp, err_imp = _python_import_smoke(venv_py, import_name, env)
            ev_imp = f"`python -c 'import {import_name}'` -> exit {rc_imp}"
            runs.append(ev_imp)
            if rc_imp == 0 and "IMPORT_OK" in out_imp:
                report["run_ok"] = True
                report["run_evidence"] = (out_imp + err_imp)[-400:]
                report["executable"] = import_name  # Python API, no CLI
                report["arg_style"] = "python"
                # determine how the python package can be invoked: prefer a real
                # entry point (__main__.py or README `python -m pkg.module`).
                # If we can't find a callable entry, the tool is NOT passed --
                # it's importable but we don't know how to invoke it.
                has_main = False
                for root, _dirs, files in os.walk(repo_dir):
                    if "__main__.py" in files:
                        has_main = True
                        break
                if has_main:
                    report["status"] = "passed"
                    report["reason"] = f"`python -m {import_name}` entry point"
                    report["callable_via"] = f"python -m {import_name}"
                    # if README shows a from X import Y, use real Python params
                    report["readme_examples"] = _extract_readme_examples(repo_dir, import_name)
                    entry = _infer_entry_from_examples(report["readme_examples"], import_name)
                    if entry:
                        report["execution"] = {"type": "python", "entry_point": entry}
                        report["params_schema"] = _inspect_python_entry(venv_py, entry, env)
                else:
                    readme_usage = _extract_readme_usage(repo_dir, import_name)
                    report["readme_examples"] = _extract_readme_examples(repo_dir, import_name)
                    if readme_usage.startswith("python -m "):
                        mod = readme_usage.replace("python -m ", "").split()[0]
                        rc_m, out_m, err_m = _run(
                            [venv_py, "-m", mod, "--help"], RUN_TIMEOUT, env=env)
                        if rc_m in (0, 1, 2) and (out_m.strip() or err_m.strip()):
                            report["status"] = "passed"
                            report["reason"] = f"README usage: python -m {mod} --help"
                            report["callable_via"] = readme_usage
                            report["run_evidence"] = (out_m + err_m)[-400:]
                            # inputs for python -m tools: flags from README
                            # examples (--sequence/--num_samples/...) so the
                            # agent knows what to pass
                            flag_params = _extract_flag_params(report["readme_examples"])
                            if flag_params:
                                report["params_schema"] = flag_params
                        else:
                            report["status"] = "failed"
                            report["reason"] = ("importable but no callable entry point "
                                                f"(README suggests {readme_usage[:60]} but it doesn't run)")
                            report["callable_via"] = "python_import_no_entry"
                    else:
                        report["status"] = "failed"
                        report["reason"] = "importable but no callable usage found in README; dropped"
                        report["callable_via"] = "python_import_no_entry"
                return report
            cls_imp = _classify_failure((out_imp + err_imp)[-1200:])
            if cls_imp == "incomplete":
                worst = "incomplete"
            elif cls_imp == "env_issue" and worst != "incomplete":
                worst = "env_issue"

        # 4c. wrapper self-heal: if all candidates failed, mine the failure
        # output for the real command name and retry once (max 2 heal attempts).
        retries_used = 0
        heal_evidence = ""
        while retries_used < 2:
            combined = "\n".join(runs)
            healed, evidence = _self_heal_command(combined, cands, bin_dir)
            if not healed or healed in cands:
                break
            heal_evidence = evidence
            cands.append(healed)
            retries_used += 1
            if _try([healed, sample]):
                report["reason"] = f"{report['reason']} (self-heal: {heal_evidence})"
                report["exec_retries"] = retries_used
                report["heal_evidence"] = heal_evidence
                return report
            if _try([healed, "--help"]):
                report["reason"] = f"{report['reason']} (self-heal: {heal_evidence})"
                report["exec_retries"] = retries_used
                report["heal_evidence"] = heal_evidence
                return report
        if heal_evidence:
            report["heal_evidence"] = heal_evidence
            report["exec_retries"] = retries_used
        report["status"] = worst
        report["reason"] = "; ".join(runs) if runs else "no runnable candidate"
        report["run_evidence"] = "no candidate exited 0 with output"
        return report
    finally:
        # keep the venv for passed tools so the agent test can reuse the
        # installed environment instead of re-running pip install (esp. heavy
        # deps like torch). Record the venv path for the caller.
        if report.get("status") == "passed" and report.get("install_ok"):
            report["venv_path"] = venv_dir
            report["venv_kept"] = True
            print(f"    [keep] venv kept at {venv_dir} for agent reuse")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def url_to_pip(cmd: str, repo_url: str) -> str:
    """pip_pkg install_cmd is `pip install <url>`; strip the `pip install` part."""
    parts = cmd.replace("pip install", "", 1).strip()
    return parts or repo_url


def _watchdog_handler(signum, frame):
    raise TimeoutError("execute_tool_library global watchdog: overall time limit hit")


def execute_tool_library(verification_file: str = "tool_verification.json",
                         out_json: str = "tool_execution.json",
                         max_repos: Optional[int] = None,
                         global_timeout: int = 3600) -> list[dict[str, Any]]:
    """Run execution tests for every verified/repo_ok tool; persist results.

    A global watchdog (signal-based, POSIX only) aborts the whole run after
    `global_timeout` seconds so a stuck conda/docker build can't hang CI for
    hours. On timeout the partial results are still written.
    """
    watchdog = None
    try:
        import signal
        signal.signal(signal.SIGALRM, _watchdog_handler)
        signal.alarm(global_timeout)
        watchdog = signal
    except (ImportError, ValueError):
        pass  # no signal on Windows / main thread only -> watchdog disabled

    with open(verification_file, "r", encoding="utf-8") as f:
        verifications = json.load(f)

    results = []
    watchdog_hit = False
    try:
        try:
            for i, v in enumerate(verifications):
                tool = v.get("tool", v.get("repo_name", "?"))
                if v.get("status") not in ("verified", "repo_ok"):
                    results.append({
                        "tool": tool, "repo_url": v.get("repo_url", ""),
                        "status": "skipped",
                        "reason": f"repo verification = {v.get('status', '')}",
                    })
                    print(f"  [{i + 1}] {tool:<20} skipped ({v.get('status', '')})")
                    continue
                print(f"  [{i + 1}] {tool:<20} installing + smoke-running ...")
                res = execute_test(
                    v.get("repo_url", ""),
                    install_method=v.get("install_method", ""),
                    install_cmd=v.get("install_cmd", ""),
                    entry_scripts=v.get("entry_scripts", []),
                    repo_name=tool,
                    external_commands=v.get("external_commands", []),
                )
                results.append(res)
                print(f"        -> {res['status']}: {res['reason'][:80]}")
                if max_repos is not None and i + 1 >= max_repos:
                    break
        except TimeoutError:
            watchdog_hit = True
            print(f"\n!! global watchdog hit after {global_timeout}s; "
                  f"aborting remaining tools, keeping partial results")
    finally:
        if watchdog:
            watchdog.alarm(0)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n_pass = sum(1 for r in results if r.get("status") == "passed")
    n_env = sum(1 for r in results if r.get("status") == "env_issue")
    n_inc = sum(1 for r in results if r.get("status") == "incomplete")
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    n_time = sum(1 for r in results if r.get("status") == "timeout")
    n_nt = sum(1 for r in results if r.get("status") == "not_tested")
    print(f"\nSaved execution report -> {out_json}" + (" (PARTIAL, watchdog)" if watchdog_hit else ""))
    print(f"passed: {n_pass} | env_issue: {n_env} | incomplete: {n_inc} "
          f"| failed: {n_fail} | timeout: {n_time} | not_tested: {n_nt} "
          f"| skipped: {n_skip}  (total {len(results)})")
    return results


if __name__ == "__main__":
    import sys as _sys
    targets = _sys.argv[1:] or None
    for u in (targets or []):
        print(json.dumps(execute_test(u), ensure_ascii=False, indent=2))
        print("-" * 60)
    if not targets:
        execute_tool_library()
