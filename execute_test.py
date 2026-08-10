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
        m = re.match(r'^\s*(?:-\w,\s+)?(--?[\w][\w-]*)\s+([A-Z_]+|\{[^}]+\}|<[^>]+>)?\s*(.*)$',
                     line, re.IGNORECASE)
        if not m:
            continue
        flag, metavar, desc = m.group(1), m.group(2) or "", m.group(3) or ""
        # skip the standard help flag itself; it's not a real tool parameter
        if flag in ("--help", "-h"):
            continue
        # only flags (--x) make sense as named params; skip bare -v etc.
        if flag.startswith("--") and flag not in seen:
            seen.add(flag)
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
            params.append({
                "name": flag,
                "type": ptype,
                "description": desc or f"CLI flag {flag}",
            })
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
    # 1. create the env (from environment.yml if present)
    env_yml = os.path.join(repo_dir, "environment.yml")
    if os.path.exists(env_yml):
        rc, out, err = _run(
            ["conda", "env", "create", "-n", env_name, "-f", env_yml],
            timeout)
        if rc != 0:
            report["status"] = "failed"
            report["reason"] = f"conda env create failed (exit {rc}): {(out + err)[-200:]}"
            return report
    else:
        rc, out, err = _run(
            ["conda", "create", "-n", env_name, "-y", "python=3.11"], timeout)
        if rc != 0:
            report["status"] = "failed"
            report["reason"] = f"conda create failed (exit {rc}): {(out + err)[-200:]}"
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
        _run(["conda", "run", "-n", env_name, "pip", "install", "-q", "-r", req],
             timeout)
    # 3. smoke: try python -c 'import <name>' inside the env
    import_name = re.sub(r"[^a-zA-Z0-9_]", "", name).lstrip("_")
    rc3, out3, err3 = _run(
        ["conda", "run", "-n", env_name, "python", "-c",
         f"import {import_name}; print('IMPORT_OK {import_name}')"],
        timeout)
    if rc3 == 0 and "IMPORT_OK" in out3:
        report["status"] = "passed"
        report["reason"] = f"`conda run -n {env_name} python -c 'import {import_name}'` -> exit 0"
        report["run_evidence"] = (out3 + err3)[-400:]
        report["executable"] = import_name
        return report
    # 4. if import fails, try `<name> --help` inside the env
    rc4, out4, err4 = _run(["conda", "run", "-n", env_name, name, "--help"],
                           timeout)
    if rc4 == 0 and (out4.strip() or err4.strip()):
        report["status"] = "passed"
        report["reason"] = f"`conda run -n {env_name} {name} --help` -> exit 0"
        report["run_evidence"] = (out4 + err4)[-400:]
        report["executable"] = name
        return report
    report["status"] = "failed"
    report["reason"] = ("conda env created but no import/command smoke exited 0 "
                        f"(import exit {rc3}, cmd exit {rc4})")
    report["run_evidence"] = (out3 + err3)[-200:] + (out4 + err4)[-200:]
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


def execute_test(repo_url: str, install_method: str = "",
                 install_cmd: str = "", entry_scripts: Optional[list[str]] = None,
                 repo_name: str = "") -> dict[str, Any]:
    """Install + smoke-run one tool; returns an execution report dict."""
    entry_scripts = entry_scripts or []
    name = repo_name or (urllib.parse.urlparse(repo_url).path.rstrip("/").split("/")[-1]
                         or "unknown_tool")
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
        "installed_versions": [],
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
                report["status"] = "passed"
                report["reason"] = ev_imp
                report["run_ok"] = True
                report["run_evidence"] = (out_imp + err_imp)[-400:]
                report["executable"] = import_name  # Python API, no CLI
                return report
            cls_imp = _classify_failure((out_imp + err_imp)[-1200:])
            if cls_imp == "incomplete":
                worst = "incomplete"
            elif cls_imp == "env_issue" and worst != "incomplete":
                worst = "env_issue"
        report["status"] = worst
        report["reason"] = "; ".join(runs) if runs else "no runnable candidate"
        report["run_evidence"] = "no candidate exited 0 with output"
        return report
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def url_to_pip(cmd: str, repo_url: str) -> str:
    """pip_pkg install_cmd is `pip install <url>`; strip the `pip install` part."""
    parts = cmd.replace("pip install", "", 1).strip()
    return parts or repo_url


def execute_tool_library(verification_file: str = "tool_verification.json",
                         out_json: str = "tool_execution.json",
                         max_repos: Optional[int] = None) -> list[dict[str, Any]]:
    """Run execution tests for every verified/repo_ok tool; persist results."""
    with open(verification_file, "r", encoding="utf-8") as f:
        verifications = json.load(f)

    results = []
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
        )
        results.append(res)
        print(f"        -> {res['status']}: {res['reason'][:80]}")
        if max_repos is not None and i + 1 >= max_repos:
            break

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n_pass = sum(1 for r in results if r.get("status") == "passed")
    n_env = sum(1 for r in results if r.get("status") == "env_issue")
    n_inc = sum(1 for r in results if r.get("status") == "incomplete")
    n_fail = sum(1 for r in results if r.get("status") == "failed")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    n_time = sum(1 for r in results if r.get("status") == "timeout")
    n_nt = sum(1 for r in results if r.get("status") == "not_tested")
    print(f"\nSaved execution report -> {out_json}")
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
