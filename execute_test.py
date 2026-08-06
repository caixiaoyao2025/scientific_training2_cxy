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

INSTALL_TIMEOUT = 300          # seconds per pip install
RUN_TIMEOUT = 60               # seconds per smoke run
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


def _venv_python(venv_dir: str) -> str:
    return os.path.join(venv_dir, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv_dir, "bin", "python")


def _venv_bin(venv_dir: str) -> str:
    return os.path.join(venv_dir, "Scripts") if os.name == "nt" \
        else os.path.join(venv_dir, "bin")


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
        "checked_at": __import__("datetime").datetime.now().isoformat(),
    }

    # conda_env installs are too heavy for a smoke test: skip without cloning
    if install_method == "conda_env":
        report["status"], report["reason"] = "skipped", \
            "conda env install too heavy for smoke test"
        return report

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
            req = os.path.join(repo_dir, install_cmd.rsplit("/", 1)[-1].split()[-1])
            if not os.path.exists(req):
                req = os.path.join(repo_dir, "requirements.txt")
            args = [venv_py, "-m", "pip", "install", "-q", "-r", req]
        elif install_method == "conda_env":
            report["status"], report["reason"] = "skipped", \
                "conda env install too heavy for smoke test"
            return report
        else:
            report["status"], report["reason"] = "skipped", \
                f"install method '{install_method}' not supported in smoke test"
            return report

        rc, out, err = _run(args, INSTALL_TIMEOUT)
        report["install_evidence"] = (out + err)[-400:]
        if rc != 0:
            report["status"], report["reason"] = "failed", \
                f"install failed (exit {rc}): {(out + err)[-200:]}"
            return report
        report["install_ok"] = True

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
        cands = _command_candidates(repo_url, entry_scripts, name)
        runs: list[str] = []
        for cand in cands:
            exe = os.path.join(bin_dir, cand + (".exe" if os.name == "nt" else ""))
            if os.path.exists(exe):
                args = [exe, sample]
            elif cand.endswith(".py"):
                args = [venv_py, os.path.join(repo_dir, cand), sample]
            elif cand.endswith(".sh"):
                args = ["bash", os.path.join(repo_dir, cand), sample]
            else:
                args = [cand, sample]
            rc, out, err = _run(args, RUN_TIMEOUT, env=env)
            ev = f"`{' '.join(args)}` -> exit {rc}"
            runs.append(ev)
            if rc == 0 and (out.strip() or err.strip()):
                report["status"] = "passed"
                report["reason"] = ev
                report["run_ok"] = True
                report["run_evidence"] = (out + err)[-400:]
                return report
        report["status"] = "failed"
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
    print(f"\nSaved execution report -> {out_json}")
    print(f"passed: {n_pass} / {len(results)}")
    return results


if __name__ == "__main__":
    import sys as _sys
    targets = _sys.argv[1:] or None
    for u in (targets or []):
        print(json.dumps(execute_test(u), ensure_ascii=False, indent=2))
        print("-" * 60)
    if not targets:
        execute_tool_library()
