"""Auto-fix failed tool tests by asking a Westlake LLM to diagnose.

When step 3.6 marks a tool failed / env_issue, this script asks an LLM whether
the failure is (a) a bug in *our* pipeline code (verify_repo.py / execute_test.py
install detection, grading, smoke logic) that we can fix, or (b) a problem with
the discovered tool itself. For (a) it proposes a concrete code change, applies
it, re-runs that single tool, and keeps the fix only if the tool now passes.

Safety:
  - Only *.py files in the pipeline are editable (never the tool's repo).
  - Every applied patch is validated by re-running execute_test for that tool.
  - If the re-run still fails, the file is restored from git (rollback).
  - No LLM change is committed unless it improves at least one tool.

Usage:
  WESTLAKE_API_KEY=... python auto_fix.py [tool_names...]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request

BASE_URL = os.environ.get("WESTLAKE_BASE_URL", "https://hpc-api.westlake.edu.cn/v1")
MODEL = os.environ.get("WESTLAKE_MODEL", "deepseek")
API_KEY = os.environ.get("WESTLAKE_API_KEY", "")

# files the LLM is allowed to propose edits to
EDITABLE = {"verify_repo.py", "execute_test.py", "discovery_to_registry.py",
            "clean.py", "convert.py"}

EDITABLE_SNIPPETS = {
    "verify_repo.py": ("_find_requirements / install-method inference / grading",
                       "reads repo files, decides language + install method + repo_ok/unverified"),
    "execute_test.py": ("install branches / _classify_failure / smoke loop / _find_installed_executable",
                        "installs into a venv, classifies failures, smoke-runs commands"),
    "discovery_to_registry.py": ("inputs schema / description assembly",
                                 "turns verified tools into registry entries"),
}


def _llm_call(prompt: str, max_tokens: int = 1200) -> str:
    if not API_KEY:
        return ""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
    )
    last_err = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            import time
            time.sleep(2 ** attempt)
    print(f"  !! LLM call failed after retries: {last_err[:200]}")
    return ""


def _build_prompt(tool: dict[str, object], relevant_files: dict[str, str]) -> str:
    return f"""You are debugging an automated bioinformatics tool-discovery pipeline.

A discovered tool failed its execution test. Decide: is the failure caused by a bug
in OUR pipeline code (which we can and should fix), or by the tool repo itself?

Tool: {tool.get('tool')}
Repo: {tool.get('repo_url')}
Install method: {tool.get('install_method')}
Status: {tool.get('status')}
Reason: {tool.get('reason')}
Run evidence: {tool.get('run_evidence')}

Our pipeline code that decides install / classify / smoke:
{json.dumps(relevant_files, ensure_ascii=False, indent=1)}

Rules:
- ONLY propose edits to files in: {sorted(EDITABLE)}
- If the failure is the tool's own problem (bad repo, missing deps, etc.) answer NO_FIX.
- If our pipeline mis-detected the install method / mis-classified / smoke logic is wrong,
  answer with EXACTLY this JSON:
  {{"file": "execute_test.py", "description": "what is wrong", "old": "exact existing code",
    "new": "replacement code"}}
- "old" must match code currently in the file, or the patch is skipped.
- Keep the change minimal and safe. No imports of tools' packages.
"""
    # NOTE: we inline relevant file contents into the prompt so the LLM can
    # propose exact old/new replacements.


def _apply_patch(filepath: str, old: str, new: str) -> bool:
    if os.path.basename(filepath) not in EDITABLE:
        return False
    try:
        src = open(filepath, "r", encoding="utf-8").read()
    except OSError:
        return False
    if old not in src:
        print(f"    patch skipped: 'old' not found in {filepath}")
        return False
    open(filepath, "w", encoding="utf-8").write(src.replace(old, new, 1))
    # sanity: must still parse
    try:
        import ast
        ast.parse(open(filepath, "r", encoding="utf-8").read())
    except SyntaxError as exc:
        # rollback
        subprocess.run(["git", "checkout", "--", filepath], check=False)
        print(f"    patch rejected: syntax error -> rolled back ({exc})")
        return False
    return True


def _rerun_single(tool: dict[str, object]) -> dict:
    """Re-run execute_test for one tool from its verification record."""
    from execute_test import execute_test as _et
    try:
        return _et(
            tool.get("repo_url", ""),
            install_method=tool.get("install_method", ""),
            install_cmd=tool.get("install_cmd", ""),
            entry_scripts=tool.get("entry_scripts", []),
            repo_name=tool.get("tool", ""),
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"rerun crashed: {exc}"}


def main() -> None:
    if not API_KEY:
        print("WESTLAKE_API_KEY not set - skipping auto-fix")
        return
    exec_file = "tool_execution.json"
    if not os.path.exists(exec_file):
        print("no tool_execution.json")
        return
    results = json.load(open(exec_file, encoding="utf-8"))

    # merge with verification records to get install metadata
    verify = {}
    if os.path.exists("tool_verification.json"):
        for r in json.load(open("tool_verification.json", encoding="utf-8")):
            verify[r.get("tool", "")] = r

    targets = [t for t in results if t.get("status") in ("failed", "env_issue", "incomplete")]
    if not targets:
        print("no fixable failures")
        return

    report = []
    for i, t in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {t.get('tool')} ({t.get('status')})")
        v = verify.get(t.get("tool", ""), {})
        merged = {**v, **t}
        relevant = {}
        for f, (what, _d) in EDITABLE_SNIPPETS.items():
            try:
                relevant[f] = open(f, "r", encoding="utf-8").read()
            except OSError:
                continue
        # keep prompt bounded: send the relevant sections only if file is huge
        prompt = _build_prompt(merged, relevant)
        if len(prompt) > 120_000:
            prompt = prompt[:120_000]
        answer = _llm_call(prompt)
        if not answer:
            report.append({"tool": t.get("tool"), "fix": "no_llm_response"})
            continue
        if answer.strip().upper().startswith("NO_FIX"):
            report.append({"tool": t.get("tool"), "fix": "no_fix"})
            print(f"    -> LLM says no fix")
            continue
        try:
            m = re.search(r"\{.*\}", answer, re.S)
            if not m:
                report.append({"tool": t.get("tool"), "fix": "no_json"})
                continue
            patch = json.loads(m.group(0))
            filep = patch.get("file", "")
            old = patch.get("old", "")
            new = patch.get("new", "")
            if not (filep and old and new):
                report.append({"tool": t.get("tool"), "fix": "incomplete_patch"})
                continue
            print(f"    proposing edit to {filep}")
            if not _apply_patch(filep, old, new):
                report.append({"tool": t.get("tool"), "fix": "patch_rejected"})
                continue
            # verify by re-running this one tool
            rerun = _rerun_single(merged)
            print(f"    rerun -> {rerun.get('status')}")
            if rerun.get("status") == "passed":
                report.append({"tool": t.get("tool"), "fix": "applied_and_passed",
                               "file": filep, "description": patch.get("description", "")})
            else:
                subprocess.run(["git", "checkout", "--", filep], check=False)
                report.append({"tool": t.get("tool"), "fix": "rolled_back_not_passing",
                               "rerun_status": rerun.get("status")})
        except Exception as exc:  # noqa: BLE001
            report.append({"tool": t.get("tool"), "fix": f"error: {exc}"})

    with open("auto_fix_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    n_ok = sum(1 for r in report if r.get("fix") == "applied_and_passed")
    print(f"\nauto-fix done: {n_ok} applied+passed / {len(report)} failures examined")


if __name__ == "__main__":
    main()
