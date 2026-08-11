"""Repository verification for auto-discovered tools.

Auto-discovery (convert.py) guesses tool metadata (command name, install
method, type) from the repo URL alone. That produces placeholder entries
like `command: tsamp {{input_file}}` that were never confirmed against the
real repository. This module closes the loop.

Verification strategy uses a *blobless shallow clone* (`--filter=blob:none
--no-checkout`): only the tree and small text blobs are downloaded on demand,
so data-heavy repos (e.g. model weights) verify in seconds instead of hanging
the pipeline on a huge download.

Status is graded:
  - verified : an invocable command exists on PATH (installed & callable)
  - repo_ok  : repo is structurally healthy (license and/or deps present and
               an entry script / command candidate exists) but not installed
  - unverified: clone failed, no entry point, or structurally broken
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional

CLONE_TIMEOUT = 90          # seconds; blobless clone of a small repo is fast
CHECK_TIMEOUT = 30          # seconds per command probe
REQUIREMENTS_FILES = ("requirements.txt", "requirements_full.txt",
                      "environment.yml", "setup.py", "setup.cfg", "pyproject.toml")
CONTAINER_FILES = ("Dockerfile", "docker-compose.yml", "docker-compose.yaml",
                   "containerfile", "environment.docker.yml")
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md",
                 "LICENSE.rst", "LICENSES/")
ENTRY_HINTS = ("predict.py", "main.py", "cli.py", "__main__.py", "run.sh",
               "run_*.py", "*.py")


@dataclass
class VerifyResult:
    repo_url: str
    repo_name: str = ""
    status: str = "unverified"      # verified | repo_ok | unverified
    reason: str = ""
    command: str = ""
    command_evidence: str = ""
    install_method: str = "pip_url"
    install_cmd: str = ""
    language: str = ""
    has_requirements: bool = False
    requirements_paths: list = field(default_factory=list)
    has_container: bool = False
    container_files: list = field(default_factory=list)
    readme_hint: str = ""       # conda | docker | "" (from README text)
    external_commands: list = field(default_factory=list)  # system deps
    declared_packages: list = field(default_factory=list)  # top-level pip deps
    has_license: bool = False
    license_path: str = ""
    license_text_snippet: str = ""
    entry_scripts: list = field(default_factory=list)
    clone_error: str = ""
    checked_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "repo_name": self.repo_name,
            "status": self.status,
            "reason": self.reason,
            "command": self.command,
            "command_evidence": self.command_evidence,
            "install_method": self.install_method,
            "install_cmd": self.install_cmd,
            "language": self.language,
            "has_requirements": self.has_requirements,
            "requirements_paths": self.requirements_paths,
            "has_container": self.has_container,
            "container_files": self.container_files,
            "readme_hint": self.readme_hint,
            "external_commands": self.external_commands,
            "declared_packages": self.declared_packages,
            "has_license": self.has_license,
            "license_path": self.license_path,
            "license_text_snippet": self.license_text_snippet,
            "entry_scripts": self.entry_scripts,
            "clone_error": self.clone_error,
            "checked_at": self.checked_at,
        }


def _parse_owner_repo(repo_url: str) -> Optional[tuple[str, str]]:
    try:
        path = urllib.parse.urlparse(repo_url).path.rstrip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            return None
        return parts[0], parts[1]
    except Exception:
        return None


def _repo_name_from_url(repo_url: str) -> str:
    parsed = _parse_owner_repo(repo_url)
    return parsed[1] if parsed else (repo_url.rstrip("/").split("/")[-1] or "unknown_tool")


def _run(args: list[str], timeout: int, cwd: Optional[str] = None) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, check=False,
            timeout=timeout, cwd=cwd, encoding="utf-8", errors="replace",
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except FileNotFoundError:
        return 127, "", f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


class BloblessRepo:
    """Read-only view of a repo backed by a blobless shallow clone."""

    def __init__(self, repo_dir: str):
        self.dir = repo_dir

    def list_files(self) -> list[str]:
        rc, out, err = _run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                            30, cwd=self.dir)
        if rc != 0:
            return []
        return out.splitlines()

    def show(self, path: str) -> str:
        rc, out, err = _run(["git", "show", f"HEAD:{path}"], CHECK_TIMEOUT, cwd=self.dir)
        return out if rc == 0 else ""


def _find_requirements(files: list[str]) -> list[str]:
    found = []
    for f in files:
        if f.startswith(".") or "/" in f and f.split("/")[0].startswith("."):
            continue
        base = f.split("/")[-1]
        # requirements/env files up to 2 levels deep (many repos keep them in
        # a subfolder, e.g. atacseq/requirements.txt); setup.py/pyproject are
        # only meaningful at the root so require depth <= 1.
        if base in REQUIREMENTS_FILES and f.count("/") <= 2:
            found.append(f)
    # keep setup.py/pyproject at root as the strongest install signal
    return sorted(found, key=lambda f: (f.count("/"), f))


def _find_license(files: list[str]) -> tuple[bool, str]:
    for f in files:
        base = f.split("/")[-1].upper()
        if base.startswith("LICENSE") or base in ("COPYING", "COPYING.MD"):
            return True, f
    return False, ""


def _find_entry_scripts(files: list[str]) -> list[str]:
    out = []
    for f in files:
        if f.startswith("test") or "/test" in f or "example" in f or "benchmark" in f:
            continue
        base = f.split("/")[-1]
        if base in ("predict.py", "main.py", "cli.py", "__main__.py", "run.sh") \
                and f.count("/") <= 2 and not f.startswith("data/"):
            out.append(f)
    return out[:10]


def _scan_external_commands(repo: "BloblessRepo", files: list[str],
                            max_files: int = 60) -> list[str]:
    """Environment grounding: find external system commands a repo invokes.

    Scans Python sources with AST for subprocess.run/call/Popen, os.system,
    os.popen and shutil.which first-string-argument calls, plus common `sys`
    pattern. These are commands (samtools, blastn, gmx, ...) that must exist on
    the SYSTEM, not in the venv -- exactly the deps the smoke test can't
    provide. Only the first string-literal argument is captured.
    """
    import ast
    cmds: set[str] = set()
    py_files = [f for f in files if f.endswith(".py") and f.count("/") <= 3
                and not f.startswith("test") and "/test" not in f
                and "/example" not in f and "/benchmark" not in f]
    py_files = py_files[:max_files]
    for f in py_files:
        src = repo.show(f)
        if not src or len(src) > 400_000:
            continue
        src = src.lstrip("\ufeff")  # strip BOM so ast.parse doesn't fail
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                fname = node.func.attr
                if fname in ("run", "call", "Popen", "check_call", "check_output"):
                    # subprocess.run(["cmd", ...]) or subprocess.run("cmd ...")
                    arg = node.args[0] if node.args else None
                    cmd = _first_cmd_literal(arg)
                    if cmd:
                        cmds.add(cmd)
                elif fname in ("system", "popen") and isinstance(node.func.value, ast.Name) \
                        and node.func.value.id == "os":
                    cmd = _first_cmd_literal(node.args[0] if node.args else None)
                    if cmd:
                        cmds.add(cmd)
                elif fname == "which" and isinstance(node.func.value, ast.Name) \
                        and node.func.value.id == "shutil":
                    cmd = _first_cmd_literal(node.args[0] if node.args else None)
                    if cmd:
                        cmds.add(cmd)
    # drop shell-ish / non-command tokens, keep only plausible executables
    out = []
    for c in sorted(cmds):
        c = c.strip().strip("'\"")
        if not c or not re.match(r"^[a-zA-Z0-9_.+\-/]+$", c):
            continue
        if c in ("ls", "cat", "echo", "mkdir", "rm", "cp", "mv", "grep",
                 "sed", "awk", "head", "tail", "sort", "wc", "find", "python",
                 "python3", "bash", "sh", "cd", "export", "source", "true",
                 "false", "pwd", "touch", "chmod", "chown", "sleep", "date",
                 "uname", "which", "env", "tee", "printf", "cut", "tr", "uniq"):
            continue
        if c.startswith(("-", "~", "/usr", "/bin", "/etc", "{")) or " " in c:
            continue
        out.append(c)
    return out[:30]


def _parse_declared_packages(repo: "BloblessRepo", files: list[str]) -> list[str]:
    """Parse top-level pip dependencies from requirements.txt / pyproject.toml.

    Returns declared package names (without version pins / extras / markers)
    so the tool's install contract can list what to install.
    """
    pkgs: set[str] = set()
    req_files = [f for f in files
                 if f.split("/")[-1] in ("requirements.txt", "requirements_full.txt",
                                          "requirements-dev.txt", "requirements_prod.txt")]
    for f in req_files[:5]:
        text = repo.show(f)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-", "--")):
                continue
            if line.startswith("-r") or line.startswith("--requirement"):
                continue
            # strip version specifiers:  numpy>=1.20  ->  numpy
            name = line.split("[")[0].split(";")[0].split("=")[0].split("~")[0] \
                       .split("<")[0].split(">")[0].split("!")[0].strip()
            if name and not name.startswith((".", "/", "{")):
                pkgs.add(name.lower().replace("_", "-"))
    # pyproject.toml [project] dependencies
    for f in files:
        if f.split("/")[-1] == "pyproject.toml" and f.count("/") <= 1:
            text = repo.show(f)
            m = re.search(r"\[project\]\s*dependencies\s*=\s*\[(.*?)\]", text, re.S)
            if m:
                for line in m.group(1).splitlines():
                    line = line.strip().rstrip(",").strip('"\'')
                    if not line:
                        continue
                    name = line.split("[")[0].split(";")[0].split("=")[0] \
                               .split("~")[0].split("<")[0].split(">")[0].split("!")[0].strip()
                    if name and name not in pkgs:
                        pkgs.add(name.lower().replace("_", "-"))
    return sorted(pkgs)[:40]


def _first_cmd_literal(arg) -> str:
    import ast as _ast
    if arg is None:
        return ""
    if isinstance(arg, _ast.Constant) and isinstance(arg.value, str):
        return arg.value.split()[0] if arg.value.strip() else ""
    if isinstance(arg, _ast.List):
        first = arg.elts[0] if arg.elts else None
        if isinstance(first, _ast.Constant) and isinstance(first.value, str):
            return first.value
    if isinstance(arg, _ast.JoinedStr):
        parts = [p for p in arg.values if isinstance(p, _ast.Constant)]
        if parts:
            return str(parts[0].value).split()[0]
    return ""


def _command_candidates(repo_name: str, files: list[str], entry_scripts: list[str]) -> list[str]:
    cands: list[str] = []
    stem = re.sub(r"[^a-zA-Z0-9_]", "_", repo_name).lower().strip("_")
    if stem:
        cands.append(stem)
    for f in entry_scripts:
        if f.endswith(".sh"):
            cands.append(f)
        else:
            cands.append(f[:-3])  # predict.py -> predict
    seen: set[str] = set()
    out = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _probe_command(cmd: str) -> tuple[bool, str]:
    exe = shutil.which(cmd)
    if exe:
        return True, f"found on PATH: {exe}"
    rc, _, err = _run([cmd, "--help"], CHECK_TIMEOUT)
    if rc in (0, 1, 2):
        return True, f"`{cmd} --help` ran (exit {rc})"
    return False, f"`{cmd} --help` failed (exit {rc}): {err[:200]}"


def verify_repo(repo_url: str, work_dir: Optional[str] = None,
                keep_dir: bool = False) -> VerifyResult:
    """Blobless-clone repo_url and gather verification evidence."""
    name = _repo_name_from_url(repo_url)
    res = VerifyResult(repo_url=repo_url, repo_name=name)
    res.checked_at = __import__("datetime").datetime.now().isoformat()

    owner_repo = _parse_owner_repo(repo_url)
    if owner_repo is None:
        res.status, res.reason = "unverified", f"not a github.com URL: {repo_url}"
        return res

    tmp = None
    try:
        if work_dir:
            repo_dir = work_dir
        else:
            tmp = tempfile.mkdtemp(prefix="verify_")
            repo_dir = os.path.join(tmp, name)
            rc, _, err = _run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--no-checkout", f"https://github.com/{owner_repo[0]}/{owner_repo[1]}",
                 repo_dir],
                CLONE_TIMEOUT)
            if rc != 0:
                res.status, res.reason = "unverified", "clone failed"
                res.clone_error = err[:300]
                return res

        repo = BloblessRepo(repo_dir)
        files = repo.list_files()
        if not files:
            res.status, res.reason = "unverified", "clone produced empty tree"
            return res

        # ---- structural evidence ----
        res.requirements_paths = _find_requirements(files)
        res.has_requirements = bool(res.requirements_paths)
        res.has_container = bool(res.container_files)
        res.container_files = [f for f in files
                               if f.split("/")[-1] in CONTAINER_FILES
                               and f.count("/") <= 1]
        res.has_container = bool(res.container_files)
        res.has_license, res.license_path = _find_license(files)
        if res.has_license:
            res.license_text_snippet = repo.show(res.license_path).strip().replace("\n", " ")[:120]
        res.entry_scripts = _find_entry_scripts(files)
        res.declared_packages = _parse_declared_packages(repo, files)

        # language hint
        bases = set(f.split("/")[-1] for f in files)
        if "setup.py" in bases or "pyproject.toml" in bases or "setup.cfg" in bases:
            res.language = "Python"
        elif "Cargo.toml" in bases:
            res.language = "Rust"
        elif "go.mod" in bases:
            res.language = "Go"
        elif "package.json" in bases:
            res.language = "Node"
        elif "DESCRIPTION" in bases or any(f.startswith("R/") for f in files):
            # R package (DESCRIPTION + R/ + man/ layout)
            res.language = "R"
        elif any(base in bases for base in ("Makefile", "CMakeLists.txt", "configure")):
            res.language = "C"

        # ---- README install hint (conda/docker text, no dep file needed) ----
        # Many paper repos document `conda create ... && python script.py` or
        # `docker run ...` in the README with no requirements/env file. Catch
        # that so those repos aren't misclassified as "no install command".
        readme_hint = ""
        for f in files:
            base = f.split("/")[-1].lower()
            if base.startswith(("readme", "installation", "quickstart", "usage")):
                text = repo.show(f).lower()
                if "conda " in text or "conda install" in text or "conda create" in text:
                    readme_hint = "conda"
                elif "docker run" in text or "docker build" in text:
                    readme_hint = readme_hint or "docker"
                if readme_hint:
                    break
        res.readme_hint = readme_hint

        # ---- install command (evidence-based) ----
        if any(p.split("/")[-1] in ("setup.py", "pyproject.toml", "setup.cfg")
               for p in res.requirements_paths):
            res.install_method, res.install_cmd = "pip_pkg", f"pip install {repo_url}"
        elif any(p.endswith("environment.yml") for p in res.requirements_paths):
            # conda env file is NOT pip-installable (can live in a subfolder)
            env_yml = next((p for p in res.requirements_paths
                            if p.endswith("environment.yml")), "environment.yml")
            res.install_method, res.install_cmd = (
                "conda_env", f"conda env create -f {env_yml}")
        elif res.has_container:
            # docker-based repo: only deployable via container image
            res.install_method, res.install_cmd = "docker", \
                f"docker build -t {res.repo_name} . && docker run {res.repo_name}"
        elif readme_hint == "docker":
            res.install_method, res.install_cmd = "docker", \
                "docker build (per README) && docker run"
        elif readme_hint == "conda":
            res.install_method, res.install_cmd = "conda_env", \
                "conda env create (per README), then python entry script"
        elif res.requirements_paths:
            res.install_method, res.install_cmd = (
                "pip_requirements", f"pip install -r {res.requirements_paths[0]}")
        elif res.language == "Rust":
            res.install_method, res.install_cmd = "cargo", f"cargo install --git {repo_url}"
        elif res.language == "Go":
            res.install_method, res.install_cmd = "go", f"go install {repo_url}@latest"
        elif res.language == "R":
            # R package: install via remotes/devtools from the local clone
            res.install_method, res.install_cmd = "r_pkg", \
                f"R -e 'install.packages(\"remotes\"); remotes::install_local(\"{repo_url}\")'"
        elif res.language == "C":
            # C/C++ repo: make / cmake build
            res.install_method, res.install_cmd = "make", \
                f"make -C {repo_url} && ./{res.repo_name}"
        elif res.language == "Node":
            # Node repo: npm install (may build native deps)
            res.install_method, res.install_cmd = "npm", \
                f"npm install --prefix {repo_url}"
        elif res.entry_scripts:
            # source-run style repo: has entry scripts but no dependency file.
            # Install method 'python_script' -> execute_test runs them directly.
            res.install_method, res.install_cmd = (
                "python_script", " ".join(res.entry_scripts[:3]))

        # ---- command probe ----
        cands = _command_candidates(name, files, res.entry_scripts)
        found_cmd, evidence = "", ""
        for c in cands:
            ok, ev = _probe_command(c)
            if ok:
                found_cmd, evidence = c, ev
                break
        res.command, res.command_evidence = found_cmd, evidence

        # ---- grading ----
        has_code = (res.entry_scripts or res.has_requirements or res.has_container
                    or bool(res.language))
        if found_cmd:
            res.status = "verified"
            res.reason = f"command '{found_cmd}' invocable"
        elif has_code:
            # a recognized language (Rust/Go/R/C/Node...) with its build file is
            # a valid installable repo even without requirements.txt. A repo with
            # code but no LICENSE is still repo_ok (license is recorded, not
            # required); a bare LICENSE with NO code signal is docs-only -> excluded.
            res.status = "repo_ok"
            res.reason = ("repo healthy (entry scripts: "
                          + (", ".join(res.entry_scripts) if res.entry_scripts else "none")
                          + "; deps: "
                          + (", ".join(res.requirements_paths[:2]) if res.requirements_paths else "")
                          + (" docker" if res.has_container else "")
                          + (f" lang={res.language}" if res.language else "")
                          + (" license" if res.has_license else " no-license")
                          + "); command not installed")
        else:
            res.status = "unverified"
            res.reason = "no entry point / dependency / language signals (docs/data-only repo)"
        return res
    finally:
        if tmp and os.path.isdir(tmp) and not keep_dir:
            shutil.rmtree(tmp, ignore_errors=True)


def verify_tool_library(tool_library: list[dict[str, Any]],
                        out_json: str = "tool_verification.json",
                        max_repos: Optional[int] = None) -> list[dict[str, Any]]:
    """Verify every tool in the standardized library; persist results."""
    results = []
    for i, tool in enumerate(tool_library):
        url = tool.get("source", {}).get("github", "")
        if not url:
            results.append({"tool": tool.get("name"), "github": "",
                            "status": "unverified", "reason": "no github url"})
            continue
        res = verify_repo(url)
        d = res.to_dict()
        d["tool"] = tool.get("name", "")
        results.append(d)
        print(f"  [{i + 1}] {d['tool']:<20} {d['status']:<12} {d['reason'][:60]}")
        if max_repos is not None and i + 1 >= max_repos:
            break
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved verification report -> {out_json}")
    n_ok = sum(1 for r in results if r.get("status") in ("verified", "repo_ok"))
    print(f"verified/repo_ok: {n_ok} / {len(results)}")
    return results


if __name__ == "__main__":
    import sys
    urls = sys.argv[1:] or [
        "https://github.com/YangLab-BUPT/tsAMP",
        "https://github.com/samarendra-pani/giggles",
    ]
    for u in urls:
        r = verify_repo(u)
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        print("-" * 60)
