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

# Scalar-type hints for _infer_scalar_type. The NAME is checked first with
# word boundaries (underscores/dashes are separators: num_samples -> "num
# samples"); only if the name has NO signal does the DESCRIPTION get consulted,
# and then with a reduced token set so prose words like "batch"/"samples" never
# override an output_dir positional into an int.
_INT_TYPE_HINTS = ("num", "count", "number", "threads", "thread", "seed",
                   "port", "size", "depth", "samples", "index", "length",
                   "width", "height", "batch", "iteration", "iterations",
                   "rounds", "times", "kmer", "offset", "step", "rank",
                   "level", "iter")
_FLOAT_TYPE_HINTS = ("rate", "ratio", "fraction", "prob", "pvalue",
                     "threshold", "alpha", "gamma", "score", "cutoff",
                     "distance", "resolution", "coverage", "quality")
_PATH_TYPE_HINTS = ("path", "file", "dir", "directory", "folder", "fasta",
                    "fastq", "bam", "cram", "bed", "vcf", "gtf", "gff", "pdb",
                    "sam", "seq", "input", "output", "out", "in", "fa", "fna",
                    "ref", "genome", "reference", "db", "index", "database")
_DESC_INT_HINTS = ("number of", "integer", "count of")
_DESC_FLOAT_HINTS = ("rate", "fraction", "ratio", "threshold", "float",
                     "probability")
_DESC_PATH_HINTS = ("path", "file", "directory", "fasta", "fastq", "bam",
                    "bed", "vcf", "gtf", "gff", "pdb", "sam")


def _infer_scalar_type(name: str, desc: str = "") -> str:
    """Best-effort scalar JSON type for a CLI param from its name/description.

    Auto-discovery sees only `--num_samples NUM_SAMPLES Number of samples`; the
    type is not always in the help (argparse prints no types). Name heuristics
    (num/count/seed -> int, rate/fraction -> float, path/file/dir -> path) are
    better than defaulting every positional to `path` -- a tool like bioemu
    takes `num_samples` as an INTEGER, and a string-typed schema makes the LLM
    emit `"1"` where the CLI (and _coerce_arguments) needs `1`.
    """
    ntext = f"{name}".lower().replace("-", " ").replace("_", " ")
    dtext = f"{desc}".lower().replace("-", " ").replace("_", " ")
    # 1) the NAME is authoritative (output_dir must stay a dir even if its
    #    description mentions "samples"/"batch")
    for kw in _INT_TYPE_HINTS:
        if re.search(rf"\b{re.escape(kw)}\b", ntext):
            return "int"
    for kw in _FLOAT_TYPE_HINTS:
        if re.search(rf"\b{re.escape(kw)}\b", ntext):
            return "float"
    for kw in _PATH_TYPE_HINTS:
        if re.search(rf"\b{re.escape(kw)}\b", ntext):
            return "path"
    # 2) description fallback (only when the name gave no signal, and only
    #    strong tokens -- avoids prose like "batch"/"samples" misfiring)
    for kw in _DESC_INT_HINTS:
        if re.search(rf"\b{re.escape(kw)}\b", dtext):
            return "int"
    for kw in _DESC_FLOAT_HINTS:
        if re.search(rf"\b{re.escape(kw)}\b", dtext):
            return "float"
    for kw in _DESC_PATH_HINTS:
        if re.search(rf"\b{re.escape(kw)}\b", dtext):
            return "path"
    return "string"


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


def _canonical_param_name(name: str) -> str:
    """Canonical input key for any CLI token: --sequence, -s, <SEQ>, SEQ,
    sequence -> 'sequence'. Dashes/underscores fold to '_' so the schema,
    the registry inputs, the command template and the argv renderer all agree
    on one key per parameter (no ONT_IN/ont_in duplicate classes)."""
    # SINGLE canonicalizer: the pipeline has exactly one parameter-name
    # definition (agent_connector.tool_spec.canonicalize_param_name), used by
    # discovery, registry, generator, LLM schema AND runner alike.
    from agent_connector.tool_spec import canonicalize_param_name  # noqa: PLC0415
    return canonicalize_param_name(name)


def _normalize_metavar(token: str) -> str:
    """Strip nargs/group decorations in the RIGHT order: ellipsis BEFORE the
    bracket/angle strip. `[INPUT]...` -> 'input' (the old code stripped the
    trailing char first and produced `INPUT]..` garbage), `<OUTPUT_DIR>` ->
    'output_dir', `{200M,500M}` -> '' (choices, not a metavar)."""
    s = str(token or "").strip()
    if not s:
        return ""
    if s.endswith("..."):
        s = s[:-3].rstrip()
    s = s.strip("<>[]{}")
    if s in ("", ".", "..."):
        return ""
    return s.lower().replace("-", "_")


def _required_flags_from_usage(help_output: str) -> set[str]:
    """Flags OUTSIDE square brackets in the usage line are required
    (argparse prints every required flag bare, optional ones wrapped in
    [...]: `kaptain [-h] --ont-in ONT_IN [ONT_IN ...] --db DB` ->
    {'--ont-in', '--db'}). ArgumentDefaultsHelpFormatter appends
    `(default: None)` to required nargs+ flags too, so `(default: ...)` in
    the description is NOT a reliable required signal -- the usage brackets
    are. Returns the flag strings (e.g. '-i', '--ont-in')."""
    if not help_output:
        return set()
    # NOTE: use the FULL usage line, NOT the 200-char truncated _extract_usage
    # (truncation can slice `[--subsampling ...` in half and flip it required).
    # argparse wraps the usage across indented continuation lines; absorb them
    # so required flags on line 2+ (`-o OUTPUT`) are seen. STOP at a blank
    # line or an unindented line -- otherwise the regex swallows the
    # description's `options:` block too (kaptain_help test).
    m = re.search(r"\busage:\s*((?:[^\n]+\n?)+)", help_output, re.IGNORECASE)
    if not m:
        return set()
    raw = m.group(1)
    lines = raw.splitlines()
    # first line: the usage itself. following lines: keep only INDENTED
    # continuation lines ([--threads THREADS] under the usage).
    usage_lines = [lines[0].strip()]
    for ln in lines[1:]:
        stripped = ln.strip()
        if not stripped:
            break
        if not ln.startswith(" ") and not ln.startswith("\t"):
            break
        usage_lines.append(stripped)
    usage = re.sub(r"\s+", " ", " ".join(usage_lines))
    # Split into bracket-groups vs outside-text with proper depth tracking
    # (usage can nest `[... [{...} ...]]`, which a regex `\[[^\]]*\]` split
    # mangles into a dangling `]` that looks like outside-text).
    required: set[str] = set()
    outside: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in usage:
        if ch == "[":
            if depth == 0 and buf:
                outside.append("".join(buf))
                buf = []
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
            if depth == 0:
                buf = []  # drop completed [...] group
        elif depth == 0:
            buf.append(ch)
    if buf:
        outside.append("".join(buf))
    for part in outside:
        for flag in re.findall(r"(?<![\w-])(-{1,2}[\w][\w-]*)", part):
            if flag not in ("-h", "--help"):
                required.add(flag)
    return required


def _extract_usage(help_output: str) -> str:
    """Extract the usage line from --help output.

    Sources, in priority order:
      - argparse/click:  `usage: prog [-h] --flag VAL ...`
      - fire:            `SYNOPSIS\\n    prog SEQUENCE NUM_SAMPLES <flags>`
    Returns just the arg spec (binary + args), truncated to 200 chars.
    """
    if not help_output:
        return ""
    m = re.search(r"\busage:\s*(.+)", help_output, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:200]
    m = re.search(r"(?:^|\n)\s*SYNOPSIS\s*\n\s*([^\n]+)", help_output,
                  re.IGNORECASE)
    if m:
        return m.group(1).strip()[:200]
    return ""


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
        line = line.rstrip()
        if not line.strip():
            continue
        # a command entry is "name<2+ spaces>description" (click/argparse
        # column layout). Continuation sentences like "The VITAP will update
        # database..." have single spaces between words and must NOT become
        # subcommands (vitap previously parsed "The" as a subcommand).
        mm = re.match(r"^\s*(\S+)\s{2,}\S", line)
        if not mm:
            continue
        name = mm.group(1)
        # clap/click auto-generate meta-subcommands that are not real tool
        # actions; probing their --help fails (or is pointless) and wrongly
        # flips subcommand_discovery_complete to False (run #36 bqtools:
        # 10/10 real subs probed OK but 'help' failed the completeness check)
        if name in ("help", "completion", "version"):
            continue
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
    # subcommand CLI: help lists "Commands:" with real entries. The old
    # requirement that the usage line contain a COMMAND token was too strict
    # (vitap's usage is `VITAP <command>` lowercase-wrapped and was missed,
    # so subcommand_details were never probed).
    has_commands = re.search(r"\bcommands?\s*:", low) is not None
    if has_commands and _parse_subcommands(help_output):
        return "subcommand"
    # positional: usage tokens after the binary that are real args (not
    # options, not [..], not pseudo tokens like options:/COMMAND)
    usage_line = _extract_usage(help_output)
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


def _merge_positionals(flags: list[dict[str, str]],
                       pos: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge flag params + positional args into ONE canonical params list.

    Both entries are deduped by canonical input key (--sequence == sequence).
    A positional keeps the parser's OWN required marker -- bare metavars
    (SEQUENCE, <INPUT>) are mandatory argv slots, but a bracketed `[INPUT]` /
    `[FILES ...]` is OPTIONAL, so it must NOT be forced required (a forced
    required would make the LLM schema demand an arg the CLI usage marks
    optional). This is THE single schema every downstream stage (registry
    command template, inputs, function schemas, argv renderer) reads, so a
    flag and its positional can never be registered as two inputs.
    """
    merged = list(flags)
    seen = {_canonical_param_name(p.get("name", "")) for p in flags}
    for pa in pos:
        key = _canonical_param_name(pa.get("name", ""))
        if not key or key in seen:
            continue
        entry = dict(pa)
        entry.setdefault("type", "path")
        entry.setdefault("description", f"Positional argument {pa.get('name', '')}")
        # parser's explicit marker wins; missing marker defaults to required
        # (a bare metavar is a mandatory argv slot, e.g. legacy/direct callers)
        entry["required"] = bool(entry.get("required", True))
        entry["positional"] = True
        merged.append(entry)
        seen.add(key)
    return merged


def _probe_cli(report: dict, exe: str, env: dict | None = None) -> dict | None:
    """Run `exe --help` and fill the report with arg_style / params / subcommands.

    Unlike the inline smoke code, this ALWAYS does subcommand discovery (probe
    each subcommand's own --help) so every install path -- cargo/go/npm/pip --
    produces the same rich schema. Returns the report on success, else None.
    """
    try:
        rc, out, err = _run([exe, "--help"], RUN_TIMEOUT, env=env)
    except Exception:
        return None
    combined = (out or err) if rc == 0 else None
    if not combined or not combined.strip():
        return None
    report["status"] = "passed"
    report["reason"] = f"`{exe} --help` -> exit {rc}"
    report["run_ok"] = True
    report["run_evidence"] = combined[-400:]
    report["params_schema"] = _parse_help_params(combined)
    report["arg_style"] = _detect_arg_style(combined)
    report["positional_args"] = _parse_positional_args(combined)
    report["subcommands"] = _parse_subcommands(combined)
    # SINGLE SOURCE OF TRUTH: canonical schema = flags + positionals merged
    # into one list (mirrors the python -m and subcommand branches below), so
    # discovery_to_registry builds command + inputs from ONE list and no
    # setdefault re-injection is needed (kaptain positional path).
    report["params_schema"] = _merge_positionals(
        report["params_schema"], report["positional_args"])
    if report["arg_style"] == "subcommand" and report["subcommands"]:
        subs = {}
        probed = 0
        ok = 0
        for sub in report["subcommands"]:
            probed += 1
            try:
                rc_s, out_s, err_s = _run([exe, sub, "--help"], RUN_TIMEOUT, env=env)
            except Exception:
                rc_s, out_s, err_s = 1, "", ""
            if rc_s == 0 and (out_s.strip() or err_s.strip()):
                ok += 1
                flags = _parse_help_params(out_s or err_s)
                pos = _parse_positional_args(out_s or err_s, skip_first=True)
                merged = _merge_positionals(flags, pos)
                subs[sub] = {
                    "params": merged,
                    "usage": _extract_usage(out_s or err_s),
                }
        report["subcommand_discovery_complete"] = (probed > 0 and ok == probed)
        if subs:
            report["subcommand_details"] = subs
    return report


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
             or os.environ.get("DEEPSEEK_MODEL") or "Doubao-Seed-Evolving")
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


def _parse_positional_args(help_output: str, skip_first: bool = False) -> list[dict[str, str]]:
    """Extract positional arguments from --help output.

    Sources:
      - argparse 'positional arguments:' block (real names + descriptions)
      - usage line:  pgv-blast [OPTIONS] seq1.gbk seq2.gbk -o out

    Returns [{name, type, description, positional, position}]. `position`
    is the order AMONG positional args (argv slot after the binary /
    subcommand), NOT the raw usage-token index -- skipped [OPTIONS] groups
    and option/value pairs must not inflate it, or the argv renderer would
    emit arguments in the wrong order.

    `skip_first` skips the first token (subcommand help lines, e.g. 'encode'
    in 'bqtools encode <INPUT> <OUTPUT>').
    """
    if not help_output:
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", help_output)  # strip ANSI
    # source 1: argparse 'positional arguments:' block (names + descriptions)
    desc_map: dict[str, str] = {}
    m = re.search(r"positional arguments?\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            line = line.rstrip()
            if not line.strip() or line.strip().startswith("-"):
                continue
            parts = line.split(None, 1)
            if parts:
                key = _normalize_metavar(parts[0])
                desc_map[key] = parts[1].strip() if len(parts) > 1 else ""

    # source 1b: fire 'POSITIONAL ARGUMENTS' block (fire.Fire help format).
    # Layout: `NAME` at indent 4, `Type: ...` / prose at indent 8. The header
    # has NO trailing colon (`POSITIONAL ARGUMENTS`), unlike argparse.
    if not m:
        m = re.search(r"POSITIONAL ARGUMENTS\s*:?\s*\n(.*?)(?:\n\s*\n|\Z)",
                      text, re.IGNORECASE | re.DOTALL)
        if m:
            pending = None
            for line in m.group(1).splitlines():
                line = line.rstrip()
                if not line.strip():
                    continue
                indent = len(line) - len(line.lstrip())
                if indent == 4 and not line.strip().startswith(("-", "Type")):
                    pending = _normalize_metavar(line.strip())
                elif indent >= 8 and pending is not None:
                    text2 = line.strip()
                    if text2.startswith("Type:"):
                        continue
                    if pending not in desc_map:
                        desc_map[pending] = text2
                    pending = None
                elif pending is not None:
                    if pending not in desc_map:
                        desc_map[pending] = line.strip()
                    pending = None

    # source 2: usage line. Accept wrapped usage (argparse indents
    # continuations); collapse the wrapping into one token stream. Stop at a
    # blank line OR an unindented line -- the options: block below usage is
    # indented too, so without the blank-line stop the regex swallows it and
    # option values (DIR_WORKING, THREADS...) leak in as fake positionals.
    m = re.search(r"(?:^|\n)\s*usage:\s*(.+?)(?=\n\s*\n|\n\S|\Z)",
                  text, re.IGNORECASE | re.DOTALL)
    usage = re.sub(r"\s+", " ", m.group(1)).strip() if m else _extract_usage(text)
    if not usage:
        return []
    parts = usage.split()
    if not parts:
        return []
    parts = parts[1:]  # drop the binary name
    if skip_first and parts and _is_probable_subcommand(parts[0]):
        parts = parts[1:]  # drop the subcommand token (bqtools encode IN OUT)

    # Bracket groups may contain spaces ([ONT_IN ...], [--dir-working DIR]) so
    # per-token startswith('[')/endswith(']') checks DO NOT work. Regroup the
    # token stream into scalars and complete [...] groups first.
    grouped: list[tuple[str, bool]] = []  # (token, is_group)
    buf: list[str] = []
    depth = 0
    for tok in parts:
        if depth == 0 and not tok.startswith("["):
            grouped.append((tok, False))
            continue
        # inside a group (or opening one)
        buf.append(tok)
        depth += tok.count("[") - tok.count("]")
        if depth <= 0:
            grouped.append((" ".join(buf), True))
            buf, depth = [], 0
    if buf:  # unbalanced brackets: treat as scalars
        grouped.extend((t, False) for t in buf)

    pseudo = {"options", "option", "options:", "commands", "command",
              "args", "arg", "arguments", "argument", "subcommand",
              "subcommands", "usage:", "flags"}
    out: list[dict[str, str]] = []
    position = 0

    def _add(candidate: str, required: bool = True) -> None:
        nonlocal position
        name = _normalize_metavar(candidate)
        if not name or name in pseudo or name.startswith("-"):
            return
        desc = (desc_map.get(name) or desc_map.get(name.upper())
                or desc_map.get(_normalize_metavar(candidate))
                or f"Positional argument {candidate}")
        out.append({"name": name, "type": _infer_scalar_type(name, desc),
                    "description": desc, "required": required,
                    "positional": True, "position": position})
        position += 1

    i = 0
    while i < len(grouped):
        token, is_group = grouped[i]
        if is_group:
            # strip nargs ellipsis FIRST: `[INPUT]...` must become `[INPUT]`
            # before [1:-1] slicing, or the tail dots slice off wrongly
            # (`[INPUT]...` -> `INPUT]..` garbage -- run #37 bqtools).
            inner_tok = token[:-3].rstrip() if token.endswith("...") else token
            inner = inner_tok[1:-1].split() if inner_tok.startswith("[") else inner_tok.split()
            # [--flag VAL] -> optional flag + value: skip. [METAVAR ...] ->
            # OPTIONAL positional (nargs): add (required False -- the brackets
            # mean the CLI can run without it, and forced-required would make
            # the LLM schema demand an arg the usage marks optional).
            if inner and not inner[0].startswith("-"):
                cand = _normalize_metavar(inner[0])
                if cand and cand not in pseudo:
                    _add(inner[0], required=False)
            i += 1
            continue
        if token.startswith("-"):
            # option flag outside brackets: skip it, its inline value, AND any
            # immediately following repeat group of that value ([ONT_IN ...]
            # after `-i ONT_IN` is the flag's nargs, NOT a positional)
            i += 1
            if i < len(grouped):
                nxt, nxt_group = grouped[i]
                if not nxt_group and not nxt.startswith("-") and not nxt.startswith("{") \
                        and not nxt.startswith("["):
                    i += 1
                    # consume following [same-value ...] groups
                    while i < len(grouped) and grouped[i][1]:
                        i += 1
            continue
        if token.startswith("{"):
            i += 1
            continue
        # bare metavar (SEQUENCE, <INPUT>, output-dir, OUT...)
        _add(token)
        i += 1

    # source 3: fire SYNOPSIS positionals in order (bioemu.sample.py SEQUENCE
    # NUM_SAMPLES OUTPUT_DIR <flags>). Only fire uses SYNOPSIS, and the POSITIONAL
    # ARGUMENTS block above already names them -- this catches name/order even if
    # the block layout differs slightly.
    syn_pos: list[str] = []
    for sm in re.finditer(r"(?:^|\n)\s*SYNOPSIS\s*\n\s*[^\n]*\n?\s*([^\n<]*<[^\n]*)\s*\n?",
                          text, re.IGNORECASE):
        for tok in sm.group(1).replace("<flags>", "").split():
            cand = _normalize_metavar(tok)
            if cand and not cand.startswith("-") and cand not in pseudo:
                if cand not in syn_pos:
                    syn_pos.append(cand)
    for cand in syn_pos:
        if cand not in {p["name"] for p in out}:
            out.append({"name": cand, "type": _infer_scalar_type(cand),
                        "description": f"Positional argument {cand}",
                        "required": True,  # bare SYNOPSIS metavar = mandatory slot
                        "positional": True, "position": position})
            position += 1

    # dedupe by name, preserving positional order
    dedup: list[dict[str, str]] = []
    seen: set[str] = set()
    for p in out:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        dedup.append(p)
    return dedup[:10]


def _is_probable_subcommand(tok: str) -> bool:
    """A usage token is probably a subcommand if it's lowercase/mixed and not a
    metavar in <...> or ALL-CAPS. e.g. 'encode' in 'bqtools encode <INPUT>'."""
    t = tok.strip("<>[]")
    if not t:
        return False
    if t.isupper() or t in ("...", "."):
        return False  # metavar or ellipsis, not a subcommand
    if t.startswith("{") or t.startswith("-"):
        return False
    return True


def _is_flag_line(line: str) -> bool:
    """A new flag definition line starts with a short (-x) or long (--xxx)
    flag token. NOTE: -{1,2}\\w -- a bare `-\\w` misses `--db` because the
    second dash is not a word char (kaptain lost 5/7 flags to this)."""
    return bool(re.match(r"^\s*-{1,2}\w", line))


_SECTION_HEADER = re.compile(
    r"^\s*(?:optional arguments|positional arguments|arguments|options|"
    r"global options|commands|flags|usage)\s*:\s*$"
    r"|^\s*(?:FLAGS|OPTIONS|POSITIONAL ARGUMENTS|ARGUMENTS|COMMANDS|"
    r"GLOBAL OPTIONS)\s*:?\s*$",
    re.IGNORECASE)


def _is_section_header(line: str) -> bool:
    """Section titles that delimit flag-definition regions: argparse's
    `options:`/`positional arguments:`, clap's `Arguments:`/`Options:`,
    fire's `FLAGS`. Flag lines only count INSIDE a section -- description /
    usage-example blocks (RawDescriptionHelpFormatter pastes `usage: kaptain
    --output results/ --subsampling 500M --fdr 5` under the usage) must NOT
    be parsed as flag definitions (run #37 kaptain: --output aliases became
    ['--subsampling','--fdr'] from exactly such an example line)."""
    return bool(_SECTION_HEADER.match(line or ""))


def _join_wrapped_help(help_output: str) -> list[str]:
    """Merge wrapped description continuation lines into their parent line.

    Two wrapping styles:
      - argparse/click column layout (continuations indent ~20+ spaces)
      - scopt/scala (flag at indent 2, `Type:`/`Default:` lines at indent 4)
    A fixed `indent >= 8` threshold misses the scopt style, so the Default:
    metadata never reaches the flag line and optional flags get marked
    required. Rule: a line is a continuation when its indent is GREATER than
    the indent of the line that started the current item.
    """
    items: list[list] = []  # [indent_of_first_line_or_None, text]
    for raw in help_output.splitlines():
        stripped = raw.strip()
        if not stripped:
            items.append([None, ""])
            continue
        indent = len(raw) - len(raw.lstrip())
        is_new_item = _is_flag_line(stripped) or re.match(
            r"^[a-zA-Z][\w -]*:\s*$", stripped) or stripped.lower().startswith(
            ("usage:", "options:", "commands:", "positional"))
        if (items and items[-1][0] is not None and not is_new_item
                and indent > items[-1][0]):
            # continuation of the previous item's description. Join with 2+
            # spaces so the flags-part/description split below still finds
            # the column gap at the join point.
            items[-1][1] += "  " + stripped
        else:
            items.append([indent, stripped])
    return [t for _, t in items]


def _parse_help_params(help_output: str) -> list[dict[str, str]]:
    """Extract CLI parameters from a --help / usage string.

    Handles common formats (argparse, typer, click, optparse), including:
      --reference PATH   Reference genome
      -r, --reference PATH
      -s SECTION, --section {refseq,genbank}   short flag with its OWN metavar
      -v, --verbose      Verbose output (boolean, takes no value)
      descriptions wrapped onto continuation lines

    Returns a list of {name, type, description, required, takes_value, aliases}.
    `required` inference at the source:
      - explicit [default: ...]  -> required False
      - boolean flag (no metavar) -> required False
      - everything else           -> required True (the command template
        renders every flag, so the agent must supply a value)
    """
    if not help_output:
        return []
    help_output = re.sub(r"\x1b\[[0-9;]*m", "", help_output)  # strip ANSI
    required_flags = _required_flags_from_usage(help_output)
    joined = _join_wrapped_help(help_output)
    # Only flag lines INSIDE a section header delimit actual flag definitions
    # (see _is_section_header). Before the first section the text is usage +
    # description/usage-examples -- parsing it turns example invocations into
    # fake flags (run #37 kaptain --output aliases=['--subsampling','--fdr']).
    section_idx = [i for i, ln in enumerate(joined) if _is_section_header(ln)]
    start = section_idx[0] if section_idx else 0
    params: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in joined[start:]:
        if not _is_flag_line(line):
            continue
        # split flags-part from description at the first 2+ space column gap
        gap = re.search(r"\s{2,}", line)
        if gap:
            flags_part, desc = line[:gap.start()].strip(), line[gap.end():].strip()
        else:
            flags_part, desc = line.strip(), ""
        flags = re.findall(r"(?<![\w-])(-{1,2}[\w][\w-]*)", flags_part)
        if not flags:
            continue
        # metavar pairing: the value token belongs to the flag it FOLLOWS.
        # For `-i ONT_IN [ONT_IN ...], --ont-in ONT_IN [ONT_IN ...]` the
        # canonical flag is --ont-in but its metavar is the FIRST non-flag
        # token (right after -i), NOT the last token on the line.
        tokens = [t for t in re.split(r"\s+", flags_part) if t]
        metavar = ""
        for t in tokens:
            if not t.startswith("-") and t not in (",",):
                mt = t
                # strip nargs decorations and short/long separator commas:
                # [ONT_IN ...] -> ONT_IN, 'N,' -> N
                mt = mt.strip("[]").rstrip(",").strip("[]")
                if mt and mt not in ("...", "."):
                    metavar = mt
                    break
            elif "=" in t and not t.startswith("="):
                # scopt/scala `--base_seed=BASE_SEED` syntax: the flag and its
                # metavar are ONE token -- a value-taking option, NOT boolean
                mt = t.split("=", 1)[1].strip("[]")
                if mt and mt not in ("...", "."):
                    metavar = mt
                    break
        # prefer the long flag as the canonical name
        long_flags = [f for f in flags if f.startswith("--")]
        name = long_flags[0] if long_flags else flags[0]
        aliases = [f for f in flags if f != name]
        if name in ("--help", "-h"):
            continue
        if name in seen:
            continue
        seen.add(name)
        # required: the usage line's brackets are authoritative (argparse
        # prints required flags bare, optional ones in [...]; ArgumentDefaults-
        # HelpFormatter appends `(default: None)` to required nargs+ flags too,
        # so `(default: ...)` alone misclassifies kaptain's -i/--db/--db-lookup/
        # -o as optional). Match the canonical flag OR any alias: usage shows
        # `-i` while the schema names it `--ont-in`.
        takes_value = bool(metavar)
        ptype = "string"
        mv = metavar.strip("<>").lower()
        # fire/scopt describe the flag's type in the desc (`Type: bool`) even
        # when a metavar is shown (`--filter_samples=FILTER_SAMPLES`); a bool
        # type overrides the metavar -> store-flag (no value slot).
        fire_bool = bool(re.search(r"\bType:\s*(?:bool|boolean)\b", desc,
                                   re.IGNORECASE))
        if not metavar or fire_bool:
            ptype = "boolean"
            takes_value = False
        elif mv in ("path", "file", "dir", "directory", "infile", "outfile"):
            ptype = "path"
        elif mv in ("int", "integer", "n", "count", "number"):
            ptype = "integer"
        elif mv in ("float", "double"):
            ptype = "float"
        else:
            # metavar is an unknown token (BATCH_SIZE, BASE_SEED): infer from
            # the NAME (num/count/seed -> int, rate -> float, path -> path)
            # instead of silently calling it a string -- bioemu's
            # `--num_samples` should reach the tool as an int, not "10".
            ptype = _infer_scalar_type(mv, desc)
        in_usage = name in required_flags or any(a in required_flags for a in aliases)
        has_default = bool(re.search(
            r"\[default[:=][^\]]*\]|\(default[:=][^)]*\)|\bdefault:\s*\S+",
            desc, re.IGNORECASE))
        # explicit `[required]`/`(required)` marker in the help description is
        # REAL evidence the flag is mandatory (argparse/clap help often writes
        # "Manifest file [required]"). A bare metavar is NOT: `--threads
        # THREADS` proves the flag takes a value, not that the CLI demands it.
        explicit_required = bool(re.search(r"\[required\]|\(required\)",
                                           desc, re.IGNORECASE))
        if in_usage:
            is_required = True
        elif required_flags:
            # usage had flags but NOT this one -> explicitly optional
            is_required = False
        else:
            # no usable usage line (fire SYNOPSIS, scopt, raw) -> fall back to
            # EXPLICIT evidence only. A metavar alone never implies required
            # (the old `not metavar` heuristic marked every bqtools flag
            # required and polluted the LLM schema into filling garbage for
            # --threads/--policy/--bitsize). Positionals are made required
            # separately by the positional merger (an argv slot is mandatory).
            is_required = explicit_required
        desc = re.sub(r"\s*\[(required|default|choices|count|append|nargs)[^\]]*\]\s*$",
                      "", desc).strip()
        # fire/scopt metadata lines leaked into the description
        # (`Type: int  Default: 10  Batch size...`) -- keep only the prose.
        desc = re.sub(r"^\s*(?:Type:.*?(?:\s+Default:.*?)?)\s{2,}",
                      "", desc, flags=re.IGNORECASE).strip()
        entry: dict[str, str] = {
            "name": name,
            "type": ptype,
            "description": desc or f"CLI flag {name}",
            "required": is_required,
            "takes_value": takes_value,
        }
        if aliases:
            entry["aliases"] = aliases
        params.append(entry)
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
        "subcommand_discovery_complete": False,
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

        # PATH with the venv's bin prepended -- shared by ALL install paths.
        # Previously `env` was defined only inside the pip smoke section, so
        # the npm/cargo/go branches' `_probe_cli(..., env=env)` crashed with
        # UnboundLocalError the first time a cargo tool's build actually
        # succeeded (bqtools, CI run #34).
        env = dict(os.environ)
        env["PATH"] = _venv_bin(venv_dir) + os.pathsep + env.get("PATH", "")

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
                for conda_env_name in env_names:  # NB: don't shadow the PATH `env` dict
                    if not conda_env_name or conda_env_name in seen_env:
                        continue
                    seen_env.add(conda_env_name)
                    rc_r, out_r, _ = _run(
                        ["conda", "run", "-n", conda_env_name, "which", "R"], 60)
                    if rc_r == 0 and out_r.strip():
                        conda_r = ["conda", "run", "-n", conda_env_name, "R"]
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
                    report["executable"] = cand
                    if _probe_cli(report, exe, env=env):
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
            # smoke via the built binary in target/release (with subcommand
            # discovery, like every other install path)
            built = os.path.join(repo_dir, "target", "release", name)
            if os.path.exists(built):
                report["executable"] = os.path.basename(built)
                if _probe_cli(report, built, env=env):
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
                report["executable"] = os.path.basename(built)
                if _probe_cli(report, built, env=env):
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
        bin_dir = _venv_bin(venv_dir)  # env already built above (venv bin on PATH)
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
                report["params_schema"] = _merge_positionals(
                    report["params_schema"], report["positional_args"])
                # for subcommand CLIs, probe each subcommand's --help to capture
                # its own parameters (so the schema tells agents how to call it)
                if report["arg_style"] == "subcommand" and report["subcommands"]:
                    subs = {}
                    exe_base = args[0]  # the executable
                    probed = 0
                    ok = 0
                    for sub in report["subcommands"]:
                        probed += 1
                        rc_s, out_s, err_s = _run(
                            [exe_base, sub, "--help"], RUN_TIMEOUT, env=env)
                        if rc_s == 0 and (out_s.strip() or err_s.strip()):
                            ok += 1
                            # params = flags (--input) + positional (<INPUT>) --
                            # positional skips the subcommand token itself
                            flags = _parse_help_params(out_s or err_s)
                            pos = _parse_positional_args(out_s or err_s, skip_first=True)
                            merged = _merge_positionals(flags, pos)
                            subs[sub] = {
                                "params": merged,
                                "usage": _extract_usage(out_s or err_s),
                            }
                    # complete only if every subcommand was probed successfully
                    report["subcommand_discovery_complete"] = (probed > 0 and ok == probed)
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
            # --help FIRST: on success the schema fields (params/arg_style/
            # subcommands) are parsed from real help output. If the sample
            # run went first and happened to exit 0 (many CLIs only warn on
            # unknown args), those fields would be parsed from arbitrary
            # run output -- garbage schema with status=passed.
            if _try([exe_path, "--help"]):
                return report
            if _try([exe_path, sample]):
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
                            # REAL parameters come from the tool's --help
                            # (authoritative + current), not the README text.
                            # README is only a fallback if --help yields nothing.
                            help_params = _parse_help_params(out_m or err_m)
                            # Positional arguments (scopt/scala style:
                            # `sample.py SEQUENCE NUM_SAMPLES OUTPUT_DIR`) are
                            # REQUIRED inputs -- dropping them (run #36 bioemu)
                            # handed the LLM a schema that could never invoke
                            # the tool.
                            help_positionals = _parse_positional_args(out_m or err_m)
                            if help_params or help_positionals:
                                # canonical schema = flags + positionals, one list
                                report["params_schema"] = _merge_positionals(
                                    help_params, help_positionals)
                                if help_positionals:
                                    report["positional_args"] = help_positionals
                            else:
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
    except (ImportError, ValueError, AttributeError):
        pass  # no SIGALRM on Windows / non-main thread -> watchdog disabled

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
                # LEVEL 1 vs LEVEL 2 isolation: a single tool's install/smoke
                # failure (or even an unexpected bug in execute_test for THAT
                # tool) is DATA -- record status=failed and keep scanning.
                # Only the global watchdog (TimeoutError) aborts the loop, and
                # genuinely systemic crashes propagate AFTER partial results
                # are persisted by the finally block below.
                try:
                    res = execute_test(
                        v.get("repo_url", ""),
                        install_method=v.get("install_method", ""),
                        install_cmd=v.get("install_cmd", ""),
                        entry_scripts=v.get("entry_scripts", []),
                        repo_name=tool,
                        external_commands=v.get("external_commands", []),
                    )
                except TimeoutError:
                    raise  # global watchdog hit: stop scanning, keep partials
                except Exception as exc:
                    res = {
                        "tool": tool,
                        "repo_url": v.get("repo_url", ""),
                        "status": "failed",
                        "reason": (f"uncaught exception in execute_test: "
                                   f"{type(exc).__name__}: {exc}"),
                    }
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
        # ALWAYS persist what we have -- even a systemic crash must not lose
        # the evidence already collected (step4 joins on this file).
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
