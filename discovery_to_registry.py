import json
import os
import re
import yaml


def _canonical_key(name) -> str:
    """Registry input key for a param: --ont-in / <ONT_IN> / python arg all
    map to `ont_in`. Mirrors execute_test._canonical_param_name so the
    command template placeholder, the `inputs` dict key and the function
    schema parameter ALWAYS agree."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"^[-<>\{\}]+", "", name).lower().replace("-", "_")

def _infer_python_entry(readme_examples: list, pkg: str) -> str:
    """From readme import examples, infer a module:Class/function entry point.

    e.g. 'from pygenomeviz import GenomeViz' -> 'pygenomeviz:GenomeViz'
         'python -m bioemu.sample ...'       -> '' (use command template)
    Returns "" if nothing usable.
    """
    for ex in readme_examples:
        m = re.search(r"from\s+([\w.]+)\s+import\s+([\w]+)", ex)
        if m:
            mod, name = m.group(1), m.group(2)
            # NOTE: repo name != python module name in general; this is a
            # best-effort heuristic, execution evidence (callable_via) wins
            # over this guess at the call site.
            if mod.split(".")[0] == pkg.replace("-", "_"):
                return f"{mod}:{name}"
    return ""


def _missing_system_commands(external: list) -> list:
    """Normalize scanned external commands (dicts with kind) to a stable form
    that downstream agents can read. Strings are kept as-is; dicts keep
    {command, kind}."""
    out = []
    for c in external:
        if isinstance(c, dict):
            if c.get("command"):
                entry = {"command": c.get("command"),
                         "kind": c.get("kind", "system_missing")}
                if c.get("install_hint"):
                    entry["install_hint"] = c["install_hint"]
                out.append(entry)
        elif isinstance(c, str):
            out.append({"command": c, "kind": "system_missing"})
    return out


def normalize_repo_url(url) -> str:
    """Canonical join key for tool_library.source.github ==
    tool_verification.repo_url == tool_execution.repo_url.

    Strips whitespace/quotes, trailing slashes and .git so all of
      https://github.com/foo/bar
      https://github.com/foo/bar/
      https://github.com/foo/bar.git
    join to the same record.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip().strip("\"'")
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


def load_tool_library(filename="tool_library_clean.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_verification(filename="tool_verification.json"):
    """Return {normalized github_url -> verify result dict}. Absent file -> {}."""
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {normalize_repo_url(r.get("repo_url", "")): r
            for r in results if r.get("repo_url")}


def _param_flag(p: dict) -> str:
    """Return the CLI flag string for a parsed --help param: the explicit
    `flag` field when present, else the param name itself when it looks like
    a flag (-x / --xxx). '' for non-flag (positional) params.

    The execute_test help parser emits params with only `name` (e.g. '-s',
    '--genus') and no `flag` field -- treating those as non-flags caused the
    command builder to fabricate {{input_file}} while the inputs builder used
    the parsed params (contract mismatch -> every tool rejected).
    """
    if p.get("flag"):
        return p["flag"]
    name = str(p.get("name", ""))
    return name if name.startswith("-") else ""


def guess_install_method(tool):
    lang = tool.get("github_metadata", {}).get("language", "").lower()
    github_url = tool.get("source", {}).get("github", "")
    name = tool.get("name", "").lower()

    if lang == "python":
        return "pip_url", github_url
    elif lang in ("go", "rust", "c", "c++"):
        return "binary_url", github_url
    else:
        return "pip_url", github_url


def guess_command(tool):
    name = tool.get("name", "").replace(" ", "_").replace("-", "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    return f"{name.lower()} {{{{input_file}}}}"


def _check_registry_contract(entry: dict) -> str:
    """Return '' if the entry's schema contract is self-consistent, else a reason.

    Rules:
      - `inputs_source: placeholder` -> the inputs were GUESSED (default
        `input_file`), never parsed from the tool's --help -> not a contract.
      - every `{{var}}` in `command` must exist in `inputs` -> a command that
        references an undeclared input would render garbage argv.
    """
    md = entry.get("_discovery_metadata") or {}
    if md.get("inputs_source") == "placeholder":
        return "placeholder inputs (never --help-parsed); schema is a guess, not a contract"
    # subcommand contract: must have complete details so to_function_schemas
    # can emit leaf functions (bqtools_encode/decode/info). A subcommand tool
    # without details would only yield a bare `{{subcommand}}` call.
    if entry.get("arg_style") == "subcommand":
        if not entry.get("subcommand_discovery_complete"):
            return "subcommand discovery incomplete (subcommand_discovery_complete=false)"
        if not entry.get("subcommand_details"):
            return "subcommand_details empty; cannot emit leaf functions"
        # validate each subcommand's params: a leaf function with unnamed or
        # unrenderable params would emit a broken schema downstream.
        # NOTE: execute_test/_probe_cli writes the key `params` (matching
        # tool_runner._render_subcommand and generator leaf expansion).
        for sub, details in entry["subcommand_details"].items():
            details = details or {}
            params = details.get("params") or details.get("params_schema") or []
            for p in params:
                if not p.get("name"):
                    return f"subcommand '{sub}' has a parameter without a name"
            cmd = details.get("command") or ""
            used_sub = set(re.findall(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}", cmd))
            declared_sub = {_canonical_key(p["name"])
                            for p in params if p.get("name")}
            declared_sub |= set((details.get("inputs") or {}).keys())
            missing_sub = sorted(used_sub - declared_sub)
            if missing_sub:
                return (f"subcommand '{sub}' command references undeclared "
                        f"inputs: {missing_sub}")
    cmd = entry.get("command") or ""
    inputs = entry.get("inputs") or {}
    used = re.findall(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}", cmd)
    if entry.get("arg_style") == "subcommand":
        # `{{subcommand}}` is injected by the dispatcher (fnmap), not an input
        used = [v for v in used if v != "subcommand"]
    missing = sorted({v for v in used if v not in inputs})
    if missing:
        return f"command references undeclared inputs: {missing} (command {cmd[:60]})"
    return ""


def _infer_outputs(parsed: list, positional: list, arg_style: str) -> dict:
    """Best-effort output contract from parsed params.

    Flags whose metavar/name looks like an output (--output, --out, -o,
    --output-html, --outdir) become declared outputs so the agent knows the
    tool writes a file there. Positional args with an OUTPUT-ish name (usage:
    ... OUTPUT_DIR) count too -- after the canonical merge they live in
    `parsed` with a dash-less name, so a separate dash-only match would drop
    them (bioemu output_dir -> wrongly stdout-only -> OUTPUT_INVALID).
    Empty dict when nothing looks like an output.
    """
    # dedupe: `parsed` already contains the merged flags+positionals, but the
    # caller also passes the raw positional_args list -- scan both without
    # double-reporting the same canonical key.
    seen_keys: set[str] = set()
    outputs: dict = {}

    def _maybe_add(p: dict, src: str) -> None:
        name = p.get("name", "")
        key = _canonical_key(name)
        plain = name.lower().strip()
        if not key or key in seen_keys:
            return
        is_flag = plain.startswith("-")
        # match: -o / -O, --out*, --output*, or dash-less positional
        # containing "out" (output_dir, OUTPUT, outdir, output-html).
        if is_flag:
            matched = plain == "-o" or plain.startswith(("--output", "--out"))
        else:
            matched = "out" in plain
        if not matched:
            return
        seen_keys.add(key)
        is_dir = ("dir" in plain) or "directory" in plain
        outputs[key] = {
            "type": "directory" if is_dir else "file",
            "description": (p.get("description") or f"Output written by {name}"),
            "source": "help_parsed",
        }

    for p in parsed:
        _maybe_add(p, "parsed")
    for pa in positional:
        _maybe_add(pa, "positional")
    if outputs:
        return outputs
    # no explicit output flag found: a CLI that just prints to stdout.
    # Mark it as console output (not a file), so the agent won't hunt for a
    # file that can never exist.
    return {"stdout": {"type": "text",
                       "description": "Tool result printed to stdout.",
                       "source": "inferred"}}


def _subcommand_outputs(subs: dict, tool_name: str) -> dict:
    """Attach each subcommand's OWN output contract to its details.

    subcommand_details[sub].params are already the merged flags+positionals,
    so _infer_outputs can derive the output kind per sub (encode -> file,
    info -> stdout). This is what makes leaf-task validation (P1-5) work:
    `_task_output_kind(bqtools, 'encode')` reads THIS instead of stdout.
    """
    out = {}
    for sub, detail in (subs or {}).items():
        entry = dict(detail)
        params = entry.get("params") or []
        entry["outputs"] = _infer_outputs(params, [], "named")
        out[sub] = entry
    return out


def tool_to_registry_entry(tool, verification=None):
    name = tool.get("name", "unknown")
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name).strip('_').lower()
    description = tool.get("description", "No description available.")
    github_url = tool.get("source", {}).get("github", "")
    stars = tool.get("github_metadata", {}).get("stars", 0)
    language = tool.get("github_metadata", {}).get("language", "")
    tags = tool.get("tags", [])
    quality_score = tool.get("quality_score", 0)
    paper_doi = tool.get("source", {}).get("paper_doi", "")
    paper_title = tool.get("source", {}).get("paper_title", "")

    join_key = normalize_repo_url(github_url)
    v = (verification or {}).get(join_key) or {}
    e = load_execution().get(join_key) or {}

    install_method, install_url = guess_install_method(tool)
    command_template = guess_command(tool)

    # Override the *guesses* with verified evidence when available.
    if v.get("install_method") and v.get("install_cmd"):
        install_method = v["install_method"]
        install_url = v["install_cmd"] if not v["install_cmd"].startswith("http") else github_url

    # Command: prefer the REAL executable validated by execute_test (step 3.6),
    # then verify's probed command, then the guessed repo name.
    exec_exe = e.get("executable") or ""
    verified_cmd = v.get("command") or ""
    positional = e.get("positional_args") or []
    arg_style = e.get("arg_style") or ""
    callable_via = e.get("callable_via") or ""
    # SINGLE SOURCE OF TRUTH: command template AND inputs below must both be
    # derived from this same parsed schema, never inferred independently.
    parsed = e.get("params_schema") or []
    schema_pending_reason = ""
    base_cmd = exec_exe or verified_cmd or ""
    # python tools with a `python -m <module>` entry point -> use that
    if arg_style == "python" and callable_via.startswith("python -m "):
        mod = callable_via.replace("python -m ", "").split()[0]
        base_cmd = f"python -m {mod}"
    if base_cmd:
        if arg_style == "python" and callable_via.startswith("python -m "):
            # python -m tools: canonical template = ordered positionals first
            # (SEQUENCE NUM_SAMPLES OUTPUT_DIR), then flags (--filter_samples).
            # Both come from the SAME parsed schema that builds `inputs`, so
            # the contract (every {{var}} declared) holds by construction.
            flag_params = [p for p in parsed if _param_flag(p)]
            pos_params = sorted(
                [p for p in parsed if p.get("positional")],
                key=lambda p: p.get("position", 0))
            if pos_params or flag_params:
                parts = [f"{{{{{p['name']}}}}}" for p in pos_params]
                for p in flag_params:
                    flag = _param_flag(p)
                    key = _canonical_key(p["name"])
                    if str(p.get("type", "")).lower() in ("bool", "boolean") \
                            or p.get("takes_value") is False:
                        # store-flag: bare flag, no value slot
                        parts.append(flag)
                    else:
                        parts.append(f"{flag} {{{{{key}}}}}")
                command_template = f"{base_cmd} {' '.join(parts)}"
            else:
                # no grounded params at all -> pending, not a fake command
                schema_pending_reason = ("python -m entry but no --help params "
                                         "parsed; command would be a guess")
                command_template = ""
        elif positional and arg_style == "positional":
            # positional CLI with optional flags: pgv-blast seq1 seq2 -o out.
            # Canonical template = ordered positionals FIRST, then flags --
            # both from the SAME parsed schema that builds `inputs`.
            pos_params = sorted(
                [p for p in parsed if p.get("positional")],
                key=lambda p: p.get("position", 0))
            flag_params = [p for p in parsed if _param_flag(p)]
            parts = [f"{{{{{p['name']}}}}}" for p in pos_params]
            for p in flag_params:
                flag = _param_flag(p)
                key = _canonical_key(p["name"])
                if str(p.get("type", "")).lower() in ("bool", "boolean") \
                        or p.get("takes_value") is False:
                    parts.append(flag)
                else:
                    parts.append(f"{flag} {{{{{key}}}}}")
            if parts:
                command_template = f"{base_cmd} {' '.join(parts)}"
            else:
                # no positional args parsed either -> nothing grounded to
                # render; do NOT fabricate {{input_file}}
                schema_pending_reason = ("positional CLI but no positional args "
                                         "parsed from --help")
                command_template = ""
        elif arg_style == "subcommand":
            # subcommand CLI: bqtools <subcommand> [args...]. The command is just
            # `<cmd> {{subcommand}}`; each subcommand's OWN params live in
            # `subcommand_details` and are expanded by to_function_schemas into
            # leaf functions (bqtools_encode). Do NOT hoist the first
            # subcommand's params into the base command -- encode/decode/info
            # take different args and a merged template is a fake contract.
            command_template = f"{base_cmd} {{{{subcommand}}}}"
        else:
            # named CLI: build the template from the SAME parsed params that
            # build `inputs` below (e.g. `macrel --output {{output}}`).
            # NEVER fabricate a {{input_file}} the inputs don't declare --
            # that self-contradictory entry is exactly what the contract
            # check rejects. No usable params -> PENDING, not a fake command.
            flag_params = [p for p in parsed if _param_flag(p)]
            if flag_params:
                parts = []
                for p in flag_params:
                    flag = _param_flag(p)
                    # placeholder must match the INPUT key (flag stripped), not
                    # the raw flag name -- `--output` -> `{{output}}`
                    key = _canonical_key(p["name"])
                    parts.append(f"{flag} {{{{ {key} }}}}")
                tmpl = " ".join(parts).replace("{{ ", "{{").replace(" }}", "}}")
                command_template = f"{base_cmd} {tmpl}"
            elif parsed or positional:
                # --help gave params but none renderable as flags/positionals:
                # schema exists but is unusable for a command template.
                schema_pending_reason = ("help output parsed but no usable "
                                         "flags/positionals to build a command "
                                         "template")
                command_template = ""
            else:
                # no --help evidence at all: any command would be a guess
                schema_pending_reason = ("no --help params parsed; command/"
                                         "inputs would be a guess")
                command_template = ""

    if not base_cmd and not schema_pending_reason:
        # no verified executable AND no probed command: the guessed
        # `name {{input_file}}` template from guess_command() is pure
        # fabrication -> pending, not active
        schema_pending_reason = "no verified executable or command to invoke"

    # inputs schema: prefer params parsed from the tool's real --help output
    # (execute_test.py step 3.6). Fall back to a placeholder, tagged with source
    # so reviewers can tell real evidence from guesses.
    parsed = e.get("params_schema") or []
    positional = e.get("positional_args") or []
    arg_style = e.get("arg_style") or ""
    if arg_style == "subcommand":
        # subcommand tools have NO top-level inputs: each subcommand's params
        # live in subcommand_details and become leaf-function parameters via
        # to_function_schemas. A fake top-level `input_file` would break the
        # contract check (command `{{subcommand}}` references no inputs, and
        # validate_arguments would demand an arg the leaf call never sets).
        inputs = {}
        inputs_src = "subcommand"
    elif parsed:
        inputs = {}
        for p in parsed:
            if not p.get("name"):
                continue
            key = _canonical_key(p["name"])
            spec = {
                "type": p.get("type", "string"),
                "description": p.get("description", ""),
                "required": True if p.get("required") is True else False,
                "source": "help_parsed",
            }
            if p.get("positional"):
                spec["positional"] = True
                if p.get("position") is not None:
                    spec["position"] = p["position"]
                spec["required"] = True  # a positional argv slot is mandatory
            inputs[key] = spec
        inputs_src = "help_parsed"
    else:
        # NO --help evidence: do NOT fabricate `input_file`. The entry is
        # routed to pending_tools.json by the pending gate in
        # convert_to_registry (third state: not active, not excluded).
        inputs = {}
        inputs_src = "placeholder"
        if not schema_pending_reason:
            schema_pending_reason = "no --help schema parsed (inputs would be a guess)"
    # positional args (usage: cmd file1 file2 -o out) are ALREADY part of
    # params_schema: execute_test._merge_positionals folds flags + positionals
    # into one canonical list, so this stage reads a SINGLE inputs source.
    # No separate injection here -- it would only duplicate entries under a
    # different key normalization (lstrip("<>[]") vs lstrip("-")).
    # NOTE: subcommand CLIs do NOT get a `subcommand` input here. The registry
    # keeps subcommands/subcommand_details so to_function_schemas can expand
    # each subcommand into its own LEAF function (bqtools_encode), and the
    # executor dispatches via fnmap -> _active_subcommand. Exposing a required
    # `subcommand` parameter would force the agent to pass it AND make
    # validate_arguments demand it -- breaking every leaf call.

    # license status is surfaced to end users in the description (the MCP
    # registry / tool list drops _discovery_metadata, so this is the only
    # user-visible channel). No license -> explicit "research-use only" note.
    license_note = ""
    if v:
        if v.get("has_license"):
            lic = v.get("license_path", "").split("/")[-1] or "license"
            license_note = f" (license: {lic})"
        else:
            license_note = " (NO license - research use only)"

    entry = {
        "name": clean_name,
        "type": "python" if arg_style == "python" else "cli",
        "command": command_template,
        "arg_style": arg_style or "named",
        "callable_via": e.get("callable_via", "") or v.get("callable_hint", ""),
        "readme_examples": (e.get("readme_examples") or v.get("readme_examples") or []),
        "readme_usage": v.get("readme_usage", ""),
        "description": f"[Auto-discovered] {description} (⭐{stars}, {language}){license_note}",
        "output_control": {
            "intercept_large_output": True,
            "max_preview_lines": 50,
        },
        "inputs": inputs,
        # output contract: tells agents what the tool produces and where, so
        # they know what "success" looks like (output file exists). Auto-disco
        # can't always know the exact path, so this is best-effort + honest.
        "outputs": _infer_outputs(parsed, positional, arg_style),
        # python-API tools: expose an execution entry_point (module:Class) so
        # run_tool_spec's python runner can invoke it: `from m import C; C(**args)`
        "execution": (
            e.get("execution")
            or ({"type": "python", "entry_point": _infer_python_entry(e.get("readme_examples", []), clean_name)}
                if arg_style == "python" and _infer_python_entry(e.get("readme_examples", []), clean_name)
                else None)
        ),
        # subcommand CLIs: subcommand names + per-subcommand param details so
        # agents know how to invoke (e.g. bqtools encode <in> <out>). Each sub
        # also carries its OWN output contract (inferred from that sub's params)
        # so leaf-task validation checks a FILE/DIR, not stdout.
        "subcommands": e.get("subcommands", []),
        "subcommand_details": _subcommand_outputs(e.get("subcommand_details", {}), clean_name),
        "subcommand_discovery_complete": e.get("subcommand_discovery_complete", False),
        # --- environment / install contract (surfaces to downstream agents) ---
        # Tells the caller what to install before invoking, and which system
        # commands the tool expects on PATH (environment grounding).
        "install": {
            "method": install_method,
            "command": (v.get("install_cmd")
                        or e.get("install_cmd")
                        or install_url
                        or ""),
            "system_commands": _missing_system_commands(v.get("external_commands", [])),
            "python_packages": e.get("installed_versions", [])[:20],
            "declared_packages": v.get("declared_packages", []),
            "missing_deps": v.get("missing_deps", []),
            "venv_path": e.get("venv_path", ""),
        },
        "_discovery_metadata": {
            "github": github_url,
            "stars": stars,
            "language": language,
            "tags": tags,
            "quality_score": quality_score,
            "paper_doi": paper_doi,
            "paper_title": paper_title,
            "install_method": install_method,
            "install_url": install_url,
            # --- verification evidence (from verify_repo.py) ---
            "verified_status": v.get("status", "unverified"),
            "verified_reason": v.get("reason", "not verified"),
            "verified_license": v.get("has_license", False),
            "verified_license_path": v.get("license_path", ""),
            "verified_entry_scripts": v.get("entry_scripts", []),
            "verified_checked_at": v.get("checked_at", ""),
            # --- environment grounding (system deps the venv can't provide) ---
            "dependencies": {
                "system_commands": v.get("external_commands", []),
                "readme_hint": v.get("readme_hint", ""),
                "container_files": v.get("container_files", []),
                "install_method": install_method,
            },
            # --- execution evidence (from execute_test.py, step 3.6) ---
            "exec_status": e.get("status", ""),
            "exec_reason": e.get("reason", ""),
            "exec_install_evidence": e.get("install_evidence", ""),
            "exec_run_evidence": e.get("run_evidence", ""),
            "exec_params_schema": e.get("params_schema", []),
            "exec_positional_args": e.get("positional_args", []),
            "exec_installed_versions": e.get("installed_versions", []),
            "exec_executable": e.get("executable", ""),
            "exec_retries": e.get("exec_retries", 0),
            "exec_heal_evidence": e.get("heal_evidence", ""),
            "inputs_source": inputs_src,
            "pending_reason": schema_pending_reason,
        }
    }

    return entry


def load_execution(filename="tool_execution.json"):
    """Return {normalized github_url -> execution result dict}. Absent file -> {}."""
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {normalize_repo_url(r.get("repo_url", "")): r
            for r in results if r.get("repo_url")}


def convert_to_registry(tools, output_file="discovered_registry.yaml",
                        verification_file="tool_verification.json",
                        min_status=("verified", "repo_ok"),
                        excluded_file="excluded_tools.json",
                        require_passed=False):
    """Convert only tools whose repo passed verification.

    Tools whose repo could not be verified (clone failure / no entry point /
    no license marker) are written to `excluded_file` with the reason, instead
    of silently producing a placeholder entry that would fail at runtime.

    When `require_passed=True`, only tools that ALSO survived the step 3.6
    execution smoke test (installed + ran on a sample input) enter the
    registry; everything else goes to `excluded_file`.
    """
    verification = load_verification(verification_file)
    execution = load_execution()
    registry = {"tools": []}
    excluded = []
    pending = []

    for tool in tools:
        github_url = tool.get("source", {}).get("github", "")
        join_key = normalize_repo_url(github_url)
        v = verification.get(join_key) or {}
        e = execution.get(join_key) or {}
        status = v.get("status", "unverified")

        # JOIN DIAGNOSTIC: a verified tool with NO execution record is almost
        # always a tool_library github <-> tool_execution repo_url join miss
        # (not a tool defect). Without this print it surfaces later as a bare
        # "placeholder inputs" contract-reject and the real cause is invisible.
        if status in min_status and not e and execution:
            sample = list(execution.keys())[:3]
            print(f"  [execution-miss] tool={tool.get('name', 'unknown')} "
                  f"github={github_url!r} -> no tool_execution.json record "
                  f"(checked normalized key {join_key!r}; "
                  f"{len(execution)} execution records exist, e.g. {sample})")

        if status not in min_status:
            excluded.append({
                "name": tool.get("name", "unknown"),
                "github": github_url,
                "status": status,
                "reason": v.get("reason", "no verification record"),
                "install_cmd": v.get("install_cmd", ""),
                "has_license": v.get("has_license", False),
                "paper_title": tool.get("source", {}).get("paper_title", ""),
            })
            continue

        # step 3.6 execution gate: only tools that actually ran make it in
        if require_passed and e.get("status") != "passed":
            excluded.append({
                "name": tool.get("name", "unknown"),
                "github": github_url,
                "status": f"exec-{e.get('status', 'unknown')}",
                "reason": e.get("reason", "no execution record"),
                "install_cmd": v.get("install_cmd", ""),
                "has_license": v.get("has_license", False),
                "paper_title": tool.get("source", {}).get("paper_title", ""),
            })
            continue

        entry = tool_to_registry_entry(tool, verification)

        # ---- PENDING-SCHEMA GATE (third state) ----
        # Tool verified + executed, but its schema could not be grounded in
        # --help/README evidence. NOT active (would be a fake contract), NOT
        # excluded (repo is fine -- only the schema is unresolved). Preserved
        # in pending_tools.json for a later LLM/manual schema pass.
        pending_reason = (entry.get("_discovery_metadata") or {}).get("pending_reason", "")
        if pending_reason:
            pending.append({
                "name": tool.get("name", "unknown"),
                "github": github_url,
                "status": "pending_schema",
                "reason": pending_reason,
                "install_cmd": v.get("install_cmd", ""),
                "has_license": v.get("has_license", False),
                "paper_title": tool.get("source", {}).get("paper_title", ""),
            })
            print(f"  [pending-schema] {entry.get('name')}: {pending_reason[:70]}")
            continue

        # ---- REGISTRY CONTRACT CHECK (hard gate at generation time) ----
        # An entry must not enter the active registry if its contract is
        # self-contradictory: placeholder inputs (schema is a guess) or a
        # command template that references an input not declared in `inputs`.
        # Checking here (not in tool_agent_test) keeps bad entries OUT of the
        # active registry entirely, so preflight/agent never see them.
        contract_err = _check_registry_contract(entry)
        if contract_err:
            excluded.append({
                "name": tool.get("name", "unknown"),
                "github": github_url,
                "status": f"contract-{entry.get('arg_style', '?')}",
                "reason": contract_err,
                "install_cmd": v.get("install_cmd", ""),
                "has_license": v.get("has_license", False),
                "paper_title": tool.get("source", {}).get("paper_title", ""),
            })
            print(f"  [contract-reject] {entry.get('name')}: {contract_err}")
            continue

        registry["tools"].append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(registry, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    with open(excluded_file, "w", encoding="utf-8") as f:
        json.dump(excluded, f, ensure_ascii=False, indent=2)

    with open("pending_tools.json", "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(registry['tools'])} tools to {output_file}")
    print(f"Pending {len(pending)} tools (schema unresolved) -> pending_tools.json")
    print(f"Excluded {len(excluded)} unverified tools -> {excluded_file}")
    for e in excluded[:10]:
        print(f"  - {e['name']}: {e['reason'][:70]}")

    high_quality = [t for t in registry["tools"] if
                     t.get("_discovery_metadata", {}).get("quality_score", 0) >= 40]
    print(f"High quality tools (score>=40): {len(high_quality)}")
    for t in high_quality[:5]:
        print(f"  - {t['name']}: {t.get('description', '')[:60]}...")

    # real outcome counts (NOT len(tools) of the input library)
    return {"active": len(registry["tools"]),
            "pending": len(pending),
            "excluded": len(excluded)}


if __name__ == "__main__":
    tools = load_tool_library()
    print(f"Loaded {len(tools)} tools from tool_library_clean.json")
    convert_to_registry(tools)
