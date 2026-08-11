import json
import os
import re
import yaml

def load_tool_library(filename="tool_library_clean.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_verification(filename="tool_verification.json"):
    """Return {github_url -> verify result dict}. Absent file -> {}."""
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {r.get("repo_url", ""): r for r in results if r.get("repo_url")}


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

    v = (verification or {}).get(github_url) or {}
    e = load_execution().get(github_url) or {}

    install_method, install_url = guess_install_method(tool)
    command_template = guess_command(tool)

    # Override the *guesses* with verified evidence when available.
    if v.get("install_method") and v.get("install_cmd"):
        install_method = v["install_method"]
        install_url = v["install_cmd"] if not v["install_cmd"].startswith("http") else github_url
    # Command: prefer the REAL executable validated by execute_test (step 3.6),
    # then verify's probed command, then the guessed repo name. Never assume
    # tool name == CLI command (pyGenomeViz -> pgv-blast).
    exec_exe = e.get("executable") or ""
    verified_cmd = v.get("command") or ""
    if exec_exe:
        command_template = f"{exec_exe} {{{{input_file}}}}"
    elif verified_cmd:
        command_template = f"{verified_cmd} {{{{input_file}}}}"

    # inputs schema: prefer params parsed from the tool's real --help output
    # (execute_test.py step 3.6). Fall back to a placeholder, tagged with source
    # so reviewers can tell real evidence from guesses.
    parsed = e.get("params_schema") or []
    if parsed:
        inputs = {
            p["name"].lstrip("-").replace("-", "_"): {
                "type": p.get("type", "string"),
                "description": p.get("description", ""),
                "source": "help_parsed",
            }
            for p in parsed
        }
        inputs_src = "help_parsed"
    else:
        inputs = {
            "input_file": {
                "type": "string",
                "description": "Input file path inside /data.",
                "source": "placeholder",
            }
        }
        inputs_src = "placeholder"

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
        "type": "cli",
        "command": command_template,
        "description": f"[Auto-discovered] {description} (⭐{stars}, {language}){license_note}",
        "output_control": {
            "intercept_large_output": True,
            "max_preview_lines": 50,
        },
        "inputs": inputs,
        # --- environment / install contract (surfaces to downstream agents) ---
        # Tells the caller what to install before invoking, and which system
        # commands the tool expects on PATH (environment grounding).
        "install": {
            "method": install_method,
            "command": install_url or install_cmd,
            "system_commands": v.get("external_commands", []),
            "python_packages": e.get("installed_versions", [])[:20],
            "declared_packages": v.get("declared_packages", []),
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
            "exec_installed_versions": e.get("installed_versions", []),
            "exec_executable": e.get("executable", ""),
            "exec_retries": e.get("exec_retries", 0),
            "exec_heal_evidence": e.get("heal_evidence", ""),
            "inputs_source": inputs_src,
        }
    }

    return entry


def load_execution(filename="tool_execution.json"):
    """Return {github_url -> execution result dict}. Absent file -> {}."""
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {r.get("repo_url", ""): r for r in results if r.get("repo_url")}


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

    for tool in tools:
        github_url = tool.get("source", {}).get("github", "")
        v = verification.get(github_url) or {}
        e = execution.get(github_url) or {}
        status = v.get("status", "unverified")

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
        registry["tools"].append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(registry, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    with open(excluded_file, "w", encoding="utf-8") as f:
        json.dump(excluded, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(registry['tools'])} tools to {output_file}")
    print(f"Excluded {len(excluded)} unverified tools -> {excluded_file}")
    for e in excluded[:10]:
        print(f"  - {e['name']}: {e['reason'][:70]}")

    high_quality = [t for t in registry["tools"] if
                    t.get("_discovery_metadata", {}).get("quality_score", 0) >= 40]
    print(f"High quality tools (score>=40): {len(high_quality)}")
    for t in high_quality[:5]:
        print(f"  - {t['name']}: {t.get('description', '')[:60]}...")


if __name__ == "__main__":
    tools = load_tool_library()
    print(f"Loaded {len(tools)} tools from tool_library_clean.json")
    convert_to_registry(tools)
