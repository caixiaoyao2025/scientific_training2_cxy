"""Pure argv rendering -- the ONLY place a ToolSpec + arguments becomes argv.

Both tool_spec.render_spec (schema/contract layer) and tool_runner (executor)
delegate HERE, so there is exactly one rendering path and neither layer holds a
second copy of the render logic. Extracted from tool_runner to break the
tool_spec <-> tool_runner module cycle: tool_spec imports the renderer directly
and no longer needs the runner.
"""

from __future__ import annotations

import re
import shlex
from typing import Any


def render_command(command_template: str, arguments: dict[str, Any]) -> list[str]:
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
                # boolean store-flag: the flag itself IS the value's
                # rendering (`--verbose True` would make argparse fail with
                # "unrecognized arguments"). True -> keep the bare flag
                # (already appended); False was filtered out above.
                if filtered.get(var) is True:
                    pass
                else:
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


def render_subcommand(spec: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Render a subcommand-CLI invocation from the CANONICAL leaf ToolSpec.

    The leaf (make_leaf_spec) already carries a concrete command
    (`bqtools encode`) and inputs scoped to that subcommand, so the argv is
    built straight from `spec.inputs` -- the renderer NEVER re-reads
    subcommand_details as a second schema source. Positionals go FIRST in
    argv position order, then flags (bare for store-flags), so
    `bqtools encode /tmp/in.fa --output out` renders exactly as the tool's
    usage describes.
    """
    # normalize EITHER brace convention ({subcommand} / {{subcommand}}) and
    # inline the active sub into its slot. make_leaf_spec already produces a
    # concrete `bqtools encode`; this is defense-in-depth so a raw base spec
    # can never leak a literal `{subcommand}` token into argv.
    sub = spec.get("_active_subcommand") or arguments.get("subcommand", "")
    cmd = (spec.get("command") or "").replace("{{subcommand}}", "{subcommand}")
    if "{subcommand}" in cmd:
        cmd = cmd.replace("{subcommand}", sub)
    argv = [t for t in cmd.split() if t]
    if len(argv) < 2:
        # defensive: not a concrete leaf; keep base exe + active sub
        exe = argv[0] if argv else (spec.get("name") or "").split("_")[0]
        argv = [exe] + ([sub] if sub else [])
    params: list[dict[str, Any]] = []
    for key, meta in (spec.get("inputs") or {}).items():
        if not isinstance(meta, dict):
            continue
        params.append({
            "key": key,
            "flag": meta.get("flag") or (f"--{key.replace('_', '-')}"
                                         if not meta.get("positional") else ""),
            "positional": bool(meta.get("positional")),
            "position": meta.get("position") if meta.get("position") is not None else 0,
            "type": meta.get("type", "string"),
            "takes_value": meta.get("takes_value"),
        })
    # positionals first (argv order), then flags in declared order
    positionals = sorted([p for p in params if p["positional"]],
                         key=lambda p: p["position"])
    flags = [p for p in params if not p["positional"]]
    for p in positionals:
        val = arguments.get(p["key"])
        if val in (None, "", False):
            continue
        # NOT shlex-quoted: this argv goes straight to subprocess (no shell),
        # so quoting would inject literal quotes into the path.
        argv.append(str(val))
    for p in flags:
        val = arguments.get(p["key"])
        if val in (None, "", False):
            continue
        store_flag = str(p["type"]).lower() in ("bool", "boolean") \
            or p.get("takes_value") is False
        argv.append(p["flag"])
        if not store_flag:
            argv.append(str(val))
    return argv
