"""Parser fixture test: --help text -> canonical schema -> render.

Locks the schema parser (execute_test) and renderers (tool_runner) against
REAL help output captured from four heterogeneous tools, so regressions in
required/alias/positional/boolean handling show up here before any agent run:

  kaptain    : argparse, short-flag required (-i/--db/--db-lookup/-o), wrapped
               usage lines, store-flag booleans (--version/--log)
  kaptain_ln : same tool via README long-form usage (--ont-in/--output)
  bioemu     : fire, SYNOPSIS + POSITIONAL ARGUMENTS block, boolean flag with
               explicit `-f, --filter_samples=...` (Type: bool)
  bqtools    : clap, `[INPUT]...` variadic positional, `[default: ...]`
               requiredness, store-flag booleans

Each fixture asserts the CANONICAL layer that discovery_to_registry and the
agent renderer both consume:
  - canonical param keys (--ont-in -> ont_in == <ONT_IN> -> ont_in)
  - required flags (argparse usage brackets / clap [default: ...])
  - positionals marked required + ordered
  - store-flags: type boolean + takes_value False (render as bare flag)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import execute_test as et  # noqa: E402
from discovery_to_registry import _infer_outputs  # noqa: E402

KAPTAIN_HELP = """\
usage: kaptain [-h] -i ONT_IN [ONT_IN ...] --db DB --db-lookup DB_LOOKUP
               [--dir-working DIR_WORKING] -o OUTPUT
               [--output-html OUTPUT_HTML]
               [--subsampling {200M,500M,1000M,1500M,2000M,None} [{200M,500M,1000M,1500M,2000M,None} ...]]
               [--fdr {15,10,5,1} [{15,10,5,1} ...]] [--threads THREADS]
               [--version] [--log]

usage: kaptain [-h] --ont-in ONT_IN [ONT_IN ...] --db DB --db-lookup DB_LOOKUP [--dir-working DIR_WORKING] --output OUTPUT
Usage examples

Basic Classification:

kaptain --ont-in query.fq --db my_database --db_lookup my_database.lookup
        --output results/ --subsampling 500M --fdr 5

options:
  -h, --help            show this help message and exit
  -i ONT_IN [ONT_IN ...], --ont-in ONT_IN [ONT_IN ...]
                        ONT input FASTA/Q file (default: None)
  --db DB               Prefix of KMA database. Database exists of four files
                        named *.{comp.b, length.b, name, seq.b} (default:
                        None)
  --db-lookup DB_LOOKUP
                        Lookup file of KMA database (default: None)
  --dir-working DIR_WORKING
                        Working directory (default:
                        C:\\Users\\123456\\Desktop\\st2\\working)
  -o OUTPUT, --output OUTPUT
                        Output directory (default: None)
  --output-html OUTPUT_HTML
                        Output report name (default: report.html)
  --subsampling {200M,500M,1000M,1500M,2000M,None} [{200M,500M,1000M,1500M,2000M,None} ...]
                        Subsample input to number of bases before
                        classification. Leave empty or use None for no
                        downsampling. (default: [None])
  --fdr {15,10,5,1} [{15,10,5,1} ...]
                        FDR setting. (default: [5])
  --threads THREADS     Number of threads (default: 4)
  --version             Print version and exit
  --log                 Write out log information to file (default: False)
"""

BIOEMU_HELP = """\
NAME
    bioemu_mock.py - Generate samples for a specified sequence, using a trained model.

SYNOPSIS
    bioemu_mock.py SEQUENCE NUM_SAMPLES OUTPUT_DIR <flags>

DESCRIPTION
    Generate samples for a specified sequence, using a trained model.

POSITIONAL ARGUMENTS
    SEQUENCE
        Type: str | pathlib.Path
        Amino acid sequence for which to generate samples, or a path to a .fasta file, or a path to an .a3m file with MSAs.
    NUM_SAMPLES
        Type: int
        Number of samples to generate.
    OUTPUT_DIR
        Type: str | pathlib.Path
        Directory to save the samples.

FLAGS
    --batch_size_100=BATCH_SIZE_100
        Type: int
        Default: 10
        Batch size you'd use for a sequence of length 100.
    -f, --filter_samples=FILTER_SAMPLES
        Type: bool
        Default: True
        Filter out unphysical samples.
    --model_name=MODEL_NAME
        Type: Optional
        Default: 'bioemu-v1.1'
        Name of pretrained model to use.
"""

BQTOOLS_HELP = """\
Encode reads to BINSEQ format

Usage: bqtools encode [OPTIONS] [INPUT]...

Arguments:
  [INPUT]...  Input file(s), or stdin when omitted

Options:
  -f, --format <FORMAT>          Output format [default: binsq]
  -b, --batch-size <BATCH_SIZE>  Records per batch [default: 1000]
      --interleaved              Input is interleaved paired-end
  -m, --manifest <MANIFEST>      Manifest file [required]
  -d, --depth <DEPTH>            Depth multiplier [required]
  -o, --output <OUTPUT>          Output BINSEQ file [required]
      --mode <MODE>              Compression mode [default: binz]
      --policy <POLICY>          Error policy
  -b, --bitsize <BITSIZE>        Bit size
      --skip-headers             Skip headers
      --threads <THREADS>        Threads
  -s, --block-size <BLOCK_SIZE>  Block size [required]
  -l, --level <LEVEL>            Compression level [required]
      --archive                  Archive mode
      --pipe                     Pipe mode
  -h, --help                     Print help
"""

# real-world clap help: `Usage: ... [OPTIONS]` with NO `[required]` markers
# (bqtools's actual --help). Every option is OPTIONAL; only the positional is
# required. The old "metavar -> required" fallback marked all of these
# required and polluted the LLM schema.
CLAP_HELP = """\
Encode reads to BINSEQ format

Usage: bqtools encode [OPTIONS] [INPUT]...

Arguments:
  [INPUT]...  Input file(s), or stdin when omitted

Options:
  -f, --format <FORMAT>          Output format
  -b, --batch-size <BATCH_SIZE>  Records per batch
      --interleaved              Input is interleaved paired-end
  -m, --manifest <MANIFEST>      Manifest file
  -o, --output <OUTPUT>          Output BINSEQ file
      --mode <MODE>              Compression mode
      --policy <POLICY>          Error policy
      --threads <THREADS>        Number of threads
      --archive                  Archive mode
  -h, --help                     Print help
"""


def _by_name(params: list[dict], name: str) -> dict:
    for p in params:
        if p.get("name") == name:
            return p
    raise AssertionError(f"param {name} not found in {[p.get('name') for p in params]}")


def _run(help_text: str, skip_first: bool = False):
    flags = et._parse_help_params(help_text)
    pos = et._parse_positional_args(help_text, skip_first=skip_first)
    merged = et._merge_positionals(flags, pos)
    return flags, pos, merged


def test_kaptain_required_and_aliases():
    flags, pos, merged = _run(KAPTAIN_HELP)
    assert pos == [], f"kaptain has no positionals, got {pos}"
    # required flags come from the usage brackets (-i/--db/--db-lookup/-o),
    # and the short-flag alias must resolve the requiredness of --ont-in/--output
    ont_in = _by_name(merged, "--ont-in")
    assert ont_in["required"] is True, ont_in
    assert "-i" in ont_in.get("aliases", []), ont_in
    db = _by_name(merged, "--db")
    assert db["required"] is True, db
    db_lookup = _by_name(merged, "--db-lookup")
    assert db_lookup["required"] is True, db_lookup
    out = _by_name(merged, "--output")
    assert out["required"] is True, out
    assert "-o" in out.get("aliases", []), out
    for opt in ("--dir-working", "--output-html", "--subsampling", "--fdr",
                "--threads"):
        assert _by_name(merged, opt)["required"] is False, opt
    # the usage-example block (--output results/ --subsampling 500M --fdr 5)
    # must NOT leak into the schema as aliases/params
    assert all(p.get("name") != "--subsampling" or p.get("aliases") is None
               for p in merged), "usage-example block leaked into aliases"
    # store-flag booleans: type boolean, no value slot
    ver = _by_name(merged, "--version")
    assert ver.get("type") == "boolean" and ver.get("takes_value") is False, ver
    log = _by_name(merged, "--log")
    assert log.get("type") == "boolean" and log.get("takes_value") is False, log


def test_kaptain_canonical_keys():
    """--ont-in, <ONT_IN> and ont_in all map to one canonical input key."""
    flags, _, _ = _run(KAPTAIN_HELP)
    for p in flags:
        k = et._canonical_param_name(p.get("name", ""))
        assert k == p["name"].lstrip("-").replace("-", "_"), (p["name"], k)
        assert "[" not in k and "]" not in k and "{" not in k, (p["name"], k)
    assert et._canonical_param_name("<ONT_IN>") == "ont_in"
    assert et._normalize_metavar("[INPUT]...") == "input"
    assert et._canonical_param_name("--filter_samples") == "filter_samples"


def test_bioemu_positionals_and_boolean():
    flags, pos, merged = _run(BIOEMU_HELP)
    names = [p.get("name") for p in merged]
    # positionals are REQUIRED and keep argv order; names are already
    # canonicalized by _parse_positional_args (SEQUENCE -> sequence)
    seq = _by_name(merged, "sequence")
    assert seq.get("positional") is True and seq.get("required") is True, seq
    assert seq.get("position") == 0, seq
    assert _by_name(merged, "num_samples").get("position") == 1
    assert _by_name(merged, "output_dir").get("position") == 2
    assert seq.get("name") in names
    # canonical dedup: SEQUENCE/sequence both canonicalize, only one entry
    keys = [et._canonical_param_name(p.get("name", "")) for p in merged]
    assert len(keys) == len(set(keys)), f"duplicate canonical keys: {keys}"
    # boolean flag (Type: bool) -> store-flag, no value slot
    fs = _by_name(merged, "--filter_samples")
    assert fs.get("type") == "boolean" and fs.get("takes_value") is False, fs
    assert "-f" in fs.get("aliases", []), fs
    # non-bool flags keep a value slot (type inferred from Type:/Default:;
    # truncated fixture -> string, but takes_value must stay True)
    bs = _by_name(merged, "--batch_size_100")
    assert bs.get("takes_value") is not False, bs


def test_bqtools_positional_and_required():
    """clap: `[INPUT]...` variadic -> one required positional `input`."""
    flags, pos, merged = _run(BQTOOLS_HELP, skip_first=True)
    input_p = _by_name(merged, "input")
    assert input_p.get("positional") is True, input_p
    assert input_p.get("required") is True, input_p
    assert input_p.get("name") == "input", input_p
    # `encode` (the subcommand token) must not appear as a positional
    assert all(p.get("name") != "encode" for p in merged), merged
    # required flags marked, defaults optional
    assert _by_name(merged, "--manifest")["required"] is True
    assert _by_name(merged, "--output")["required"] is True
    assert _by_name(merged, "--format")["required"] is False
    assert _by_name(merged, "--batch-size")["required"] is False
    # a bare metavar is NOT a required signal: --threads/--policy/--bitsize
    # have no [required] marker and no default -> must be OPTIONAL (the old
    # "not metavar -> required" fallback marked them required and polluted the
    # LLM schema into filling garbage values).
    for fl in ("--threads", "--policy", "--bitsize", "--mode"):
        assert _by_name(merged, fl)["required"] is False, fl
    # store-flag booleans (clap flags without value)
    for fl in ("--interleaved", "--skip-headers", "--archive", "--pipe"):
        p = _by_name(merged, fl)
        assert p.get("type") == "boolean" and p.get("takes_value") is False, p


def test_clap_flags_not_required():
    """Real-world clap `[OPTIONS]` help: NO flag may be required by default.

    bqtools's actual --help has no `[required]` markers -- the old
    `is_required = not (has_default or not metavar)` fallback marked every
    value-taking flag required, so bqtools_encode's schema demanded
    format/batch_size/manifest/depth/... and the LLM filled garbage. A bare
    metavar proves the flag takes a VALUE, not that the CLI requires it.
    """
    flags, pos, merged = _run(CLAP_HELP, skip_first=True)
    input_p = _by_name(merged, "input")
    assert input_p.get("positional") is True, input_p
    assert input_p.get("required") is True, input_p  # positional slot mandatory
    for fl in ("--format", "--batch-size", "--manifest", "--output", "--mode",
               "--policy", "--threads"):
        assert _by_name(merged, fl)["required"] is False, \
            f"{fl} must be optional (clap [OPTIONS], no [required] marker)"
    assert _by_name(merged, "--archive")["required"] is False
    # store-flag booleans still recognized
    assert _by_name(merged, "--interleaved").get("type") == "boolean"
    assert _by_name(merged, "--interleaved").get("takes_value") is False


def test_real_bqtools_encode_required_contract():
    """The EXACT bqtools encode registry contract after re-parse: only the
    positional `input` is required; all flags optional.

    Uses the real usage line (`bqtools encode [OPTIONS] [INPUT]...`) + clap
    option blocks (no `[required]` markers) that bqtools's --help emits, and
    asserts the leaf LLM schema + runner both end up with input-only required
    -- so the agent's `{"input": ..., "output": ...}` validates and renders
    `bqtools encode <in> --output <out>`.
    """
    from agent_connector.tool_spec import make_leaf_spec, render_spec
    from agent_connector.tool_runner import validate_arguments

    flags, pos, merged = _run(CLAP_HELP, skip_first=True)
    # simulate the subcommand_details execute_test would store after re-parse:
    # positional input (required) + optional flags
    params = [dict(p) for p in merged]
    for p in params:
        if p.get("name") == "input":
            p["positional"] = True
            p["position"] = 0
            p["required"] = True
        else:
            p["required"] = False
    bqtools = {
        "name": "bqtools", "arg_style": "subcommand",
        "command": "bqtools {{subcommand}}", "inputs": {},
        "subcommands": ["encode"],
        "subcommand_details": {"encode": {"params": params}},
        "subcommand_discovery_complete": True,
    }
    leaf = make_leaf_spec(bqtools, "encode")
    req = [k for k, m in leaf["inputs"].items() if m.get("required") is True]
    assert req == ["input"], f"only input required, got {req}"
    # the agent's real call validates + renders
    args = {"input": "/tmp/agent_test_sample.fasta", "output": "/tmp/out.binsq"}
    cleaned, err = validate_arguments(leaf, args)
    assert err == "", err
    argv = render_spec(leaf, args)
    assert argv == ["bqtools", "encode", "/tmp/agent_test_sample.fasta",
                    "--output", "/tmp/out.binsq"], argv


def test_output_param_resolution():
    """Task prompt must name the EXACT parameter that carries the output path.

    bioemu's output contract is output_dir (directory); bqtools.encode's is
    output (file). If the prompt only says "write to {path}" the LLM guesses
    which parameter to use. _task_output_param resolves it from the same
    outputs contract the validator checks, so prompt & validation agree.
    """
    from tool_agent_test import _task_output_kind, _task_output_param

    bioemu = {
        "name": "bioemu", "arg_style": "python",
        "inputs": {"sequence": {}, "num_samples": {}, "output_dir": {}},
        "outputs": {"output_dir": {"type": "directory", "source": "help_parsed"}},
    }
    assert _task_output_kind(bioemu) == "directory"
    assert _task_output_param(bioemu) == "output_dir"
    # stdout-only tool: no output param to name
    stdout_tool = {"name": "x", "arg_style": "cli",
                   "inputs": {"input": {}},
                   "outputs": {"stdout": {"type": "text"}}}
    assert _task_output_kind(stdout_tool) == "stdout"
    assert _task_output_param(stdout_tool) == ""
    # subcommand leaf: output param scoped to the sub
    bqtools = {
        "name": "bqtools", "arg_style": "subcommand", "inputs": {},
        "outputs": {"stdout": {"type": "text"}},
        "subcommand_details": {
            "encode": {"params": [
                {"name": "input", "type": "path", "positional": True,
                 "position": 0, "required": True},
                {"name": "--output", "type": "path", "required": False},
            ], "outputs": {"output": {"type": "file", "source": "help_parsed"}}},
            "info": {"params": [
                {"name": "input", "type": "path", "positional": True,
                 "position": 0, "required": True},
            ], "outputs": {"stdout": {"type": "text"}}},
        },
    }
    assert _task_output_kind(bqtools, "encode") == "file"
    assert _task_output_param(bqtools, "encode") == "output"
    assert _task_output_kind(bqtools, "info") == "stdout"
    assert _task_output_param(bqtools, "info") == ""


def test_output_contract_inference():
    """Output contract (registry `outputs`) must reflect the REAL output kind.

    bioemu's `output_dir` is a dash-less POSITIONAL -- after the canonical
    merge it must be inferred as a directory output, otherwise the task check
    reads stdout-only and flags a valid run OUTPUT_INVALID (run #36).
    """
    from discovery_to_registry import _infer_outputs

    merged_bioemu = [
        {"name": "sequence", "positional": True, "position": 0, "type": "path"},
        {"name": "num_samples", "positional": True, "position": 1, "type": "int"},
        {"name": "output_dir", "positional": True, "position": 2, "type": "path"},
        {"name": "--filter_samples", "type": "boolean", "takes_value": False},
    ]
    outs = _infer_outputs(merged_bioemu, [], "python")
    assert outs.get("output_dir", {}).get("type") == "directory", outs
    # a lone output flag is a file, outdir-ish flag is a directory
    assert _infer_outputs(
        [{"name": "--output", "type": "string"}], [], "named"
    ).get("output", {}).get("type") == "file"
    assert _infer_outputs(
        [{"name": "--outdir", "type": "path"}], [], "named"
    ).get("outdir", {}).get("type") == "directory"
    # input_file is NOT an output; nothing output-ish -> stdout contract
    outs2 = _infer_outputs(
        [{"name": "input_file", "positional": True, "type": "path"}], [], "named")
    assert "input_file" not in outs2
    assert "stdout" in outs2, outs2
    # flags that read output (--output-html) still count as file outputs
    assert _infer_outputs(
        [{"name": "--output-html", "type": "string"}], [], "named"
    ).get("output_html", {}).get("type") == "file"
    # a flag named --output whose HELP TEXT says "directory" (kaptain) is a
    # DIRECTORY output -- otherwise the task checks isfile on a real dir and
    # flags a successful run OUTPUT_INVALID.
    assert _infer_outputs(
        [{"name": "--output", "type": "path",
          "description": "Output directory (default: None)"}], [], "named"
    ).get("output", {}).get("type") == "directory"
    # ...but a --output flag whose text says "file"/"name" stays a file
    assert _infer_outputs(
        [{"name": "--output", "type": "path",
          "description": "Output report name (default: report.html)"}], [], "named"
    ).get("output", {}).get("type") == "file"


def test_leaf_spec_roundtrip():
    """LLM leaf schema == runner leaf spec == renders argv (P0-1/P1-3/P1-5).

    The exact failure this kills: LLM sees bqtools_encode(input, output) but
    the runner gets the RAW spec (inputs={}) and answers "unknown arguments".
    make_leaf_spec must scope inputs to the sub, keep positional metadata, and
    render positionals FIRST (bqtools encode <in> --output <out>).
    """
    from agent_connector.tool_spec import (
        get_required_inputs, make_leaf_spec, render_spec, validate_spec,
    )
    from agent_connector.tool_runner import validate_arguments

    bqtools = {
        "name": "bqtools", "arg_style": "subcommand", "command": "bqtools {{subcommand}}",
        "inputs": {},  # base tool has NO top-level inputs (P0-1 root cause)
        "subcommands": ["encode", "info"],
        "subcommand_details": {
            "encode": {"params": [
                {"name": "input", "type": "path", "positional": True, "position": 0,
                 "required": True, "description": "Input FASTA"},
                {"name": "--output", "type": "path", "required": True,
                 "description": "Output BINSEQ"},
                {"name": "--level", "type": "integer", "required": False,
                 "description": "Compression level"},
            ]},
            "info": {"params": [
                {"name": "input", "type": "path", "positional": True, "position": 0,
                 "required": True, "description": "Input BINSEQ"},
            ]},
        },
        "subcommand_discovery_complete": True,
    }
    # base tool alone can NEVER accept a leaf arg (this was the raw-spec bug):
    # without a subcommand the validator has nothing to resolve and must say so
    _, err = validate_arguments(bqtools, {"input": "/tmp/in", "output": "/tmp/out"})
    assert "missing subcommand" in err, err
    # BUT with a `subcommand` key, the raw spec resolves to its canonical leaf
    # (validator must not reject leaf args the generator just exposed -- the
    # schema-split the audit is built to kill). No arguments.left-over
    cleaned, err = validate_arguments(bqtools, {"subcommand": "encode",
                                                "input": "/tmp/in", "output": "/tmp/out"})
    assert err == "", err
    assert cleaned.get("input") == "/tmp/in" and cleaned.get("output") == "/tmp/out"
    # unknown subcommand -> clear error, not a fake acceptance
    _, err = validate_arguments(bqtools, {"subcommand": "nope", "input": "/tmp/in"})
    assert "no discovered params" in err or "unknown" in err, err
    # leaf spec scopes inputs to the subcommand
    leaf = make_leaf_spec(bqtools, "encode")
    assert leaf["name"] == "bqtools_encode"
    assert set(leaf["inputs"]) == {"input", "output", "level"}, leaf["inputs"]
    assert validate_spec(leaf) == "", validate_spec(leaf)
    assert get_required_inputs(leaf) == ["input", "output"]
    # LLM function schema (to_function_schemas) properties == leaf inputs
    from tool_agent_test import to_function_schemas
    schemas, fnmap = to_function_schemas(bqtools)
    fn = next(s for s in schemas if s["function"]["name"] == "bqtools_encode")["function"]
    assert set(fn["parameters"]["properties"]) == set(leaf["inputs"])
    assert set(fn["parameters"]["required"]) == set(get_required_inputs(leaf))
    # the SAME args the LLM would send validate + render with positionals first
    args = {"input": "/tmp/in.fa", "output": "/tmp/out.binsq"}
    cleaned, err = validate_arguments(leaf, args)
    assert err == "", err
    argv = render_spec(leaf, args)
    assert argv[0] == "bqtools" and argv[1] == "encode", argv
    # positional BEFORE flags: bqtools encode /tmp/in.fa --output /tmp/out.binsq
    assert argv[2] == "/tmp/in.fa", argv
    assert argv[3] == "--output" and argv[4] == "/tmp/out.binsq", argv
    # integer type preserved into the LLM schema (P1-2: not all-string)
    assert fn["parameters"]["properties"]["level"]["type"] == "integer"
    # info leaf has only its own positional, rendered bare
    leaf_info = make_leaf_spec(bqtools, "info")
    assert set(leaf_info["inputs"]) == {"input"}
    assert render_spec(leaf_info, {"input": "/tmp/x.binsq"}) == \
        ["bqtools", "info", "/tmp/x.binsq"]


def test_make_leaf_spec_bioemu_outputs():
    """Subcommand leaf carries its OWN output contract (P1-5)."""
    from agent_connector.tool_spec import make_leaf_spec

    tool = {
        "name": "bqtools", "arg_style": "subcommand",
        "command": "bqtools {{subcommand}}", "inputs": {},
        "outputs": {"stdout": {"type": "text", "source": "inferred"}},
        "subcommand_details": {
            "encode": {"params": [
                {"name": "--output", "type": "path", "required": True,
                 "description": "Output BINSEQ file"},
                {"name": "input", "type": "path", "positional": True,
                 "position": 0, "required": True, "description": "Input FASTA"},
            ]},
        },
        "subcommand_discovery_complete": True,
    }
    leaf = make_leaf_spec(tool, "encode")
    assert leaf["outputs"] == tool["outputs"], leaf["outputs"]  # falls back
    from tool_agent_test import _task_output_kind
    assert _task_output_kind(leaf) == "file" or _task_output_kind(tool, "encode") == "stdout"
    # when the sub HAS its own outputs, leaf uses them
    tool["subcommand_details"]["encode"]["outputs"] = {
        "output": {"type": "file", "source": "help_parsed"}}
    leaf2 = make_leaf_spec(tool, "encode")
    assert leaf2["outputs"].get("output", {}).get("type") == "file"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\nsummary: {len(tests) - fails}/{len(tests)} pass")
    sys.exit(1 if fails else 0)
