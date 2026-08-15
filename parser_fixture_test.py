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
    # store-flag booleans (clap flags without value)
    for fl in ("--interleaved", "--skip-headers", "--archive", "--pipe"):
        p = _by_name(merged, fl)
        assert p.get("type") == "boolean" and p.get("takes_value") is False, p


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
