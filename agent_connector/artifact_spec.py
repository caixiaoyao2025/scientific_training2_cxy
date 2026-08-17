"""ArtifactSpec: describes what scientific data format a ToolSpec input expects.

Extends the existing ToolSpec inputs with artifact_type, extensions, and
semantic description so that:
  1. The LLM knows what data format to provide (not just "string")
  2. The test harness generates correct fixtures (FASTA for bqtools,
     .cool/.mcool for cooltools, .bed for features, etc.)
  3. The runner can validate file extensions before execution

Design:
  - Each ToolSpec input can optionally carry an ``artifact`` field
  - Artifact types are a controlled vocabulary (not free text)
  - Each artifact type maps to: extensions, fixture generator, description
  - The agent test reads artifacts to pick the right fixture per parameter
"""
from __future__ import annotations

from typing import Any


# Controlled vocabulary of artifact types. Each entry maps to:
#   extensions: file extensions that match this artifact
#   description: human/LLM-readable explanation
#   fixture_ext: default extension for generated fixtures
#   fixture_content: generator function name (in fixture_generator.py)
ARTIFACT_TYPES: dict[str, dict] = {
    # --- Sequence formats ---
    "fasta": {
        "extensions": [".fasta", ".fa", ".fna", ".faa"],
        "description": "DNA/RNA/protein sequences in FASTA format",
        "fixture_ext": ".fasta",
    },
    "fastq": {
        "extensions": [".fastq", ".fq"],
        "description": "Sequencing reads in FASTQ format",
        "fixture_ext": ".fastq",
    },
    "binseq": {
        "extensions": [".binseq", ".vbq", ".cbq", ".bq"],
        "description": "Compressed sequence archive in BINSEQ format",
        "fixture_ext": ".vbq",
    },
    # --- Hi-C / contact matrix formats ---
    "cool": {
        "extensions": [".cool"],
        "description": "Hi-C contact matrix in .cool format (cooltools/snipping)",
        "fixture_ext": ".cool",
    },
    "mcool": {
        "extensions": [".mcool", ".cool"],
        "description": "Hi-C contact matrix in .mcool/.cool format",
        "fixture_ext": ".mcool",
    },
    # --- Genomic feature formats ---
    "bed": {
        "extensions": [".bed", ".bed.gz"],
        "description": "Genomic features/regions in BED format",
        "fixture_ext": ".bed",
    },
    "bedpe": {
        "extensions": [".bedpe"],
        "description": "Paired-end genomic features in BEDPE format",
        "fixture_ext": ".bedpe",
    },
    "bigwig": {
        "extensions": [".bw", ".bigwig"],
        "description": "Genomic signal in bigWig format",
        "fixture_ext": ".bw",
    },
    # --- Structure formats ---
    "pdb": {
        "extensions": [".pdb"],
        "description": "Protein structure in PDB format",
        "fixture_ext": ".pdb",
    },
    "mmcif": {
        "extensions": [".cif", ".mmcif"],
        "description": "Protein structure in mmCIF format",
        "fixture_ext": ".cif",
    },
    # --- Alignment formats ---
    "bam": {
        "extensions": [".bam"],
        "description": "Aligned reads in BAM format",
        "fixture_ext": ".bam",
    },
    "sam": {
        "extensions": [".sam"],
        "description": "Aligned reads in SAM format",
        "fixture_ext": ".sam",
    },
    # --- Variant formats ---
    "vcf": {
        "extensions": [".vcf", ".vcf.gz"],
        "description": "Variant calls in VCF format",
        "fixture_ext": ".vcf",
    },
    # --- Generic ---
    "text": {
        "extensions": [".txt", ".tsv", ".csv"],
        "description": "Plain text / tabular data",
        "fixture_ext": ".txt",
    },
    "json": {
        "extensions": [".json"],
        "description": "JSON data",
        "fixture_ext": ".json",
    },
    "directory": {
        "extensions": [],
        "description": "Output directory",
        "fixture_ext": "",
    },
}


def infer_artifact_from_description(desc: str) -> str | None:
    """Try to infer artifact type from a parameter description string.

    Returns an ARTIFACT_TYPES key or None if no match.
    Uses word-boundary matching to avoid false positives like
    'threads' matching 'fastq'."""
    import re as _re
    d = (desc or "").lower()
    # order matters: more specific first. Each pattern uses word boundaries
    # to avoid substring false positives.
    patterns = [
        (r"\.mcool\b|mcool\b", "mcool"),
        (r"\.cool\b|contact\s+matrix|hic\b", "cool"),
        (r"\.bed\b|bed\s+file|bedpe\b|genomic\s+feature", "bed"),
        (r"\.bam\b|bam\s+file", "bam"),
        (r"\.sam\b|sam\s+file", "sam"),
        (r"\.vcf\b|variant\s+call", "vcf"),
        (r"\.fasta\b|fasta\s+file|sequence\s+file", "fasta"),
        (r"\.fastq\b|fastq\s+file|sequencing\s+read", "fastq"),
        (r"\.pdb\b|protein\s+structure", "pdb"),
        (r"\.bw\b|bigwig|signal\s+track", "bigwig"),
        (r"\.json\b", "json"),
        (r"output\s+(directory|dir)", "directory"),
    ]
    for pat, artifact in patterns:
        if _re.search(pat, d):
            return artifact
    return None


def annotate_inputs_with_artifacts(inputs: dict[str, Any],
                                   tool_name: str = "",
                                   sub_name: str = "") -> dict[str, Any]:
    """Add artifact_type to inputs that don't already have one.

    Uses a three-layer strategy:
      1. Explicit annotation already present → skip
      2. Tool-specific rules (bqtools input → binseq, etc.)
      3. Description-based inference (regex on param description)
    Only annotates string/path type parameters (not int/bool/float flags).
    Returns the mutated inputs dict."""
    for key, meta in inputs.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("artifact"):
            continue  # already annotated
        # Only annotate string/path type parameters
        ptype = (meta.get("type") or "string").lower()
        if ptype in ("integer", "int", "float", "number", "boolean", "bool"):
            continue
        # Layer 2: tool-specific rules
        artifact = _tool_specific_artifact(tool_name, sub_name, key, meta)
        if not artifact:
            # Layer 3: description inference
            desc = meta.get("description") or ""
            artifact = infer_artifact_from_description(desc)
        if artifact:
            meta["artifact"] = artifact
            meta["artifact_info"] = ARTIFACT_TYPES.get(artifact, {})
    return inputs


# Tool-specific artifact annotations. Maps (tool_name, param_name) or
# (tool_name, sub_name, param_name) to an artifact type. These are the
# ground-truth annotations that description inference can't reach.
_TOOL_ARTIFACTS: dict[tuple[str, ...], str] = {
    # bqtools: all subcommands take BINSEQ as input
    ("bqtools", "input"): "binseq",
    # bqtools encode: input is FASTA/FASTQ
    ("bqtools", "encode", "input"): "fasta",
    # bqtools decode: input is BINSEQ
    ("bqtools", "decode", "input"): "binseq",
    # bqtools grep: input is BINSEQ
    ("bqtools", "grep", "input"): "binseq",
    # bqtools split: input is BINSEQ
    ("bqtools", "split", "input"): "binseq",
    # bqtools sample: input is BINSEQ
    ("bqtools", "sample", "input"): "binseq",
    # bqtools cat: input is BINSEQ
    ("bqtools", "cat", "input"): "binseq",
    # bqtools revcomp: input is BINSEQ
    ("bqtools", "revcomp", "input"): "binseq",
    # bqtools verify: input is BINSEQ
    ("bqtools", "verify", "input"): "binseq",
    # bqtools info: input is BINSEQ
    ("bqtools", "info", "input"): "binseq",
    # bqtools grep pattern/reg params
    ("bqtools", "grep", "reg"): "text",
    # bioemu
    ("bioemu", "sequence"): "fasta",
    # kaptain
    ("kaptain", "ont_in"): "fastq",
}


def _tool_specific_artifact(tool_name: str, sub_name: str,
                            param_key: str, meta: dict) -> str | None:
    """Look up artifact type from tool-specific rules."""
    # Try (tool, sub, param) first, then (tool, param)
    key3 = (tool_name, sub_name, param_key)
    key2 = (tool_name, param_key)
    return _TOOL_ARTIFACTS.get(key3) or _TOOL_ARTIFACTS.get(key2)


def artifact_for_param(spec: dict, param_key: str) -> str | None:
    """Get the artifact type for a specific parameter from a leaf ToolSpec."""
    inputs = spec.get("inputs") or {}
    meta = inputs.get(param_key) or {}
    return meta.get("artifact")
