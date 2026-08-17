"""Fixture generator: create minimal sample files by artifact type.

Instead of feeding every tool a FASTA file, this creates the RIGHT kind
of test input for each parameter based on its artifact_type.

Each generator creates the smallest valid file possible — just enough
for the tool to parse it without crashing on format validation.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from agent_connector.artifact_spec import ARTIFACT_TYPES


# Minimal FASTA content (reused by multiple generators)
_MINIMAL_FASTA = ">seq1\nACGTACGT\n>seq2\nTTTTTTTT\n"
_MINIMAL_FASTQ = "@read1\nACGTACGT\n+\nIIIIIIII\n"
_MINIMAL_BED = "chr1\t100\t200\tfeature1\nchr1\t300\t400\tfeature2\n"
_MINIMAL_BEDPE = "chr1\t100\t200\tchr1\t300\t400\tfeature1\n"
_MINIMAL_VCF = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t100\t.\tA\tG\t.\tPASS\t.\n"
_MINIMAL_SAM = "@HD\tVN:1.6\tSO:coordinate\nread1\t0\tchr1\t100\t60\t4M\t*\t0\t0\tACGT\t*\n"
_MINIMAL_JSON = '{"key": "value"}\n'
_MINIMAL_TEXT = "col1\tcol2\tcol3\nval1\tval2\tval3\n"


def _write_fixture(path: str, content: str | bytes) -> str:
    """Write content to a file, creating parent dirs. Returns the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as f:
        f.write(content)
    return path


def generate_fixture(artifact_type: str, output_dir: str,
                     basename: str = "") -> str:
    """Generate a minimal fixture file for the given artifact type.

    Returns the path to the generated file. The file is the smallest
    valid example that a real tool can parse without format errors.

    For binary formats (.cool, .mcool, .bam), we create a marker file
    with a comment explaining that the real tool needs the actual format
    — the test harness catches the format error and reports it as a
    known limitation instead of a wrong-function failure.
    """
    info = ARTIFACT_TYPES.get(artifact_type, {})
    ext = info.get("fixture_ext", ".txt")
    if not basename:
        basename = "sample"
    path = os.path.join(output_dir, basename + ext)

    # Text-based formats: create minimal valid content
    if artifact_type == "fasta":
        return _write_fixture(path, _MINIMAL_FASTA)
    if artifact_type == "fastq":
        return _write_fixture(path, _MINIMAL_FASTQ)
    if artifact_type == "bed":
        return _write_fixture(path, _MINIMAL_BED)
    if artifact_type == "bedpe":
        return _write_fixture(path, _MINIMAL_BEDPE)
    if artifact_type == "vcf":
        return _write_fixture(path, _MINIMAL_VCF)
    if artifact_type == "sam":
        return _write_fixture(path, _MINIMAL_SAM)
    if artifact_type == "text":
        return _write_fixture(path, _MINIMAL_TEXT)
    if artifact_type == "json":
        return _write_fixture(path, _MINIMAL_JSON)

    # Binary formats: we can't create real .cool/.mcool/.bam in pure Python
    # without heavy deps. Create a marker file so the test harness knows
    # this fixture is a placeholder.
    if artifact_type in ("cool", "mcool", "bam", "bigwig"):
        marker = (f"# placeholder for {artifact_type} format\n"
                  f"# real tool needs actual binary format\n")
        return _write_fixture(path, marker)

    # Fallback: minimal text file
    return _write_fixture(path, _MINIMAL_TEXT)


def generate_fixtures_for_tool(tool: dict, output_dir: str) -> dict[str, str]:
    """Generate one fixture per input parameter that has an artifact type.

    Returns {param_key: fixture_path} for inputs that need file fixtures.
    Inputs without artifact_type (e.g. integer flags) are skipped.
    """
    fixtures = {}
    for sub_name, detail in (tool.get("subcommand_details") or {}).items():
        for key, meta in (detail.get("inputs") or {}).items():
            artifact = meta.get("artifact")
            if not artifact or artifact == "directory":
                continue
            fixture_path = generate_fixture(
                artifact, output_dir,
                basename=f"{tool.get('name', 'tool')}_{sub_name}_{key}")
            fixtures[key] = fixture_path
    # non-subcommand tools
    if tool.get("arg_style") != "subcommand":
        for key, meta in (tool.get("inputs") or {}).items():
            artifact = meta.get("artifact")
            if not artifact or artifact == "directory":
                continue
            fixture_path = generate_fixture(
                artifact, output_dir,
                basename=f"{tool.get('name', 'tool')}_{key}")
            fixtures[key] = fixture_path
    return fixtures
