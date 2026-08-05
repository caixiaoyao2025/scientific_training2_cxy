"""Callable helper functions used by Python-execution tools in the registry."""

from pathlib import Path


def count_lines(file_path: str) -> dict:
    """Return the number of lines in a text file."""
    path = Path(file_path)
    with path.open(encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return {"file": str(path), "line_count": line_count, "bytes": path.stat().st_size}


def fasta_contig_stats(fasta_path: str) -> dict:
    """Count sequences and total bases in a FASTA file."""
    sequences = 0
    total_bases = 0
    with Path(fasta_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                sequences += 1
            else:
                total_bases += len(line.strip())
    return {"sequences": sequences, "total_bases": total_bases}


def split_pairwise_overlaps(a_bed_path: str, b_bed_path: str) -> dict:
    """Count per-record overlaps between two BED files with no dependency on bedtools."""
    b_intervals = []
    with Path(b_bed_path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                b_intervals.append((parts[0], int(parts[1]), int(parts[2])))
    overlaps = 0
    overlap_records = []
    with Path(a_bed_path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            for b_chrom, b_start, b_end in b_intervals:
                if chrom == b_chrom and start < b_end and b_start < end:
                    overlaps += 1
                    overlap_records.append(
                        f"{chrom}:{start}-{end}\t{b_chrom}:{b_start}-{b_end}"
                    )
                    break
    return {"overlaps": overlaps, "sample_overlaps": overlap_records[:10]}
