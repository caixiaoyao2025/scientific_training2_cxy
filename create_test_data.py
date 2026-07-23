"""Generate realistic simulated bioinformatics test data."""
import os, random

random.seed(42)
os.makedirs("data", exist_ok=True)

# ---- 1. sample.fastq (simulated paired-end WGS reads, E. coli-like) ----
bases = "ACGT"
base_qual = {b: 0 for b in bases}

def gen_qual(length, high_qual_prob=0.85):
    """Most bases high quality (Illumina), some errors. Phred+33: chars 33-104 (Q0-Q71)."""
    quals = []
    for _ in range(length):
        if random.random() < high_qual_prob:
            quals.append(chr(random.randint(53, 104)))  # Q20-Q71
        else:
            quals.append(chr(random.randint(33, 48)))    # Q0-Q15
    return "".join(quals)

def gen_seq(length):
    return "".join(random.choices(bases, weights=[0.25, 0.25, 0.25, 0.25], k=length))

with open("data/sample.fastq", "w") as f:
    for i in range(50):
        seq_len = random.choice([101, 151, 151])
        seq = gen_seq(seq_len)
        qual = gen_qual(seq_len, high_qual_prob=random.choice([0.9, 0.85, 0.7, 0.5]))
        f.write(f"@INST:{random.randint(1,8)}:FLOWCELL:{random.randint(1,4)}:{random.randint(100,999)}:{random.randint(10,999)}:{random.randint(10,999)} {i+1}:N:0:1\n")
        f.write(f"{seq}\n+\n{qual}\n")

# ---- 2. sample.sam (simulated alignments to E. coli reference) ----
# Header references E. coli K-12 MG1655 chromosome (4.6M bp)
ref_len = 4641652
with open("data/sample.sam", "w") as f:
    f.write("@HD\tVN:1.6\tSO:coordinate\n")
    f.write(f"@SQ\tSN:NC_000913.3\tLN:{ref_len}\n")
    f.write("@RG\tID:sample01\tSM:EC_K12\tPL:ILLUMINA\tLB:lib01\n")
    for i in range(30):
        pos = random.randint(1, ref_len - 200)
        mapq = random.choice([0, 10, 20, 30, 30, 40, 40, 60])
        flag = random.choice([0, 0, 0, 16])  # mostly forward strand
        seq_len = random.choice([101, 151])
        seq = gen_seq(seq_len)
        qual_str = gen_qual(seq_len, 0.9)
        cigar = f"{seq_len}M"
        f.write(f"read_{i+1}\t{flag}\tNC_000913.3\t{pos}\t{mapq}\t{cigar}\t*\t0\t0\t{seq}\t{qual_str}\tNM:i:0\n")

# ---- 3. sample.bed (simulated regulatory regions in E. coli) ----
regions = [
    ("lacZ", 366000, 369000),
    ("lacY", 369000, 370500),
    ("lacA", 370500, 371200),
    ("araB", 70000, 72500),
    ("araA", 72500, 74000),
    ("araD", 74000, 75500),
    ("recA", 2700000, 2701500),
    ("lexA", 4200000, 4201000),
    ("rpsA", 100000, 102000),
    ("dnaA", 2333000, 2336000),
    ("gyrA", 2323000, 2326000),
    ("gyrB", 2326000, 2328500),
    ("parC", 3585000, 3587500),
    ("parE", 3587500, 3590000),
    ("ompF", 1030000, 1033000),
    ("ompC", 1033000, 1036000),
    ("tonB", 1300000, 1301500),
    ("exbB", 1301500, 1303000),
    ("fur", 820000, 821200),
    ("fnr", 1370000, 1371500),
]
with open("data/sample.bed", "w") as f:
    f.write("track name=EC_regulatory description=E_coli_K12_regulatory_regions\n")
    for name, start, end in regions:
        score = random.randint(500, 1000)
        strand = random.choice(["+", "+", "-"])
        f.write(f"NC_000913.3\t{start}\t{end}\t{name}\t{score}\t{strand}\n")

# ---- 4. sample_annotation.gff (simulated gene annotations) ----
with open("data/sample_annotation.gff", "w") as f:
    f.write("##gff-version 3\n")
    for name, start, end in regions:
        strand = random.choice(["+", "-"])
        f.write(f"NC_000913.3\tSIMULATED\tgene\t{start+1}\t{end}\t.\t{strand}\t.\tID=gene:{name};Name={name}\n")

print(f"Created: sample.fastq ({os.path.getsize('data/sample.fastq')//1024}KB), "
      f"sample.sam ({os.path.getsize('data/sample.sam')//1024}KB), "
      f"sample.bed ({os.path.getsize('data/sample.bed')}B), "
      f"sample_annotation.gff ({os.path.getsize('data/sample_annotation.gff')}B)")
