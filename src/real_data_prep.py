"""
Stage: Real data pipeline (BED peaks + reference genome -> sequence,label CSV)

Takes real ENCODE eCLIP-seq peak coordinates (a BED file) and the
reference genome FASTA, and produces a CSV in the exact same format
as toy_sequences.csv -- so everything downstream (data_prep.py,
train_model.py, run_multi_seed.py) works unchanged.

WHAT YOU NEED BEFORE RUNNING THIS (see README section "Getting real data"):
1. A BED file of eCLIP peaks for one RBP, downloaded from encodeproject.org
2. The hg38 reference genome FASTA (or relevant species/assembly)

USAGE:
    python real_data_prep.py \\
        --bed path/to/peaks.bed \\
        --genome path/to/hg38.fa \\
        --out ../data/real_sequences.csv \\
        --seq_len 101 \\
        --n_negatives_per_positive 1
"""

import argparse
import random
import pandas as pd
from pyfaidx import Fasta


def extract_sequence(genome, chrom, center, seq_len, strand="+"):
    """
    Pull a fixed-length window from the genome, centered on `center`.
    Returns None if the window would run off the end of the chromosome
    (this happens near chromosome edges -- just skip those, they're rare).
    """
    half = seq_len // 2
    start = center - half
    end = start + seq_len

    if chrom not in genome:
        return None
    chrom_len = len(genome[chrom])
    if start < 0 or end > chrom_len:
        return None

    seq = genome[chrom][start:end].seq.upper()

    if strand == "-":
        # reverse complement for minus-strand peaks
        complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        seq = "".join(complement.get(b, "N") for b in reversed(seq))

    return seq


def gc_content(seq):
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def generate_negative(genome, chrom, positive_center, seq_len,
                       positive_gc, chrom_len, max_tries=50, gc_tolerance=0.05):
    """
    Pick a random location on the SAME chromosome, at least 500nt away
    from the positive site (to avoid accidentally grabbing real binding
    signal), and roughly GC-matched to the positive example.

    This is the negative-sampling logic your friend should sanity-check --
    it's the single biggest place genomics-ML pipelines go wrong.
    """
    half = seq_len // 2

    for _ in range(max_tries):
        candidate_center = random.randint(half, chrom_len - half)
        if abs(candidate_center - positive_center) < 500:
            continue  # too close to the real binding site, try again

        seq = extract_sequence(genome, chrom, candidate_center, seq_len)
        if seq is None or "N" in seq:
            continue

        if abs(gc_content(seq) - positive_gc) <= gc_tolerance:
            return seq

    return None  # gave up after max_tries -- caller should handle this


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True, help="Path to eCLIP peaks BED file")
    parser.add_argument("--genome", required=True, help="Path to reference genome FASTA")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--seq_len", type=int, default=101)
    parser.add_argument("--n_negatives_per_positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading genome: {args.genome}")
    genome = Fasta(args.genome)

    print(f"Loading peaks: {args.bed}")
    peaks = pd.read_csv(
        args.bed, sep="\t", header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
        usecols=range(6),
    )
    print(f"Found {len(peaks)} peaks")

    rows = []
    skipped = 0

    for _, peak in peaks.iterrows():
        chrom = peak["chrom"]
        center = (peak["start"] + peak["end"]) // 2
        strand = peak.get("strand", "+")

        pos_seq = extract_sequence(genome, chrom, center, args.seq_len, strand)
        if pos_seq is None or "N" in pos_seq:
            skipped += 1
            continue

        rows.append({"sequence": pos_seq, "label": 1})

        pos_gc = gc_content(pos_seq)
        chrom_len = len(genome[chrom]) if chrom in genome else 0

        for _ in range(args.n_negatives_per_positive):
            neg_seq = generate_negative(
                genome, chrom, center, args.seq_len, pos_gc, chrom_len
            )
            if neg_seq is not None:
                rows.append({"sequence": neg_seq, "label": 0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"\nSkipped {skipped} peaks (edge of chromosome or contained 'N')")
    print(f"Wrote {len(df)} rows ({(df['label']==1).sum()} positive, "
          f"{(df['label']==0).sum()} negative) to {args.out}")


if __name__ == "__main__":
    main()
