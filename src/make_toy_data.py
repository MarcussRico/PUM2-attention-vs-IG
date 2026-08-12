"""
Generates a small SYNTHETIC dataset so we can test the pipeline and model
code right now, before real ENCODE eCLIP-seq data is downloaded.

This is NOT real biological data -- it's fake sequences with a made-up
"motif" (GCAUG) planted in positive examples, and random sequence for
negatives. Once your friend pulls the real ENCODE data, you swap this
file out for the real CSV and nothing else in the pipeline changes.
"""

import random
import pandas as pd
from pathlib import Path

random.seed(42)

NUCLEOTIDES = ["A", "U", "G", "C"]
MOTIF = "GCAUG"  # pretend this is a known RBP binding motif (PUM2-like)
SEQ_LEN = 101
N_SAMPLES_PER_CLASS = 500


def random_sequence(length: int) -> str:
    return "".join(random.choice(NUCLEOTIDES) for _ in range(length))


def make_positive_sequence() -> str:
    """Random sequence with the motif planted at a random position."""
    seq = list(random_sequence(SEQ_LEN))
    insert_pos = random.randint(0, SEQ_LEN - len(MOTIF))
    seq[insert_pos:insert_pos + len(MOTIF)] = list(MOTIF)
    return "".join(seq)


def make_negative_sequence() -> str:
    """Pure random sequence, no motif (roughly -- small chance it appears by luck)."""
    return random_sequence(SEQ_LEN)


def main():
    rows = []
    for _ in range(N_SAMPLES_PER_CLASS):
        rows.append({"sequence": make_positive_sequence(), "label": 1})
    for _ in range(N_SAMPLES_PER_CLASS):
        rows.append({"sequence": make_negative_sequence(), "label": 0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = Path(__file__).parent.parent / "data" / "toy_sequences.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
