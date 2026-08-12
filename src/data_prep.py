"""
Stage 1: Sequence -> Numbers pipeline

Converts raw RNA sequences (A, U, G, C) into one-hot encoded numpy arrays
that a PyTorch model can consume.

Usage:
    from data_prep import one_hot_encode, load_sequences_from_fasta, build_dataset

    # Single sequence
    arr = one_hot_encode("AUGCUUAGGC")   # -> shape (seq_len, 4)

    # Whole dataset from a CSV of sequence,label pairs
    X, y = build_dataset("data/example_sequences.csv")
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Fixed nucleotide order -> index mapping. Keep this consistent everywhere.
NUCLEOTIDES = ["A", "U", "G", "C"]
NUC_TO_IDX = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}


def one_hot_encode(sequence: str, seq_len: int = 101) -> np.ndarray:
    """
    Convert a single RNA sequence string into a one-hot encoded array.

    Args:
        sequence: string of A/U/G/C (T is auto-converted to U, case-insensitive)
        seq_len: fixed length to pad/truncate to. All sequences in a batch
                  must be the same length for the CNN to work.

    Returns:
        np.ndarray of shape (seq_len, 4), dtype float32
    """
    sequence = sequence.upper().replace("T", "U")  # handle DNA-style input

    arr = np.zeros((seq_len, 4), dtype=np.float32)

    for i, nuc in enumerate(sequence[:seq_len]):
        if nuc in NUC_TO_IDX:
            arr[i, NUC_TO_IDX[nuc]] = 1.0
        # Unknown characters (N, etc.) are left as all-zero rows on purpose --
        # this tells the model "no information here" rather than guessing.

    return arr


def build_dataset(csv_path: str, seq_len: int = 101):
    """
    Load a CSV with columns ['sequence', 'label'] and convert to
    a full numeric dataset ready for training.

    label should be 1 (RBP binds here) or 0 (does not bind).

    Returns:
        X: np.ndarray shape (n_samples, seq_len, 4)
        y: np.ndarray shape (n_samples,)
    """
    df = pd.read_csv(csv_path)
    assert "sequence" in df.columns and "label" in df.columns, \
        "CSV must have 'sequence' and 'label' columns"

    X = np.stack([one_hot_encode(seq, seq_len) for seq in df["sequence"]])
    y = df["label"].values.astype(np.float32)

    return X, y


def save_dataset(X: np.ndarray, y: np.ndarray, out_dir: str, prefix: str = "dataset"):
    """Save processed arrays to disk so you don't re-run encoding every time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_X.npy", X)
    np.save(out_dir / f"{prefix}_y.npy", y)
    print(f"Saved: {out_dir / f'{prefix}_X.npy'}  shape={X.shape}")
    print(f"Saved: {out_dir / f'{prefix}_y.npy'}  shape={y.shape}")


def load_saved_dataset(out_dir: str, prefix: str = "dataset"):
    """Load previously saved .npy arrays."""
    out_dir = Path(out_dir)
    X = np.load(out_dir / f"{prefix}_X.npy")
    y = np.load(out_dir / f"{prefix}_y.npy")
    return X, y


if __name__ == "__main__":
    # Quick self-test with a couple of toy sequences
    test_seq = "AUGCUUAGGCAUGC"
    encoded = one_hot_encode(test_seq, seq_len=14)
    print("Test sequence:", test_seq)
    print("Encoded shape:", encoded.shape)
    print(encoded)
