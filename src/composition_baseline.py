"""
Composition-only baseline.

Fits a logistic regression on mononucleotide and dinucleotide frequencies alone
-- no positional information, no sequence order, no motif. Whatever AUC this
reaches is the floor a sequence model must clear before any of its performance
can be attributed to learning the binding motif rather than base composition.

This is the control for the negative-sampling confound: if the composition-only
baseline is close to the full model's AUC, the classifier is mostly separating
classes on composition and any interpretability claim built on it is unsafe.

Run:
    python composition_baseline.py
    python composition_baseline.py --data ../data/real_sequences_v1_chromosome_negatives.csv

Writes/appends ../outputs/composition_baseline.csv
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DINUCS = ["".join(p) for p in itertools.product("ACGT", repeat=2)]
DI_IX = {d: i for i, d in enumerate(DINUCS)}
BASES = "ACGT"


def featurise(seqs, seq_len=101):
    """20 features per sequence: 16 dinucleotide + 4 mononucleotide frequencies."""
    X = np.zeros((len(seqs), 20), dtype=np.float32)
    for r, s in enumerate(seqs):
        for i in range(len(s) - 1):
            j = DI_IX.get(s[i:i + 2])
            if j is not None:
                X[r, j] += 1
        for k, b in enumerate(BASES):
            X[r, 16 + k] = s.count(b)
    return X / float(seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/real_sequences.csv")
    ap.add_argument("--out", default="../outputs/composition_baseline.csv")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default=None,
                    help="name for this dataset in the results log")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"{args.data}: {len(df):,} sequences "
          f"({(df.label == 1).sum():,} positive, {(df.label == 0).sum():,} negative)")

    X = featurise(df.sequence.values)
    y = df.label.values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=3000).fit(scaler.transform(Xtr), ytr)
    prob = model.predict_proba(scaler.transform(Xte))[:, 1]

    auc_roc = roc_auc_score(yte, prob)
    auc_pr = average_precision_score(yte, prob)
    print(f"\nComposition-only AUC-ROC : {auc_roc:.4f}")
    print(f"Composition-only AUC-PR  : {auc_pr:.4f}")

    order = np.argsort(-np.abs(model.coef_[0]))[:5]
    names = DINUCS + list(BASES)
    print("\nStrongest composition features:")
    for i in order:
        print(f"  {names[i]:<4} coefficient {model.coef_[0][i]:+.3f}")

    print("\nInterpretation: a sequence model's AUC above this value is what it "
          "gained\nfrom sequence order and motif content rather than base composition.")

    row = {
        "dataset": args.label or os.path.basename(args.data),
        "n_sequences": len(df),
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "test_size": args.test_size,
        "seed": args.seed,
    }
    exists = os.path.exists(args.out)
    pd.DataFrame([row]).to_csv(args.out, mode="a", header=not exists, index=False)
    print(f"\nLogged to {args.out}")


if __name__ == "__main__":
    main()
