"""
Stage 7: Real biological validation of the attention mechanism.

Question: on REAL PUM2 eCLIP data, does the CNN-LSTM+Attention model's
attention weight actually peak near the known PUM2 binding motif -- or
is it just noise that happens to get the classification right?

Ground truth motif: PUM2 Binding Element (PBE) consensus UGUANAUA
(N = any base), from White et al. 2001, RNA, PMID 11780640 (SELEX-derived
consensus [UGUANAUARNNNNBBBBSCCS], core 8-mer UGUANAUA is the
well-established minimal recognition element used throughout the
follow-up literature, e.g. PAR-CLIP/eCLIP studies).

This reuses the EXACT same test split as train_model.py (fixed
SPLIT_SEED=42) so we're only looking at genuinely held-out data, and
evaluates across all 5 "final" multi-seed checkpoints (seed 0-4) to
match the project's own multi-seed methodology rather than trusting a
single run.

For every correctly-predicted positive test sequence:
  1. Find the attention-weight argmax position.
  2. Regex-search the sequence for all occurrences of the degenerate
     PBE motif (TGTA.ATA in DNA alphabet).
  3. "Hit" = attention peak falls within a motif span (+/- 2nt slack,
     matching the toy-data check_attention.py convention).
  4. Compare against a random-position NULL baseline (uniform random
     position per sequence) to show attention does better than chance,
     not just "the motif is common near window center."

Run:
    python validate_pum2_motif.py
"""

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

from data_prep import build_dataset
from dataset import RBPDataset
from cnn_lstm_attention import CNNLSTMAttention

DATA_CSV = "../data/real_sequences.csv"
SEQ_LEN = 101
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SPLIT_SEED = 42  # must match train_model.py exactly
SEEDS = [0, 1, 2, 3, 4]
SLACK = 2
MOTIF_LEN = 8

# Degenerate PBE core motif UGUANAUA -> DNA alphabet, N = any base
MOTIF_REGEX = re.compile("TGTA.ATA")


def manual_auc_roc(labels, scores):
    """AUC-ROC via Mann-Whitney U, no sklearn dependency needed."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # tie correction via average ranks
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks_per_val = np.zeros(len(counts))
    np.add.at(sum_ranks_per_val, inv, ranks)
    avg_rank = sum_ranks_per_val[inv] / counts[inv]
    rank_sum_pos = avg_rank[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def find_motif_spans(seq):
    """All (start, end) spans of the degenerate PBE motif in this sequence."""
    return [(m.start(), m.start() + MOTIF_LEN) for m in MOTIF_REGEX.finditer(seq)]


def is_hit(pos, spans, slack=SLACK):
    return any((start - slack) <= pos <= (end - 1 + slack) for start, end in spans)


def main():
    print(f"Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)
    X, y = build_dataset(DATA_CSV, seq_len=SEQ_LEN)
    full_dataset = RBPDataset(X, y)

    n = len(full_dataset)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    test_indices = test_ds.indices
    print(f"Total: {n}  Test set: {len(test_indices)} sequences "
          f"(same split as train_model.py, SPLIT_SEED={SPLIT_SEED})")

    test_sequences = df["sequence"].values[test_indices]
    test_sequences = np.array([s.upper() for s in test_sequences])
    test_labels = y[test_indices]

    X_test = torch.stack([full_dataset.X[i] for i in test_indices])
    y_test = torch.tensor([full_dataset.y[i].item() for i in test_indices])

    device = torch.device("cpu")

    per_seed_summary = []

    for seed in SEEDS:
        ckpt_path = f"../outputs/cnn_lstm_attention_seed{seed}.pt"
        model = CNNLSTMAttention(seq_len=SEQ_LEN).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        all_logits, all_attn = [], []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                xb = X_test[i:i + batch_size].to(device)
                logits, attn = model(xb, return_attention=True)
                all_logits.append(logits)
                all_attn.append(attn)
        logits = torch.cat(all_logits).numpy()
        attn = torch.cat(all_attn).numpy()
        probs = 1 / (1 + np.exp(-logits))

        auc = manual_auc_roc(test_labels, probs)

        # Correctly-predicted positives (true label 1, predicted prob > 0.5)
        correct_pos_mask = (test_labels == 1) & (probs > 0.5)
        # Stricter high-confidence subset
        high_conf_mask = (test_labels == 1) & (probs > 0.9)

        def score_subset(mask, rng):
            idxs = np.where(mask)[0]
            hits = 0
            has_motif = 0
            null_hits = 0
            distances = []
            for idx in idxs:
                seq = test_sequences[idx]
                spans = find_motif_spans(seq)
                peak_pos = int(np.argmax(attn[idx]))
                if spans:
                    has_motif += 1
                    if is_hit(peak_pos, spans):
                        hits += 1
                    nearest = min(
                        abs(peak_pos - ((s + e - 1) / 2)) for s, e in spans
                    )
                    distances.append(nearest)
                    rand_pos = rng.integers(0, SEQ_LEN)
                    if is_hit(rand_pos, spans):
                        null_hits += 1
            n_eval = len(idxs)
            return {
                "n": n_eval,
                "n_with_motif": has_motif,
                "hit_rate_overall": hits / n_eval if n_eval else float("nan"),
                "hit_rate_given_motif": hits / has_motif if has_motif else float("nan"),
                "null_hit_rate_given_motif": null_hits / has_motif if has_motif else float("nan"),
                "median_distance_nt": float(np.median(distances)) if distances else float("nan"),
            }

        rng = np.random.default_rng(seed)
        result_all = score_subset(correct_pos_mask, rng)
        result_hc = score_subset(high_conf_mask, rng)

        print(f"\n=== seed {seed} ===")
        print(f"  Test AUC-ROC (sanity check vs reported 0.8197 +/- 0.0018): {auc:.4f}")
        print(f"  Correctly-predicted positives: {result_all['n']} "
              f"({result_all['n_with_motif']} contain the PBE motif)")
        print(f"  Attention hit rate | motif present: {result_all['hit_rate_given_motif']:.3f}  "
              f"(random-position null: {result_all['null_hit_rate_given_motif']:.3f})")
        print(f"  Median attention-peak-to-nearest-motif distance: {result_all['median_distance_nt']:.1f} nt")
        print(f"  High-confidence subset (p>0.9): n={result_hc['n']}, "
              f"hit rate | motif present: {result_hc['hit_rate_given_motif']:.3f} "
              f"(null: {result_hc['null_hit_rate_given_motif']:.3f})")

        per_seed_summary.append({
            "seed": seed, "auc": auc, **{f"all_{k}": v for k, v in result_all.items()},
            **{f"hc_{k}": v for k, v in result_hc.items()},
        })

    print(f"\n{'='*70}")
    print("SUMMARY ACROSS 5 SEEDS (correctly-predicted positives, motif present)")
    print(f"{'='*70}")
    hit_rates = [r["all_hit_rate_given_motif"] for r in per_seed_summary]
    null_rates = [r["all_null_hit_rate_given_motif"] for r in per_seed_summary]
    aucs = [r["auc"] for r in per_seed_summary]
    print(f"Test AUC-ROC:            {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"Attention hit rate:      {np.mean(hit_rates):.4f} +/- {np.std(hit_rates):.4f}")
    print(f"Random-position null:    {np.mean(null_rates):.4f} +/- {np.std(null_rates):.4f}")
    print(f"Lift over null:          {np.mean(hit_rates) - np.mean(null_rates):+.4f}")

    out = pd.DataFrame(per_seed_summary)
    out.to_csv("../outputs/pum2_motif_validation.csv", index=False)
    print("\nSaved per-seed results to ../outputs/pum2_motif_validation.csv")


if __name__ == "__main__":
    main()
