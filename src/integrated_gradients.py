"""
Stage 8: Gradient-based interpretability (Integrated Gradients), the
missing half of the paper's "attention-based vs. gradient-based"
comparison. Applies IG to the SAME cnn_lstm_attention model and the SAME
correctly-predicted test positives used in validate_pum2_motif.py, so
attention weights and IG attributions can be compared head-to-head on
identical predictions.

Integrated Gradients (Sundararajan et al. 2017): attribution for input
x w.r.t. baseline x' is

    IG(x) = (x - x') * integral_{alpha=0}^{1} dF(x' + alpha*(x-x')) / dx  d(alpha)

approximated with m Riemann steps. Baseline here is the all-zero one-hot
vector -- this matches the codebase's own convention (data_prep.py already
uses an all-zero row to mean "no information here" for unknown chars), so
"no sequence" is a principled, non-arbitrary baseline for this model.

Run:
    python integrated_gradients.py
"""

import argparse
import os
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
SPLIT_SEED = 42
SEEDS = [0, 1, 2, 3, 4]
SLACK = 2
MOTIF_LEN = 8
M_STEPS = 16          # IG integration steps (reduced for runtime budget in this session)
OUTER_BATCH = 256      # sequences per outer batch (expanded by M_STEPS internally)
CONF_THRESHOLD = 0.9   # only run IG on the high-confidence correctly-predicted positives
                        # (runtime budget -- full correctly-predicted set is ~5x larger)
OUT_CSV = "../outputs/pum2_ig_validation_full.csv"

MOTIF_REGEX = re.compile("TGTA.ATA")


def find_motif_spans(seq):
    return [(m.start(), m.start() + MOTIF_LEN) for m in MOTIF_REGEX.finditer(seq)]


def is_hit(pos, spans, slack=SLACK):
    return any((start - slack) <= pos <= (end - 1 + slack) for start, end in spans)


def integrated_gradients_batch(model, x, m_steps=M_STEPS):
    """
    x: (B, 4, seq_len) tensor, NOT requiring grad.
    Returns: attribution (B, seq_len) -- summed |IG| across the 4 channels.
    """
    B = x.shape[0]
    alphas = torch.linspace(1.0 / m_steps, 1.0, m_steps)  # (m,)
    # scaled inputs: (m, B, 4, L) -> (m*B, 4, L)   [baseline = 0, so interpolation = alpha * x]
    scaled = (alphas.view(-1, 1, 1, 1) * x.unsqueeze(0)).reshape(-1, *x.shape[1:])
    scaled.requires_grad_(True)

    logits = model(scaled)  # (m*B,)
    grads = torch.autograd.grad(logits.sum(), scaled)[0]  # (m*B, 4, L)

    grads = grads.reshape(m_steps, B, *x.shape[1:])
    avg_grad = grads.mean(dim=0)  # (B, 4, L)
    ig = avg_grad * x            # baseline = 0, so (x - baseline) = x
    attribution = ig.abs().sum(dim=1)  # (B, L) -- sum |attribution| across 4 nucleotide channels
    return attribution.detach()


def load_test_set():
    print(f"Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)
    X, y = build_dataset(DATA_CSV, seq_len=SEQ_LEN)
    full_dataset = RBPDataset(X, y)

    n = len(full_dataset)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    _, _, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    test_indices = test_ds.indices
    test_sequences = np.array([s.upper() for s in df["sequence"].values[test_indices]])
    test_labels = y[test_indices]
    X_test = torch.stack([full_dataset.X[i] for i in test_indices])
    return test_sequences, test_labels, X_test


def run_seed(seed, conf_threshold, m_steps, outer_batch, test_sequences, test_labels, X_test):
    device = torch.device("cpu")
    ckpt_path = f"../outputs/cnn_lstm_attention_seed{seed}.pt"
    model = CNNLSTMAttention(seq_len=SEQ_LEN).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_logits, all_attn = [], []
    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            xb = X_test[i:i + 512]
            logits, attn = model(xb, return_attention=True)
            all_logits.append(logits)
            all_attn.append(attn)
    logits = torch.cat(all_logits).numpy()
    attn = torch.cat(all_attn).numpy()
    probs = 1 / (1 + np.exp(-logits))

    correct_pos_mask = (test_labels == 1) & (probs > conf_threshold)
    idxs = np.where(correct_pos_mask)[0]
    print(f"=== seed {seed} === ({len(idxs)} correctly-predicted positives, p>{conf_threshold}, m_steps={m_steps})")

    ig_hits, ig_has_motif, ig_null_hits = 0, 0, 0
    agree_hits, agree_total = 0, 0
    ig_distances = []

    for start in range(0, len(idxs), outer_batch):
        batch_idxs = idxs[start:start + outer_batch]
        xb = X_test[batch_idxs]
        ig_attr = integrated_gradients_batch(model, xb, m_steps=m_steps).numpy()  # (b, L)
        ig_peaks = ig_attr.argmax(axis=1)

        rng = np.random.default_rng(seed * 100000 + start)
        for j, idx in enumerate(batch_idxs):
            seq = test_sequences[idx]
            spans = find_motif_spans(seq)
            ig_peak = int(ig_peaks[j])
            attn_peak = int(np.argmax(attn[idx]))

            if spans:
                ig_has_motif += 1
                if is_hit(ig_peak, spans):
                    ig_hits += 1
                nearest = min(abs(ig_peak - ((s + e - 1) / 2)) for s, e in spans)
                ig_distances.append(nearest)
                rand_pos = rng.integers(0, SEQ_LEN)
                if is_hit(rand_pos, spans):
                    ig_null_hits += 1

            agree_total += 1
            if abs(ig_peak - attn_peak) <= 5:
                agree_hits += 1

    result = {
        "seed": seed,
        "conf_threshold": conf_threshold,
        "m_steps": m_steps,
        "n": len(idxs),
        "n_with_motif": ig_has_motif,
        "ig_hit_rate_given_motif": ig_hits / ig_has_motif if ig_has_motif else float("nan"),
        "ig_null_hit_rate_given_motif": ig_null_hits / ig_has_motif if ig_has_motif else float("nan"),
        "ig_median_distance_nt": float(np.median(ig_distances)) if ig_distances else float("nan"),
        "attn_ig_agreement_rate": agree_hits / agree_total if agree_total else float("nan"),
    }
    print(f"  IG hit rate | motif present: {result['ig_hit_rate_given_motif']:.3f} "
          f"(null: {result['ig_null_hit_rate_given_motif']:.3f})")
    print(f"  IG median distance to motif: {result['ig_median_distance_nt']:.1f} nt")
    print(f"  Attention <-> IG agreement (peaks within 5nt): {result['attn_ig_agreement_rate']:.3f}")
    return result


def append_result(result, out_csv=OUT_CSV):
    df_row = pd.DataFrame([result])
    file_exists = os.path.exists(out_csv)
    df_row.to_csv(out_csv, mode="a", index=False, header=not file_exists)
    print(f"Appended result to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Which model seed to run (0-4)")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--m_steps", type=int, default=24)
    parser.add_argument("--outer_batch", type=int, default=256)
    args = parser.parse_args()

    test_sequences, test_labels, X_test = load_test_set()
    result = run_seed(
        args.seed, args.conf_threshold, args.m_steps, args.outer_batch,
        test_sequences, test_labels, X_test,
    )
    append_result(result)


if __name__ == "__main__":
    main()
