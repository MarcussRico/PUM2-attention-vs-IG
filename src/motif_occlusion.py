"""
Causal test: does the model's prediction actually depend on the PBE motif?

Attention and Integrated Gradients both say *where* a model is looking. Neither
shows the model's decision depends on what is there. This does, by intervention:

  1. take correctly-predicted positives that contain an exact PBE (TGTA.ATA)
  2. shuffle the 8 nucleotides of the motif in place (composition preserved,
     motif destroyed) and re-predict
  3. as a control, shuffle a random 8-nt window elsewhere in the same sequence

If the prediction collapses on (2) but not (3), the model causally depends on
the motif and the attribution methods are pointing at something real. If both
drops are similar, the model is using diffuse context and a high motif hit rate
means only that the motif co-occurs with what the model actually uses.

Run on any experimental cell:
    python motif_occlusion.py --data ../data/real_sequences.csv --model-dir ../outputs
    python motif_occlusion.py --data ../data/real_sequences_cellA_chrom_centred.csv \\
                              --model-dir ../outputs_cellA
"""

import argparse
import os
import re

import numpy as np
import torch
from torch.utils.data import random_split

from cnn_lstm_attention import CNNLSTMAttention
from data_prep import build_dataset, one_hot_encode
from dataset import RBPDataset

SEQ_LEN = 101
VAL_SPLIT, TEST_SPLIT = 0.15, 0.15
SPLIT_SEED = 42          # must match train_model.py
SEEDS = [0, 1, 2, 3, 4]
MOTIF_LEN = 8
MOTIF_REGEX = re.compile("TGTA.ATA")


def one_hot(seqs):
    """Encode via the project's own encoder.

    Do NOT reimplement this. The codebase orders nucleotides A,U,G,C; writing a
    fresh A,C,G,U mapping here silently swaps C and U, leaves every prediction
    subtly wrong, and produces plausible-looking but meaningless numbers.
    """
    X = np.stack([one_hot_encode(s, seq_len=SEQ_LEN) for s in seqs])  # (N, L, 4)
    return torch.from_numpy(X).permute(0, 2, 1).contiguous()          # (N, 4, L)


def predict(model, X, device, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits, _ = model(X[i:i + batch].to(device), return_attention=True)
            out.append(logits.cpu())
    logits = torch.cat(out).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def shuffled(sub, rng):
    """Shuffle a substring until it differs from the original where possible."""
    chars = list(sub)
    for _ in range(20):
        rng.shuffle(chars)
        if "".join(chars) != sub:
            break
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/real_sequences.csv")
    ap.add_argument("--model-dir", default="../outputs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or os.path.basename(os.path.normpath(args.model_dir))

    device = torch.device("cpu")
    X_all, y_all = build_dataset(args.data, seq_len=SEQ_LEN)
    import pandas as pd
    seqs_all = pd.read_csv(args.data).sequence.str.upper().values

    ds = RBPDataset(X_all, y_all)
    n = len(ds)
    n_val, n_test = int(n * VAL_SPLIT), int(n * TEST_SPLIT)
    _, _, test_ds = random_split(
        ds, [n - n_val - n_test, n_val, n_test],
        generator=torch.Generator().manual_seed(SPLIT_SEED))
    idx = np.array(test_ds.indices)
    seqs, labels = seqs_all[idx], y_all[idx]

    rows = []
    for seed in SEEDS:
        ckpt = os.path.join(args.model_dir, f"cnn_lstm_attention_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  seed {seed}: no checkpoint, skipping")
            continue
        model = CNNLSTMAttention(seq_len=SEQ_LEN).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        probs = predict(model, one_hot(seqs), device)
        keep = np.where((labels == 1) & (probs > 0.5))[0]
        n_motif = sum(1 for i in keep if MOTIF_REGEX.search(seqs[i]))
        if n_motif < 100:
            print(f"  WARNING seed {seed}: only {n_motif} motif-containing correct "
                  f"positives ({len(keep)} correct positives total). "
                  f"validate_pum2_motif.py reports ~700 -- encoding or split mismatch?")

        rng = np.random.default_rng(1000 + seed)
        orig, occ_motif, occ_ctrl, occ_gc = [], [], [], []
        for i in keep:
            s = seqs[i]
            m = MOTIF_REGEX.search(s)
            if not m:
                continue
            a, b = m.start(), m.start() + MOTIF_LEN
            orig.append(s)
            occ_motif.append(s[:a] + shuffled(s[a:b], rng) + s[b:])
            # control window of the same length, not overlapping the motif
            for _ in range(50):
                c = int(rng.integers(0, SEQ_LEN - MOTIF_LEN))
                if c + MOTIF_LEN <= a or c >= b:
                    break
            occ_ctrl.append(s[:c] + shuffled(s[c:c + MOTIF_LEN], rng) + s[c + MOTIF_LEN:])
            gc = "".join(rng.choice(list("GC"), MOTIF_LEN))
            occ_gc.append(s[:a] + gc + s[b:])

        if not orig:
            continue
        p0 = predict(model, one_hot(orig), device)
        p1 = predict(model, one_hot(occ_motif), device)
        p2 = predict(model, one_hot(occ_ctrl), device)
        p3 = predict(model, one_hot(occ_gc), device)

        rows.append(dict(
            cell=label, seed=seed, n=len(orig),
            p_original=p0.mean(),
            p_motif_shuffled=p1.mean(),
            p_control_shuffled=p2.mean(),
            drop_motif=(p0 - p1).mean(),
            drop_control=(p0 - p2).mean(),
            p_motif_to_GC=p3.mean(),
            drop_motif_to_GC=(p0 - p3).mean(),
            excess_drop=((p0 - p1) - (p0 - p2)).mean(),
            frac_flipped_motif=float(np.mean(p1 < 0.5)),
            frac_flipped_control=float(np.mean(p2 < 0.5)),
        ))
        r = rows[-1]
        print(f"  seed {seed}: n={r['n']:<5} p {r['p_original']:.3f} -> "
              f"motif-shuffled {r['p_motif_shuffled']:.3f} "
              f"(drop {r['drop_motif']:+.3f}), control {r['p_control_shuffled']:.3f} "
              f"(drop {r['drop_control']:+.3f})")

    if not rows:
        print("no results")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n{'=' * 68}\nMOTIF OCCLUSION — {label}\n{'=' * 68}")
    for k, name in [("drop_motif", "drop when the PBE is shuffled (order destroyed)"),
                    ("drop_motif_to_GC", "drop when the PBE -> GC-rich 8-mer"),
                    ("drop_control", "drop when a random 8-mer is shuffled"),
                    ("excess_drop", "excess drop attributable to the motif"),
                    ("frac_flipped_motif", "fraction pushed below p=0.5 by motif")]:
        v = df[k].values
        print(f"  {name:<42}{v.mean():+.4f} ± {v.std(ddof=1):.4f}")
    try:
        from scipy import stats
        t, p = stats.ttest_rel(df.drop_motif, df.drop_control)
        print(f"\n  motif vs control, paired across seeds: t={t:.2f}  p={p:.4f}")
        print("  -> " + ("model causally depends on the motif" if p < 0.05 and df.excess_drop.mean() > 0
                         else "no detectable causal dependence beyond a generic 8-mer disruption"))
    except Exception:
        pass

    out = args.out or os.path.join(args.model_dir, "motif_occlusion.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
