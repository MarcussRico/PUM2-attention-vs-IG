"""
Stage 9: Generate paper figures, in the spirit of RBPsuite 2.0's output
visualizations (per-nucleotide contribution score tracks, motif marking,
prediction score plots) -- but built around OUR actual contribution:
comparing attention vs. Integrated Gradients against the real PUM2 motif,
not just showing raw prediction scores.

Figures:
  fig1_roc_curves.png       -- ROC curves, 3 models, real PUM2 data (seed 0)
  fig2_auc_comparison.png   -- AUC-ROC/AUC-PR bar chart, 3 models, 5-seed mean+/-std
  fig3_example_tracks.png   -- per-nucleotide attention & IG tracks for 4 example
                                sequences, true motif position shaded (RBPsuite-style
                                per-sequence visualization, but comparing 2 methods)
  fig4_hitrate_comparison.png -- attention vs IG vs random-null hit rate, 5-seed
                                mean+/-std (the headline result, as a chart)
  fig5_distance_distribution.png -- attention vs IG: distribution of distance from
                                attribution peak to true motif center (seed 0, full set)
  fig6_agreement_scatter.png -- attention peak position vs IG peak position,
                                visualizing how rarely they agree

Run:
    python generate_figures.py
"""

import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from data_prep import build_dataset
from dataset import RBPDataset
from baseline_cnn import BaselineCNN
from cnn_lstm import CNNLSTM
from cnn_lstm_attention import CNNLSTMAttention
from integrated_gradients import integrated_gradients_batch, find_motif_spans, is_hit, MOTIF_LEN

DATA_CSV = "../data/real_sequences.csv"
SEQ_LEN = 101
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SPLIT_SEED = 42
FIG_DIR = "../outputs/figures"
SLACK = 2

import os
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {"baseline": "#7f8c8d", "cnn_lstm": "#2980b9", "cnn_lstm_attention": "#c0392b",
          "attention": "#c0392b", "ig": "#27ae60", "null": "#95a5a6"}


def load_test_set():
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


def manual_roc_curve(labels, scores):
    order = np.argsort(-scores)
    labels = labels[order]
    P = labels.sum()
    N = len(labels) - P
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / P
    fpr = fps / N
    return np.concatenate([[0], fpr, [1]]), np.concatenate([[0], tpr, [1]])


def auc_trapz(fpr, tpr):
    return np.trapz(tpr, fpr)


def fig1_and_2(test_sequences, test_labels, X_test):
    models = {"baseline": BaselineCNN, "cnn_lstm": CNNLSTM, "cnn_lstm_attention": CNNLSTMAttention}
    labels_pretty = {"baseline": "Baseline CNN", "cnn_lstm": "CNN-LSTM", "cnn_lstm_attention": "CNN-LSTM-Attention"}
    device = torch.device("cpu")

    # --- Fig 1: ROC curves, seed 0, representative run ---
    plt.figure(figsize=(5.5, 5))
    for name, cls in models.items():
        model = cls(seq_len=SEQ_LEN).to(device)
        model.load_state_dict(torch.load(f"../outputs/{name}_seed0.pt", map_location=device))
        model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                all_logits.append(model(X_test[i:i + 512]))
        logits = torch.cat(all_logits)
        probs = torch.sigmoid(logits).numpy()
        fpr, tpr = manual_roc_curve(test_labels, probs)
        auc = auc_trapz(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{labels_pretty[name]} (AUC={auc:.3f})", color=COLORS[name], linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Real PUM2 eCLIP Data (seed 0)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig1_roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig1_roc_curves.png")

    # --- Fig 2: AUC-ROC / AUC-PR bar chart, 5-seed mean+/-std, from results_log.csv ---
    df = pd.read_csv("../outputs/results_log.csv")
    order = ["baseline", "cnn_lstm", "cnn_lstm_attention"]
    means_roc = [df[df.model == m].test_auc_roc.mean() for m in order]
    stds_roc = [df[df.model == m].test_auc_roc.std() for m in order]
    means_pr = [df[df.model == m].test_auc_pr.mean() for m in order]
    stds_pr = [df[df.model == m].test_auc_pr.std() for m in order]

    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - width / 2, means_roc, width, yerr=stds_roc, capsize=4, label="AUC-ROC",
           color=[COLORS[m] for m in order], alpha=0.85)
    ax.bar(x + width / 2, means_pr, width, yerr=stds_pr, capsize=4, label="AUC-PR",
           color=[COLORS[m] for m in order], alpha=0.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([labels_pretty[m] for m in order], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Classification Performance — Real PUM2 Data\n(5-seed mean ± std)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig2_auc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig2_auc_comparison.png")


def per_sequence_ig_and_attention(seed=0, m_steps=24):
    """Recompute per-sequence attention + IG data for seed 0 (needed for figs 3, 5, 6 --
    the summary CSVs only kept aggregated stats, not per-sequence detail)."""
    test_sequences, test_labels, X_test = load_test_set()
    device = torch.device("cpu")
    model = CNNLSTMAttention(seq_len=SEQ_LEN).to(device)
    model.load_state_dict(torch.load(f"../outputs/cnn_lstm_attention_seed{seed}.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        all_logits, all_attn = [], []
        for i in range(0, len(X_test), 512):
            xb = X_test[i:i + 512]
            logits, attn = model(xb, return_attention=True)
            all_logits.append(logits)
            all_attn.append(attn)
        logits = torch.cat(all_logits).numpy()
        attn = torch.cat(all_attn).numpy()
    probs = 1 / (1 + np.exp(-logits))
    correct_pos_mask = (test_labels == 1) & (probs > 0.5)
    idxs = np.where(correct_pos_mask)[0]

    rows = []
    ig_curves = {}  # idx -> (ig_attribution array, attn array, seq, spans)
    for start in range(0, len(idxs), 256):
        batch_idxs = idxs[start:start + 256]
        xb = X_test[batch_idxs]
        ig_attr = integrated_gradients_batch(model, xb, m_steps=m_steps).numpy()
        for j, idx in enumerate(batch_idxs):
            seq = test_sequences[idx]
            spans = find_motif_spans(seq)
            ig_peak = int(ig_attr[j].argmax())
            attn_peak = int(attn[idx].argmax())
            row = {"idx": int(idx), "ig_peak": ig_peak, "attn_peak": attn_peak,
                   "has_motif": len(spans) > 0, "prob": float(probs[idx])}
            if spans:
                motif_center = min(spans, key=lambda sp: abs(ig_peak - (sp[0] + sp[1] - 1) / 2))
                mc = (motif_center[0] + motif_center[1] - 1) / 2
                row["motif_start"] = motif_center[0]
                row["motif_end"] = motif_center[1]
                row["ig_hit"] = is_hit(ig_peak, spans, SLACK)
                row["attn_hit"] = is_hit(attn_peak, spans, SLACK)
                row["ig_dist"] = abs(ig_peak - mc)
                row["attn_dist"] = abs(attn_peak - mc)
            rows.append(row)
            if spans:
                ig_curves[int(idx)] = (ig_attr[j].copy(), attn[idx].copy(), seq, spans)

    per_seq_df = pd.DataFrame(rows)
    per_seq_df.to_csv(f"../outputs/pum2_per_sequence_seed{seed}.csv", index=False)
    print(f"Saved per-sequence detail to ../outputs/pum2_per_sequence_seed{seed}.csv "
          f"({len(per_seq_df)} sequences, {per_seq_df.has_motif.sum()} with motif)")
    return per_seq_df, ig_curves


def fig3_example_tracks(ig_curves, per_seq_df):
    """Pick 4 motif-containing examples: 2 where attention and IG disagree sharply,
    2 where they roughly agree -- shows the range of behavior, not just cherry-picked
    disagreement."""
    with_motif = per_seq_df[per_seq_df.has_motif].copy()
    with_motif["peak_gap"] = (with_motif.ig_peak - with_motif.attn_peak).abs()
    disagree = with_motif.sort_values("peak_gap", ascending=False).head(2)
    agree = with_motif[with_motif.peak_gap <= 3].sort_values("prob", ascending=False).head(2)
    examples = pd.concat([disagree, agree])

    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.3 * len(examples)), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, examples.iterrows()):
        idx = int(row["idx"])
        ig_attr, attn_w, seq, spans = ig_curves[idx]
        positions = np.arange(SEQ_LEN)
        ig_norm = ig_attr / (ig_attr.max() + 1e-9)
        attn_norm = attn_w / (attn_w.max() + 1e-9)
        ax.plot(positions, attn_norm, color=COLORS["attention"], label="Attention", linewidth=1.6)
        ax.plot(positions, ig_norm, color=COLORS["ig"], label="Integrated Gradients", linewidth=1.6)
        for (s, e) in spans:
            ax.axvspan(s, e, color="gold", alpha=0.35, label="True PUM2 motif" if (s, e) == spans[0] else None)
        tag = "DISAGREE" if row["peak_gap"] > 5 else "agree"
        ax.set_ylabel("norm.\nimportance", fontsize=8)
        ax.set_title(f"Test seq #{idx}  (p={row['prob']:.2f}, peak gap={int(row['peak_gap'])}nt, {tag})",
                     fontsize=9, loc="left")
        ax.set_ylim(-0.05, 1.15)
    axes[-1].set_xlabel("Position in 101-nt window")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    plt.suptitle("Attention vs. Integrated Gradients: Per-Nucleotide Importance Tracks\n"
                 "(gold band = true PUM2 motif location; RBPsuite-style per-sequence visualization)",
                 fontsize=10, y=1.0)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_example_tracks.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig3_example_tracks.png")


def fig4_hitrate_comparison():
    attn_df = pd.read_csv("../outputs/pum2_motif_validation.csv")
    ig_df = pd.read_csv("../outputs/pum2_ig_validation_full.csv")

    methods = ["Random\nnull", "Attention", "Integrated\nGradients"]
    means = [attn_df.all_null_hit_rate_given_motif.mean(),
             attn_df.all_hit_rate_given_motif.mean(),
             ig_df.ig_hit_rate_given_motif.mean()]
    stds = [attn_df.all_null_hit_rate_given_motif.std(),
            attn_df.all_hit_rate_given_motif.std(),
            ig_df.ig_hit_rate_given_motif.std()]
    colors = [COLORS["null"], COLORS["attention"], COLORS["ig"]]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar(methods, means, yerr=stds, capsize=6, color=colors, alpha=0.85, width=0.6)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.03, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Hit rate (peak within true motif span)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Motif Localization: Attention vs. Integrated Gradients\n(real PUM2 data, 5-seed mean ± std)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_hitrate_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig4_hitrate_comparison.png")


def fig5_distance_distribution(per_seq_df):
    with_motif = per_seq_df[per_seq_df.has_motif]
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.arange(0, 52, 2)
    ax.hist(with_motif.attn_dist, bins=bins, alpha=0.55, label="Attention", color=COLORS["attention"])
    ax.hist(with_motif.ig_dist, bins=bins, alpha=0.55, label="Integrated Gradients", color=COLORS["ig"])
    ax.axvline(with_motif.attn_dist.median(), color=COLORS["attention"], linestyle="--", linewidth=1.5)
    ax.axvline(with_motif.ig_dist.median(), color=COLORS["ig"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Distance from attribution peak to true motif center (nt)")
    ax.set_ylabel("Number of test sequences")
    ax.set_title("Distribution of Localization Error\n(seed 0, full correctly-predicted set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig5_distance_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig5_distance_distribution.png")


def fig6_agreement_scatter(per_seq_df):
    with_motif = per_seq_df[per_seq_df.has_motif]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(with_motif.attn_peak, with_motif.ig_peak, s=10, alpha=0.35, color="#34495e")
    ax.plot([0, SEQ_LEN], [0, SEQ_LEN], "r--", linewidth=1, label="Perfect agreement")
    ax.set_xlabel("Attention peak position (nt)")
    ax.set_ylabel("Integrated Gradients peak position (nt)")
    corr = np.corrcoef(with_motif.attn_peak, with_motif.ig_peak)[0, 1]
    ax.set_title(f"Attention vs. IG Peak Position (seed 0)\nPearson r = {corr:.3f} — mostly disagree")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig6_agreement_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved fig6_agreement_scatter.png")


if __name__ == "__main__":
    test_sequences, test_labels, X_test = load_test_set()
    fig1_and_2(test_sequences, test_labels, X_test)
    per_seq_df, ig_curves = per_sequence_ig_and_attention(seed=0, m_steps=24)
    fig3_example_tracks(ig_curves, per_seq_df)
    fig4_hitrate_comparison()
    fig5_distance_distribution(per_seq_df)
    fig6_agreement_scatter(per_seq_df)
    print("\nAll figures saved to ../outputs/figures/")
