"""
Figures for the 2x2 benchmark-design study.

Produces:
  fig7_design_2x2.png       attention motif localisation across the four designs
  fig8_occlusion.png        causal motif-occlusion test
  fig9_shortcuts.png        the two shortcuts, measured

Run after run_ablation.py and motif_occlusion.py have completed all cells:
    python generate_design_figures.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = "../outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

NAVY, BLUE, RED, GREEN, GREY = "#0B3C5D", "#1D71A8", "#C0504D", "#3E9B6C", "#9AA5AE"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

CELLS = [
    ("A", "chromosome", "peak-centered", "../outputs_cellA"),
    ("B", "region-matched", "peak-centered", "../outputs_ablation"),
    ("C", "chromosome", "random-padded", "../outputs_cellC"),
    ("D", "region-matched", "random-padded", "../outputs"),
]
DATASETS = {
    "A": "../data/real_sequences_cellA_chrom_centred.csv",   # file on disk keeps its original name
    "B": "../data/real_sequences_ablation_fixedcenter.csv",
    "C": "../data/real_sequences_cellC_chrom_padded.csv",
    "D": "../data/real_sequences.csv",
}


def load():
    hit, null, auc, occ = {}, {}, {}, {}
    for k, _, _, d in CELLS:
        v = pd.read_csv(f"{d}/pum2_motif_validation.csv")
        hit[k] = v.all_hit_rate_given_motif.values
        null[k] = v.all_null_hit_rate_given_motif.values
        r = pd.read_csv(f"{d}/results_log.csv")
        r = r[r.model == "cnn_lstm_attention"].drop_duplicates("seed")
        auc[k] = r.test_auc_roc.values
        f = f"{d}/motif_occlusion.csv"
        if os.path.exists(f):
            occ[k] = pd.read_csv(f)
    return hit, null, auc, occ


# ---------------------------------------------------------------- fig 7
def fig_2x2(hit, null, auc):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3),
                             gridspec_kw={"width_ratios": [1.2, 1]},
                             constrained_layout=True)
    ax = axes[0]
    groups = ["chromosome", "region-matched"]
    x = np.arange(len(groups))
    w = 0.36
    for i, (pos_style, colour, key_by_group) in enumerate([
            ("peak-centered", RED, {"chromosome": "A", "region-matched": "B"}),
            ("random-padded", GREEN, {"chromosome": "C", "region-matched": "D"})]):
        means = [hit[key_by_group[g]].mean() for g in groups]
        errs = [hit[key_by_group[g]].std(ddof=1) for g in groups]
        bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=4,
                      label=pos_style, color=colour, edgecolor="white", linewidth=0.8)
        for b, m, g in zip(bars, means, groups):
            ax.text(b.get_x() + b.get_width() / 2, m + errs[groups.index(g)] + 0.03,
                    f"{m:.2f}", ha="center", fontweight="bold", fontsize=10)
    nullmean = np.mean([null[k].mean() for k in null])
    ax.axhline(nullmean, ls="--", lw=1.2, color=GREY)
    ax.text(0.5, nullmean + 0.045, "random-position null", ha="center",
            fontsize=8.5, color="#6b7680", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_xlabel("negative sampling"); ax.set_ylabel("attention motif hit rate")
    ax.set_ylim(0, 1.32); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(title="positive windows", frameon=False, loc="upper center",
              ncol=2, fontsize=9, title_fontsize=9,
              bbox_to_anchor=(0.5, 1.02), handlelength=1.4)
    ax.set_title("a   Attention faithfulness depends on benchmark design",
                 loc="left", fontweight="bold", color=NAVY, fontsize=10.5, pad=26)

    ax = axes[1]
    order = ["A", "B", "C", "D"]
    for k, colour in zip(order, [RED, RED, GREEN, GREEN]):
        ax.errorbar(auc[k].mean(), hit[k].mean(),
                    yerr=hit[k].std(ddof=1), xerr=auc[k].std(ddof=1),
                    fmt="o", ms=9, color=colour, capsize=3, mec="white", mew=1.2)
        ax.annotate(k, (auc[k].mean(), hit[k].mean()),
                    textcoords="offset points", xytext=(11, -4),
                    fontweight="bold", fontsize=10, color=NAVY)
    ax.set_xlabel("classification AUC-ROC")
    ax.set_ylabel("attention motif hit rate")
    ax.set_ylim(0.22, 1.10); ax.set_xlim(0.678, 0.822)
    ax.set_title("b   Higher accuracy, less faithful attention",
                 loc="left", fontweight="bold", color=NAVY, fontsize=10.5, pad=26)
    for k, dx, dy in [("A", 12, -3), ("B", 12, -3), ("C", 12, -3), ("D", 12, -3)]:
        pass
    fig.savefig(f"{FIG_DIR}/fig7_design_2x2.png")
    plt.close(fig)
    print("Saved fig7_design_2x2.png")


# ---------------------------------------------------------------- fig 8
def fig_occlusion(occ):
    if len(occ) < 4:
        print("skipping fig8 — occlusion results incomplete")
        return
    order = ["A", "B", "C", "D"]
    labels = ["A\nchrom\ncentered", "B\nregion\ncentered", "C\nchrom\npadded", "D\nregion\npadded"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1]},
                             constrained_layout=True)

    ax = axes[0]
    x = np.arange(4); w = 0.26
    series = [("PBE shuffled", "drop_motif", RED),
              ("PBE → GC-rich", "drop_motif_to_GC", BLUE),
              ("random 8-mer shuffled", "drop_control", GREY)]
    for i, (name, col, colour) in enumerate(series):
        m = [occ[k][col].mean() for k in order]
        e = [occ[k][col].std(ddof=1) for k in order]
        ax.bar(x + (i - 1) * w, m, w, yerr=e, capsize=3, label=name,
               color=colour, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("drop in predicted probability")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_ylim(0, 0.46)
    ax.set_title("a   The model causally depends on the motif in every design",
                 loc="left", fontweight="bold", color=NAVY, fontsize=10.5, pad=10)

    ax = axes[1]
    m = [occ[k].frac_flipped_motif.mean() * 100 for k in order]
    e = [occ[k].frac_flipped_motif.std(ddof=1) * 100 for k in order]
    cols = [RED, RED, GREEN, GREEN]
    bars = ax.bar(x, m, 0.6, yerr=e, capsize=4, color=cols,
                  edgecolor="white", linewidth=0.8)
    for b, v, err in zip(bars, m, e):
        ax.text(b.get_x() + b.get_width() / 2, v + err + 2.0, f"{v:.0f}%",
                ha="center", fontweight="bold", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("predictions flipped below p = 0.5")
    ax.set_ylim(0, max(np.array(m) + np.array(e)) * 1.22)
    ax.set_title("b   Predictions overturned by destroying the motif", loc="left",
                 fontweight="bold", color=NAVY, fontsize=10.5, pad=10)
    fig.savefig(f"{FIG_DIR}/fig8_occlusion.png")
    plt.close(fig)
    print("Saved fig8_occlusion.png")


# ---------------------------------------------------------------- fig 9
def fig_shortcuts():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)

    ax = axes[0]
    for key, colour, style, name in [("B", RED, "-", "peak-centered positives"),
                                     ("D", GREEN, "-", "random-padded positives")]:
        d = pd.read_csv(DATASETS[key])
        pos = d[d.label == 1].sequence.values
        arr = np.frombuffer("".join(pos).encode(), dtype="S1").reshape(len(pos), 101)
        ax.plot((arr == b"T").mean(axis=0), color=colour, lw=1.6, label=name)
    d = pd.read_csv(DATASETS["D"])
    neg = d[d.label == 0].sequence.values
    arr = np.frombuffer("".join(neg).encode(), dtype="S1").reshape(len(neg), 101)
    ax.plot((arr == b"T").mean(axis=0), color=GREY, lw=1.4, ls="--", label="negatives")
    ax.set_xlabel("position in 101-nt window"); ax.set_ylabel("T frequency")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("a   Shortcut 1: positional structure in positives", loc="left",
                 fontweight="bold", color=NAVY, fontsize=10.5, pad=10)

    ax = axes[1]
    cb = pd.read_csv("../outputs/composition_baseline.csv").drop_duplicates("dataset", keep="last")
    want = {"A chrom negs + centered": "A", "B region negs + centered": "B",
            "C chrom negs + padded": "C", "D region negs + padded": "D"}
    keys, vals = [], []
    for name, k in want.items():
        r = cb[cb.dataset == name]
        if len(r):
            keys.append(k); vals.append(float(r.auc_roc.iloc[0]))
    order = np.argsort(keys)
    keys = [keys[i] for i in order]; vals = [vals[i] for i in order]
    # colour by NEGATIVE sampling here -- that is what drives the composition gap
    cols = {"A": NAVY, "B": BLUE, "C": NAVY, "D": BLUE}
    bars = ax.bar(keys, vals, 0.6, color=[cols[k] for k in keys],
                  edgecolor="white", linewidth=0.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=NAVY, label="chromosome negatives"),
                       Patch(facecolor=BLUE, label="region-matched negatives")],
              frameon=False, fontsize=8.5, loc="upper right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontweight="bold", fontsize=9.5)
    ax.axhline(0.5, ls="--", lw=1.2, color=GREY)
    ax.text(-0.45, 0.507, "chance", ha="left", fontsize=8.5, color="#6b7680")
    ax.set_ylim(0.45, 0.80); ax.set_ylabel("AUC-ROC from composition alone")
    ax.set_xlabel("design")
    ax.set_title("b   Shortcut 2: base composition", loc="left",
                 fontweight="bold", color=NAVY, fontsize=10.5, pad=10)
    fig.savefig(f"{FIG_DIR}/fig9_shortcuts.png")
    plt.close(fig)
    print("Saved fig9_shortcuts.png")


if __name__ == "__main__":
    hit, null, auc, occ = load()
    fig_2x2(hit, null, auc)
    fig_occlusion(occ)
    fig_shortcuts()
    print(f"\nAll design figures written to {FIG_DIR}/")
