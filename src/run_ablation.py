"""
2x2 factorial: how do dataset design choices affect interpretability conclusions?

Two conventional choices in RBP binding-site benchmarks are crossed:

                          | peak-centred positives | random-padded positives
  ------------------------+------------------------+-------------------------
  chromosome negatives    |   cell A               |   cell C
  region-matched negatives|   cell B               |   cell D

Cell D is the main v2 run (../outputs). Cell B is the first ablation
(../outputs_ablation). This script builds and runs whichever cells are missing,
then reports both main effects and their interaction on attention's motif
localisation.

Every cell uses the SAME peak file, so the contrasts are clean -- unlike the
original v1 dataset, which also differed in which ENCODE BED was used.

Only the attention model is trained (5 seeds per cell); that is the model the
interpretability analysis runs on.

Run:
    python run_ablation.py                 # builds and runs missing cells
    python run_ablation.py --report-only   # just re-print the table
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]


def rel(*p):
    return os.path.normpath(os.path.join(HERE, "..", *p))


# name -> (negative mode, fixed_center, dataset path, output dir)
CELLS = {
    "A  chrom negs  + centred": ("chromosome", True,
                                 rel("data", "real_sequences_cellA_chrom_centred.csv"),
                                 rel("outputs_cellA")),
    "B  region negs + centred": ("region", True,
                                 rel("data", "real_sequences_ablation_fixedcenter.csv"),
                                 rel("outputs_ablation")),
    "C  chrom negs  + padded": ("chromosome", False,
                                rel("data", "real_sequences_cellC_chrom_padded.csv"),
                                rel("outputs_cellC")),
    "D  region negs + padded": ("region", False,
                                rel("data", "real_sequences.csv"),
                                rel("outputs")),
}


def run(cmd, label):
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print("$ " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        sys.exit(f"\nFAILED: {label}")
    print(f"[{(time.time() - t0) / 60:.1f} min]")


def ensure_cell(name, spec, args):
    neg_mode, fixed, data, outdir = spec
    val_csv = os.path.join(outdir, "pum2_motif_validation.csv")
    if os.path.exists(val_csv):
        print(f"  {name:<28} already complete -> {os.path.basename(outdir)}")
        return
    print(f"  {name:<28} MISSING, will build and run")
    os.makedirs(outdir, exist_ok=True)
    py = sys.executable

    if not os.path.exists(data):
        cmd = [py, "real_data_prep.py", "--bed", args.bed, "--genome", args.genome,
               "--out", data, "--negative-mode", neg_mode]
        if neg_mode != "chromosome":
            cmd += ["--gtf", args.gtf]
        if fixed:
            cmd += ["--fixed-center"]
        run(cmd, f"{name}: build dataset")

    log = os.path.join(outdir, "results_log.csv")
    if os.path.exists(log):
        os.remove(log)
    for seed in SEEDS:
        run([py, "train_model.py", "--model", "cnn_lstm_attention", "--cpu",
             "--seed", str(seed), "--data", data, "--model-dir", outdir],
            f"{name}: train seed {seed}")

    run([py, "validate_pum2_motif.py", "--data", data,
         "--model-dir", outdir, "--out", val_csv],
        f"{name}: motif validation")


def report():
    import numpy as np
    import pandas as pd
    from scipy import stats

    hits, aucs = {}, {}
    for name, (_, _, _, outdir) in CELLS.items():
        f = os.path.join(outdir, "pum2_motif_validation.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f).sort_values("seed")
        hits[name] = d.all_hit_rate_given_motif.values
        r = os.path.join(outdir, "results_log.csv")
        if os.path.exists(r):
            rr = pd.read_csv(r)
            rr = rr[rr.model == "cnn_lstm_attention"].drop_duplicates("seed")
            aucs[name] = rr.test_auc_roc.values

    print(f"\n{'=' * 82}\nATTENTION MOTIF LOCALISATION ACROSS DATASET DESIGNS\n{'=' * 82}")
    print(f"{'design':<30}{'hit rate':>18}{'seed SD':>10}{'AUC-ROC':>12}{'n seeds':>9}")
    for name in CELLS:
        if name not in hits:
            print(f"{name:<30}{'not run':>18}")
            continue
        h = hits[name]
        a = aucs.get(name)
        auc = f"{a.mean():.4f}" if a is not None and len(a) else "-"
        print(f"{name:<30}{h.mean():>11.3f} ± {h.std(ddof=1):.3f}"
              f"{h.std(ddof=1):>10.3f}{auc:>12}{len(h):>9}")
    print("-" * 82)
    print(f"{'v1 (superseded: different BED)':<30}{0.359:>11.3f} ± {0.152:.3f}"
          f"{0.152:>10.3f}{'0.8206':>12}{5:>9}")
    print("=" * 82)

    keys = list(CELLS)
    if all(k in hits for k in keys):
        A, B, C, D = (hits[k] for k in keys)
        neg_effect = ((B.mean() + D.mean()) - (A.mean() + C.mean())) / 2
        pad_effect = ((C.mean() + D.mean()) - (A.mean() + B.mean())) / 2
        interaction = (D.mean() - C.mean()) - (B.mean() - A.mean())
        print("\nMAIN EFFECTS on attention motif hit rate")
        print(f"  region-matched negatives   {neg_effect:+.3f}")
        print(f"  random-padded positives    {pad_effect:+.3f}")
        print(f"  interaction                {interaction:+.3f}")
        print("\nSimple effects (paired across seeds, df=4)")
        for lab, x, y in [("negatives, holding centred  (B vs A)", B, A),
                          ("negatives, holding padded   (D vs C)", D, C),
                          ("padding, holding chrom negs (C vs A)", C, A),
                          ("padding, holding region negs(D vs B)", D, B)]:
            t, p = stats.ttest_rel(x, y)
            print(f"  {lab:<38}{(x - y).mean():+.3f}   t={t:6.2f}  p={p:.4f}")
    else:
        missing = [k for k in keys if k not in hits]
        print(f"\n(main effects need all four cells; missing: {', '.join(missing)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", default="../data/PUM2_K562.bed")
    ap.add_argument("--genome", default=os.path.expanduser("~/hg38.fa"))
    ap.add_argument("--gtf", default="../data/gencode.v50.annotation.gtf.gz")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        print("Checking which cells of the 2x2 already exist:\n")
        for name, spec in CELLS.items():
            ensure_cell(name, spec, args)
    report()


if __name__ == "__main__":
    main()
