"""
Stage 6: Run every model across multiple seeds and report mean +/- std.

This is what turns "we got 0.95 AUC" (which could just be a lucky seed,
as we literally just saw with the MPS vs CPU discrepancy) into a
trustworthy result: "we got 0.94 +/- 0.02 across 5 independent runs."

Run:
    python run_multi_seed.py
"""

import time
import numpy as np
from train_model import main as train_one_run, MODEL_REGISTRY

SEEDS = [0, 1, 2, 3, 4]  # 5 seeds is a reasonable default for a first paper


def run_all():
    results = {}  # model_name -> list of (auc_roc, auc_pr) across seeds
    overall_start = time.time()

    for model_name in MODEL_REGISTRY.keys():
        print(f"\n{'='*50}")
        print(f"Running {model_name} across {len(SEEDS)} seeds...")
        print(f"{'='*50}")

        runs = []
        for seed in SEEDS:
            seed_start = time.time()
            auc_roc, auc_pr = train_one_run(
                model_name, force_cpu=True, seed=seed, verbose=False
            )
            elapsed_min = (time.time() - seed_start) / 60
            total_elapsed_min = (time.time() - overall_start) / 60
            print(f"  seed={seed}: AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
                  f"({elapsed_min:.1f} min this run, {total_elapsed_min:.1f} min total so far)")
            runs.append((auc_roc, auc_pr))

        results[model_name] = runs

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("SUMMARY: mean +/- std across seeds")
    print(f"{'='*60}")
    print(f"{'Model':<22}{'AUC-ROC':<20}{'AUC-PR':<20}")

    for model_name, runs in results.items():
        auc_rocs = np.array([r[0] for r in runs])
        auc_prs = np.array([r[1] for r in runs])
        print(f"{model_name:<22}"
              f"{auc_rocs.mean():.4f} +/- {auc_rocs.std():.4f}    "
              f"{auc_prs.mean():.4f} +/- {auc_prs.std():.4f}")

    return results


if __name__ == "__main__":
    run_all()