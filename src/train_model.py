"""
Stage 4: Generic train/eval script -- works with ANY model that takes
(batch, 4, seq_len) and outputs (batch,) raw logits.

This replaces train_baseline.py going forward -- same logic, just
parameterized so we don't rewrite this for every new architecture
(CNN-LSTM now, CNN-LSTM-Attention in Stage 5).

Run:
    python train_model.py --model baseline
    python train_model.py --model cnn_lstm
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score

from data_prep import build_dataset
from dataset import RBPDataset
from baseline_cnn import BaselineCNN
from cnn_lstm import CNNLSTM
from cnn_lstm_attention import CNNLSTMAttention

MODEL_REGISTRY = {
    "baseline": BaselineCNN,
    "cnn_lstm": CNNLSTM,
    "cnn_lstm_attention": CNNLSTMAttention,
}

# ---- Config ----
DATA_CSV = "../data/real_sequences.csv"
MODEL_DIR = "../outputs"   # overridable via --model-dir, so ablations cannot clobber the main run
SEQ_LEN = 101
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
SPLIT_SEED = 42  # keep the train/val/test split IDENTICAL across all seed runs --
                  # we want to measure variance from model init/training only,
                  # not from accidentally comparing different data splits


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(force_cpu=False):
    """Pick the best available device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))

    auc_roc = roc_auc_score(labels, probs)
    auc_pr = average_precision_score(labels, probs)
    return auc_roc, auc_pr


def main(model_name, force_cpu=False, seed=SEED, verbose=True, epochs=None, data_csv=None):
    if epochs is None:
        epochs = EPOCHS
    if data_csv is None:
        data_csv = DATA_CSV
    set_seed(seed)
    device = get_device(force_cpu=force_cpu)
    if verbose:
        print(f"Using device: {device}")
        print(f"Model: {model_name}  Seed: {seed}")
        print(f"Data: {data_csv}")

    X, y = build_dataset(data_csv, seq_len=SEQ_LEN)
    full_dataset = RBPDataset(X, y)

    n = len(full_dataset)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SPLIT_SEED),  # fixed, not the varying seed
    )
    if verbose:
        print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(seq_len=SEQ_LEN).to(device)
    print(f"Num params: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Early stopping: track the best validation AUC-ROC seen so far, and
    # stop if it doesn't improve for `patience` epochs in a row. Restores
    # the best-performing weights (not just whatever the last epoch left us
    # with) before final evaluation.
    patience = 5
    min_delta = 0.001  # improvement must be at least this much to "count" -- otherwise
                        # tiny noise-level fluctuations keep resetting the patience
                        # counter and early stopping never actually triggers
    best_val_auc = -1.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / n_train
        val_auc_roc, val_auc_pr = evaluate(model, val_loader, device)
        if verbose:
            print(f"Epoch {epoch:2d}/{epochs}  train_loss={avg_loss:.4f}  "
                  f"val_AUC-ROC={val_auc_roc:.4f}  val_AUC-PR={val_auc_pr:.4f}")

        if val_auc_roc > best_val_auc + min_delta:
            best_val_auc = val_auc_roc
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement >= {min_delta} for {patience} epochs, best val_AUC-ROC={best_val_auc:.4f})")
                break

    # Restore the best checkpoint before final evaluation -- otherwise we'd
    # report whatever the LAST epoch happened to be, which may be worse
    # than the peak (especially relevant now that we stop early).
    if best_state is not None:
        model.load_state_dict(best_state)

    test_auc_roc, test_auc_pr = evaluate(model, test_loader, device)
    if verbose:
        print(f"\n--- Final Test Set Results ({model_name}) ---")
        print(f"AUC-ROC: {test_auc_roc:.4f}")
        print(f"AUC-PR:  {test_auc_pr:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = f"{MODEL_DIR}/{model_name}_seed{seed}.pt"
    torch.save(model.state_dict(), save_path)
    if verbose:
        print(f"\nSaved model to {save_path}")

    # Log results to a CSV so we build a proper record over time instead
    # of relying on terminal output / screenshots
    results_path = f"{MODEL_DIR}/results_log.csv"
    result_row = {
        "model": model_name,
        "seed": seed,
        "test_auc_roc": round(test_auc_roc, 4),
        "test_auc_pr": round(test_auc_pr, 4),
        "num_params": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
    }
    file_exists = os.path.exists(results_path)
    with open(results_path, "a") as f:
        if not file_exists:
            f.write(",".join(result_row.keys()) + "\n")
        f.write(",".join(str(v) for v in result_row.values()) + "\n")
    if verbose:
        print(f"Logged results to {results_path}")

    return test_auc_roc, test_auc_pr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="baseline")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if MPS/CUDA available")
    parser.add_argument("--epochs", type=int, default=None, help="Override default EPOCHS (useful for quick timing tests)")
    parser.add_argument("--data", type=str, default=None, help="Override default DATA_CSV path")
    parser.add_argument("--model-dir", type=str, default=None, help="Where to write checkpoints and results_log.csv")
    parser.add_argument("--seed", type=int, default=SEED, help="Training seed")
    args = parser.parse_args()
    if args.model_dir:
        MODEL_DIR = args.model_dir
    main(args.model, force_cpu=args.cpu, seed=args.seed, epochs=args.epochs,
         data_csv=args.data)