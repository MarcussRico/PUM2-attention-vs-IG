"""
Stage 3: Train the baseline CNN on our (toy, for now) dataset.

Run this directly:
    python train_baseline.py

Once real ENCODE data is ready, just change DATA_CSV below to point at
the real file -- everything else stays the same.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score

from data_prep import build_dataset
from dataset import RBPDataset
from baseline_cnn import BaselineCNN

# ---- Config ----
DATA_CSV = "../data/toy_sequences.csv"
SEQ_LEN = 101
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model, loader, device):
    """Run model on a loader, return AUC-ROC and AUC-PR."""
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
    probs = 1 / (1 + np.exp(-logits))  # sigmoid, done manually since model outputs raw logits

    auc_roc = roc_auc_score(labels, probs)
    auc_pr = average_precision_score(labels, probs)
    return auc_roc, auc_pr


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and split data ---
    X, y = build_dataset(DATA_CSV, seq_len=SEQ_LEN)
    full_dataset = RBPDataset(X, y)

    n = len(full_dataset)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # --- Model, loss, optimizer ---
    model = BaselineCNN(seq_len=SEQ_LEN).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Training loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / n_train
        val_auc_roc, val_auc_pr = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={avg_loss:.4f}  "
              f"val_AUC-ROC={val_auc_roc:.4f}  val_AUC-PR={val_auc_pr:.4f}")

    # --- Final test set evaluation ---
    test_auc_roc, test_auc_pr = evaluate(model, test_loader, device)
    print("\n--- Final Test Set Results (Baseline CNN) ---")
    print(f"AUC-ROC: {test_auc_roc:.4f}")
    print(f"AUC-PR:  {test_auc_pr:.4f}")

    # Save the trained model (auto-create outputs dir if it doesn't exist yet)
    import os
    os.makedirs("../outputs", exist_ok=True)
    torch.save(model.state_dict(), "../outputs/baseline_cnn.pt")
    print("\nSaved model to ../outputs/baseline_cnn.pt")


if __name__ == "__main__":
    main()