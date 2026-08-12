"""
PyTorch Dataset wrapper for our one-hot encoded RBP binding data.

Takes the numpy arrays produced by data_prep.py and wraps them so
DataLoader can batch/shuffle them during training.
"""

import torch
from torch.utils.data import Dataset


class RBPDataset(Dataset):
    """
    X: np.ndarray of shape (n_samples, seq_len, 4)  -- one-hot sequences
    y: np.ndarray of shape (n_samples,)              -- binary labels
    """

    def __init__(self, X, y):
        # PyTorch Conv1d expects shape (batch, channels, length) -- so we
        # transpose from (seq_len, 4) to (4, seq_len) per sample.
        # channels = 4 nucleotides, length = seq_len
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
