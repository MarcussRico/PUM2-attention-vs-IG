"""
Stage 3: Baseline CNN model (DeepBind-style)

This is deliberately simple -- it's the floor we need to beat with the
CNN-LSTM-Attention model later. One conv layer to detect motif-like
patterns, pooling to summarize, then a dense layer to classify.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Input:  (batch, 4, seq_len)   -- one-hot encoded RNA, 4 channels (A,U,G,C)
    Output: (batch,)              -- probability of binding (after sigmoid)
    """

    def __init__(self, seq_len=101, num_filters=32, kernel_size=9, dropout=0.25):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same",
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)  # global max pool -> (batch, num_filters, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: (batch, 4, seq_len)
        x = self.conv(x)          # -> (batch, num_filters, seq_len)
        x = self.relu(x)
        x = self.pool(x)          # -> (batch, num_filters, 1)
        x = x.squeeze(-1)         # -> (batch, num_filters)
        x = self.dropout(x)
        x = self.fc(x)            # -> (batch, 1)
        return x.squeeze(-1)      # -> (batch,) raw logits (no sigmoid -- use BCEWithLogitsLoss)


if __name__ == "__main__":
    # Quick shape sanity check
    model = BaselineCNN(seq_len=101)
    dummy_input = torch.randn(16, 4, 101)  # batch of 16
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Num params:", sum(p.numel() for p in model.parameters()))
