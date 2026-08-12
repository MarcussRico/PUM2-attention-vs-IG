"""
Stage 5: CNN + BiLSTM + Attention model

This is the novel piece of your paper. Same CNN + BiLSTM front-end as
Stage 4, but instead of global max-pooling (which throws away *where*
in the sequence the important signal was), we add an attention layer
that learns a weight for every position, then takes a weighted sum.

Two things this buys you:
1. A weighted sum can capture information distributed across
   multiple positions, not just the single strongest one (max-pool
   only ever looks at the loudest position).
2. The attention weights themselves are interpretable -- you can
   plot them as a heatmap over the sequence and compare against known
   motif databases (ATtRACT, RBPDB). This is the exact analysis that
   feeds Stage 7 of the roadmap and the interpretability-comparison
   angle from the literature review.

forward() returns both the prediction AND the attention weights,
so downstream analysis code can grab them without re-running the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Learns a scalar importance weight for each position in a sequence,
    then returns a weighted sum (context vector) plus the raw weights
    for visualization.

    Input:  (batch, seq_len, hidden_dim)
    Output: context (batch, hidden_dim), weights (batch, seq_len)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # A small feedforward network that scores each position.
        # tanh keeps scores in a stable range before softmax.
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = self.attn_score(lstm_out)        # -> (batch, seq_len, 1)
        scores = scores.squeeze(-1)                # -> (batch, seq_len)
        weights = F.softmax(scores, dim=1)         # -> (batch, seq_len), sums to 1 per sample

        # Weighted sum over the sequence dimension
        weights_expanded = weights.unsqueeze(-1)   # -> (batch, seq_len, 1)
        context = (lstm_out * weights_expanded).sum(dim=1)  # -> (batch, hidden_dim)

        return context, weights


class CNNLSTMAttention(nn.Module):
    """
    Input:  (batch, 4, seq_len)
    Output: logits (batch,), attention_weights (batch, seq_len)
    """

    def __init__(self, seq_len=101, num_filters=32, kernel_size=9,
                 lstm_hidden=32, dropout=0.25):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding="same",
        )
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = AttentionLayer(hidden_dim=lstm_hidden * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x, return_attention=False):
        # x: (batch, 4, seq_len)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = x.permute(0, 2, 1)              # -> (batch, seq_len, num_filters)
        lstm_out, _ = self.lstm(x)          # -> (batch, seq_len, lstm_hidden*2)

        context, attn_weights = self.attention(lstm_out)  # (batch, hidden*2), (batch, seq_len)

        out = self.dropout2(context)
        out = self.fc(out)                  # -> (batch, 1)
        logits = out.squeeze(-1)            # -> (batch,)

        if return_attention:
            return logits, attn_weights
        return logits


if __name__ == "__main__":
    model = CNNLSTMAttention(seq_len=101)
    dummy_input = torch.randn(16, 4, 101)

    logits = model(dummy_input)
    print("Logits shape:", logits.shape)

    logits, attn = model(dummy_input, return_attention=True)
    print("Logits shape:", logits.shape)
    print("Attention weights shape:", attn.shape)
    print("Attention weights sum to 1 per sample:", attn[0].sum().item())
    print("Num params:", sum(p.numel() for p in model.parameters()))
