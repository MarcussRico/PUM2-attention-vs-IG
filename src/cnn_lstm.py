"""
Stage 4: CNN + BiLSTM model (iDeepS-style, no attention yet)

Same conv front-end as the baseline (detects local motif-like patterns),
but instead of immediately pooling and classifying, we feed the
conv output through a BiLSTM so the model can learn dependencies
between positions -- e.g. "motif X matters more when motif Y is nearby."

Attention gets added on top of this in Stage 5.
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    Input:  (batch, 4, seq_len)   -- one-hot encoded RNA
    Output: (batch,)              -- raw logits (binding probability after sigmoid)
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

        # BiLSTM reads the sequence of conv-filter activations left-to-right
        # AND right-to-left, then concatenates both directions per position.
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout2 = nn.Dropout(dropout)
        # lstm_hidden * 2 because bidirectional concatenates forward + backward
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # x: (batch, 4, seq_len)
        x = self.conv(x)                    # -> (batch, num_filters, seq_len)
        x = self.relu(x)
        x = self.dropout1(x)

        # LSTM expects (batch, seq_len, features) -- conv gives us
        # (batch, features, seq_len), so transpose before feeding in.
        x = x.permute(0, 2, 1)              # -> (batch, seq_len, num_filters)
        lstm_out, _ = self.lstm(x)          # -> (batch, seq_len, lstm_hidden*2)

        # Global max pool over the sequence dimension to get one vector
        lstm_out = lstm_out.permute(0, 2, 1)  # -> (batch, lstm_hidden*2, seq_len)
        pooled = self.pool(lstm_out).squeeze(-1)  # -> (batch, lstm_hidden*2)

        pooled = self.dropout2(pooled)
        out = self.fc(pooled)               # -> (batch, 1)
        return out.squeeze(-1)              # -> (batch,)


if __name__ == "__main__":
    model = CNNLSTM(seq_len=101)
    dummy_input = torch.randn(16, 4, 101)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Num params:", sum(p.numel() for p in model.parameters()))
