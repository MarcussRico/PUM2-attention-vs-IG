"""
Sanity check for Stage 5: does the attention layer actually point at
the planted motif (GCAUG) in our synthetic positive examples?

This is the toy-data equivalent of the real interpretability validation
we'll do in Stage 7 against ATtRACT/RBPDB -- except here we KNOW the
ground truth (we planted the motif ourselves), so we can directly check
if attention is learning something real or just noise.
"""

import numpy as np
import pandas as pd
import torch

from data_prep import one_hot_encode
from cnn_lstm_attention import CNNLSTMAttention

MOTIF = "GCAUG"
SEQ_LEN = 101
MODEL_PATH = "../outputs/cnn_lstm_attention.pt"
DATA_CSV = "../data/toy_sequences.csv"


def main():
    device = torch.device("cpu")  # inference is cheap, CPU is fine

    model = CNNLSTMAttention(seq_len=SEQ_LEN).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    df = pd.read_csv(DATA_CSV)
    positives = df[df["label"] == 1].reset_index(drop=True)

    hits = 0
    n_check = 20  # check first 20 positive examples

    print(f"Checking if attention weights peak near the planted '{MOTIF}' motif...\n")

    for i in range(n_check):
        seq = positives.loc[i, "sequence"]
        true_motif_pos = seq.find(MOTIF)  # ground truth: where we planted it

        x = one_hot_encode(seq, seq_len=SEQ_LEN)
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(1, 0).unsqueeze(0)  # (1, 4, seq_len)

        with torch.no_grad():
            logits, attn_weights = model(x_tensor, return_attention=True)

        attn_weights = attn_weights.squeeze(0).numpy()  # (seq_len,)
        predicted_peak_pos = int(np.argmax(attn_weights))

        # "Hit" if the attention peak falls within the motif's span (+/- a few nt slack)
        motif_span = range(true_motif_pos - 2, true_motif_pos + len(MOTIF) + 2)
        is_hit = predicted_peak_pos in motif_span
        hits += is_hit

        marker = "✓" if is_hit else "✗"
        print(f"{marker} Sample {i:2d}: motif at {true_motif_pos:3d}-{true_motif_pos+len(MOTIF)}, "
              f"attention peak at {predicted_peak_pos:3d}, "
              f"predicted_prob={torch.sigmoid(logits).item():.3f}")

    print(f"\nAttention correctly localized motif in {hits}/{n_check} samples "
          f"({100*hits/n_check:.0f}%)")


if __name__ == "__main__":
    main()
