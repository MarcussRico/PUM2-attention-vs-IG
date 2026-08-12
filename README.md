# Attention vs. Integrated Gradients for PUM2 Binding-Site Prediction

Code and result logs for the preprint *"Attention versus Gradient-Based Interpretability for
RNA-Binding Protein Binding-Site Prediction: A Case Study on PUM2."*

We train CNN, CNN-BiLSTM, and CNN-BiLSTM-Attention models to predict PUM2 binding sites from
ENCODE eCLIP-seq data, then compare attention weights against Integrated Gradients (IG)
attributions from the *identical* trained model — validated against PUM2's experimentally
characterized binding motif (UGUANAUA) rather than against each other.

**Main finding.** Despite statistically indistinguishable classification accuracy, the two
interpretability methods agree on the important sequence position only 17.6% ± 7.0% of the time.
IG recovers the true motif 0.791 ± 0.046 of the time versus 0.359 ± 0.152 for attention.

## Results

Real PUM2 eCLIP data (K562), five seeds, mean ± sample SD:

| Model | Parameters | AUC-ROC | AUC-PR |
|---|---|---|---|
| Baseline CNN | 1,217 | 0.7473 ± 0.0025 | 0.7345 ± 0.0027 |
| CNN-BiLSTM | 18,145 | 0.8206 ± 0.0041 | 0.8174 ± 0.0045 |
| CNN-BiLSTM-Attention | 18,210 | 0.8197 ± 0.0020 | 0.8175 ± 0.0026 |

Motif localization (hit rate over correctly classified positives containing an exact PBE match):

| Method | Hit rate | Random-position null | Median distance to motif |
|---|---|---|---|
| Attention | 0.359 ± 0.152 | 0.127 ± 0.010 | 3.5–20.5 nt |
| Integrated Gradients | **0.791 ± 0.046** | 0.121 ± 0.015 | **1.5–2.5 nt** |

Paired across seeds: IG vs. attention, t = 6.50, df = 4, p = 0.003.

## Repository layout

```
data/
  real_sequences.csv      171,998 PUM2 sequences (86,993 pos / 85,005 neg), 101 nt
  toy_sequences.csv       1,000 synthetic sequences with a planted GCAUG motif
src/
  real_data_prep.py       BED peaks + hg38 FASTA -> labelled sequence CSV
  make_toy_data.py        generate the synthetic sanity-check dataset
  dataset.py              one-hot encoding and torch Dataset
  baseline_cnn.py         CNN
  cnn_lstm.py             CNN + BiLSTM
  cnn_lstm_attention.py   CNN + BiLSTM + attention
  train_model.py          single-model training / evaluation
  run_multi_seed.py       trains all three architectures across seeds 0-4
  integrated_gradients.py IG implementation (zero baseline, 24 Riemann steps)
  validate_pum2_motif.py  attention & IG vs. the PBE consensus motif
  generate_figures.py     produces every figure in the paper
outputs/
  results_log.csv         per-seed AUC-ROC / AUC-PR for all three models
  pum2_motif_validation.csv    per-seed attention hit rates
  pum2_ig_validation_full.csv  per-seed IG hit rates and attention/IG agreement
  figures/                Figures 1-6
  *.pt                    trained model weights, one per architecture per seed
```

## Reproducing

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

The dataset in `data/real_sequences.csv` is committed, so training runs without any external
downloads:

```bash
cd src
python run_multi_seed.py          # trains all 3 models x 5 seeds -> outputs/results_log.csv
python validate_pum2_motif.py     # attention & IG motif localization
python generate_figures.py        # regenerates Figures 1-6
```

To rebuild the dataset from source instead, you need the ENCODE narrowPeak BED file for
experiment [ENCSR661ICQ](https://www.encodeproject.org/experiments/ENCSR661ICQ/) and an hg38
reference FASTA:

```bash
python real_data_prep.py --bed PUM2_K562.bed --genome hg38.fa --out ../data/real_sequences.csv
```

Train on CPU. The PyTorch MPS backend produces unstable results for the LSTM-containing models
(AUC-ROC 0.82 on MPS vs. 0.95 on CPU for the same seed on toy data); `train_model.py --cpu`
forces CPU.

## Known limitation

Negatives are drawn from anywhere on the same chromosome, matched on length and GC content, but
**not** restricted to transcribed or UTR regions. Positives and negatives therefore differ in
dinucleotide composition beyond GC (TT 11.1% vs. 9.1%; poly-A/T runs ≥6 nt in 22.7% vs. 18.1%),
so the classifier may be partly exploiting a UTR-like compositional signal rather than the PUM2
motif specifically. See Section 5 of the paper. Regenerating negatives from annotated transcript
regions is the top item of future work.

## Data source

PUM2 binding sites: ENCODE eCLIP-seq experiment ENCSR661ICQ (K562), Van Nostrand et al. 2016,
*Nature Methods* 13:508–514. Motif ground truth: White et al. 2001, *RNA* 7(12):1855–1866.

## License

MIT — see [LICENSE](LICENSE).
