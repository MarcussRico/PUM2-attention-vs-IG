# Benchmark Design Determines Whether Attention Is Faithful

Code, data and results for a controlled study of how benchmark construction affects
interpretability conclusions in RNA-binding protein binding-site models.

**Short version.** We train the same CNN-BiLSTM-Attention model on four datasets built
from the *same* ENCODE PUM2 peaks, varying only two conventional design choices. How
well attention localizes PUM2's experimentally characterized binding motif (UGUANAUA)
ranges from **0.398 to 0.955** across those four datasets. A causal occlusion test shows
the model depends on the motif in all four, so the low figure is genuine unfaithfulness,
not a mis-specified target.

## The 2 × 2

|  | peak-centred positives | random-padded positives |
|---|---|---|
| **chromosome negatives** | **A** — 0.398 ± 0.203 | **C** — 0.921 ± 0.044 |
| **region-matched negatives** | **B** — 0.799 ± 0.080 | **D** — 0.955 ± 0.015 |

Attention motif hit rate, five seeds, mean ± SD. Random-position null ≈ 0.12.

- Random padding: main effect **+0.339**
- Region-matched negatives: main effect **+0.218**
- Interaction **−0.366** (sub-additive; either fix alone recovers most of the effect)

Classification accuracy runs the *other* way — 0.803 (A) → 0.695 (D) — because the
accuracy in A is partly borrowed from shortcuts.

## Supporting results

**Composition-only baseline.** Logistic regression on mono- and dinucleotide frequencies
alone: AUC-ROC 0.704 (A), 0.696 (C), 0.619 (B), 0.616 (D). This is the floor a sequence
model must clear before any of its performance reflects sequence order.

**Causal motif occlusion.** Shuffling the eight motif nucleotides in place drops predicted
probability by 0.279–0.344 and flips 35–52% of predictions below 0.5. Shuffling a random
8-mer elsewhere in the same sequence moves it by 0.0007–0.0023 — a ~200× specificity
ratio, p < 10⁻⁵ in every design.

**Attention vs Integrated Gradients (design D).** Statistically indistinguishable:
0.955 ± 0.015 vs 0.951 ± 0.013, paired t = 0.88, p = 0.43.

## Layout

```
data/
  PUM2_K562.bed                              ENCODE ENCSR661ICQ rep 1, 67,093 peaks, hg38
  real_sequences.csv                         design D  (region-matched negatives, random padding)
  real_sequences_ablation_fixedcenter.csv    design B
  real_sequences_cellA_chrom_centred.csv     design A  (the conventional design)
  real_sequences_cellC_chrom_padded.csv      design C
  real_sequences_v1_chromosome_negatives.csv superseded first-pass dataset
src/
  real_data_prep.py          builds any design: --negative-mode {region,transcript,chromosome},
                             --fixed-center; prints composition and positional diagnostics
  run_ablation.py            runs the full 2 × 2, skipping completed cells
  motif_occlusion.py         causal test: shuffle the motif, measure the drop
  composition_baseline.py    composition-only floor
  generate_design_figures.py Figs. 1-3 of the paper
  generate_figures.py        Figs. 4-6 (ROC, attention vs IG, per-nucleotide tracks)
  train_model.py             --model, --seed, --data, --model-dir
  validate_pum2_motif.py     attention motif localization
  integrated_gradients.py    IG attribution and attention/IG agreement
outputs/          design D  (+ figures/)
outputs_cellA/    design A
outputs_ablation/ design B
outputs_cellC/    design C
outputs_v1/       superseded first-pass results
```

Each `outputs*` directory holds `results_log.csv`, `pum2_motif_validation.csv`,
`motif_occlusion.csv` and the five per-seed model checkpoints.

## Reproducing

Python 3.10+. Two inputs are not in this repository because of their size:

- **hg38 reference FASTA** (~3 GB), from UCSC — must use `chr1`-style contig names
- **GENCODE annotation** (~129 MB), comprehensive/CHR/GTF from
  <https://www.gencodegenes.org/human/> — place in `data/`

```bash
pip install -r requirements.txt
cd src

# rebuild a dataset (the committed CSVs were produced this way)
python real_data_prep.py --bed ../data/PUM2_K562.bed --genome ~/hg38.fa \
  --gtf ../data/gencode.v50.annotation.gtf.gz \
  --out ../data/real_sequences.csv --negative-mode region

# full 2 x 2 (skips cells that already have results)
python run_ablation.py

# supporting analyses
python composition_baseline.py
python motif_occlusion.py --data ../data/real_sequences.csv --model-dir ../outputs
python generate_design_figures.py
```

Train on CPU. The PyTorch MPS backend gives unstable results for the LSTM-containing
models; `train_model.py --cpu` forces CPU.

## Limitations

One RBP (PUM2), one cell line (K562). Motif localization is scorable only on the 7.2–7.4%
of correctly classified positives containing an exact consensus match. Peaks are
replicate-level rather than IDR-thresholded, so 44.9% fall in introns. Five seeds per cell
gives four degrees of freedom. Only the attention architecture was retrained across all
four cells. See the paper's Limitations section.

## Earlier version

The first pass at this project used chromosome-wide negatives and peak-centred windows,
and concluded that attention was unfaithful and gradients should be preferred. Those
results and their README are kept at [docs/README_v1.md](docs/README_v1.md), with data in
`data/real_sequences_v1_chromosome_negatives.csv` and results in `outputs_v1/`. They
correspond approximately to design A above, and are superseded by it.

## Data source

PUM2 binding sites: ENCODE eCLIP-seq ENCSR661ICQ (K562), Van Nostrand et al. 2016,
*Nature Methods* 13:508–514. Motif ground truth: White et al. 2001, *RNA* 7(12):1855–1866.

## License

MIT — see [LICENSE](LICENSE).
