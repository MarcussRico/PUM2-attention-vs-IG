#!/bin/bash
# Steps 1-5 of RERUN_CHECKLIST.md, automated.
#
#   bash setup_rerun.sh
#
# Finds the ENCODE BED in Downloads, unpacks and installs it, checks the genome
# and GTF, backs up the v1 dataset and outputs, and clears the results log.
# Nothing is deleted or overwritten without an explicit prompt.

set -u

PROJ="$HOME/Documents/RBP"
GENOME="${GENOME:-$HOME/hg38.fa}"
BED_DEST="$PROJ/data/PUM2_K562.bed"

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

fail() { red "FAILED: $*"; exit 1; }

# ---------------------------------------------------------------- 1. genome
step "1. Reference genome"
[ -f "$GENOME" ] || fail "no genome at $GENOME (override with: GENOME=/path/to/hg38.fa bash setup_rerun.sh)"
grn "found $GENOME ($(du -h "$GENOME" | cut -f1))"

if [ -f "$GENOME.fai" ]; then
  FIRST=$(cut -f1 "$GENOME.fai" | head -1)
  grn "index present, first contig: $FIRST"
else
  FIRST=$(grep -m1 '^>' "$GENOME" | sed 's/^>//' | awk '{print $1}')
  ylw "no .fai index — pyfaidx will build one on first run (~10 min)"
  grn "first contig: $FIRST"
fi

case "$FIRST" in
  chr*) grn "UCSC-style naming — matches GENCODE" ;;
  *)    red "contig names look Ensembl-style ('$FIRST', not 'chr...')."
        red "GENCODE's GTF uses chr1/chr2/... — these will not match."
        fail "stop here and tell Claude before continuing" ;;
esac

# ---------------------------------------------------------------- 2. BED
step "2. ENCODE peak file"
if [ -f "$BED_DEST" ]; then
  grn "already installed: $BED_DEST"
else
  CAND=$(ls -t "$HOME/Downloads/"ENCFF*.bed.gz "$HOME/Downloads/"ENCFF*.bed \
                "$HOME/Downloads/"*narrowPeak* 2>/dev/null | head -1)
  [ -n "$CAND" ] || fail "no ENCFF*.bed(.gz) found in ~/Downloads — download ENCFF372VPV first"
  echo "found: $CAND"
  mkdir -p "$PROJ/data"
  case "$CAND" in
    *.gz) gunzip -c "$CAND" > "$BED_DEST" || fail "could not gunzip $CAND" ;;
    *)    cp "$CAND" "$BED_DEST" ;;
  esac
  grn "installed -> $BED_DEST"
fi

NLINES=$(wc -l < "$BED_DEST" | tr -d ' ')
NCOLS=$(head -1 "$BED_DEST" | awk -F'\t' '{print NF}')
STRANDS=$(cut -f6 "$BED_DEST" | sort -u | tr '\n' ' ')
echo "lines: $NLINES   columns: $NCOLS   strand values: $STRANDS"

[ "$NCOLS" -ge 6 ] || fail "expected >=6 tab-separated columns, got $NCOLS — is this narrowPeak?"
case "$STRANDS" in
  *+*|*-*) : ;;
  *) red "no +/- strand values in column 6 — positives cannot be oriented correctly"
     fail "wrong file format" ;;
esac

if [ "$NLINES" -lt 60000 ] || [ "$NLINES" -gt 120000 ]; then
  ylw "WARNING: $NLINES peaks is outside the expected 60k-120k range."
  ylw "v1 produced 86,993 positives. Send this number to Claude before training."
else
  grn "peak count is in the expected range (v1 -> 86,993 positives)"
fi

# ---------------------------------------------------------------- 3. GTF
step "3. GENCODE annotation"
GTF=$(ls -t "$PROJ/data/"gencode*.gtf.gz "$PROJ/data/"gencode*.gtf 2>/dev/null | head -1)
if [ -z "${GTF:-}" ]; then
  CAND=$(ls -t "$HOME/Downloads/"gencode*.gtf.gz "$HOME/Downloads/"gencode*.gtf 2>/dev/null | head -1)
  if [ -n "$CAND" ]; then
    cp "$CAND" "$PROJ/data/" && GTF="$PROJ/data/$(basename "$CAND")"
    grn "installed -> $GTF"
  else
    red "no gencode*.gtf.gz in ~/Downloads or data/"
    red "download 'Comprehensive gene annotation (CHR), GTF' from:"
    red "  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/latest_release/"
    fail "GTF missing"
  fi
else
  grn "found $GTF"
fi
case "$(basename "$GTF")" in
  *basic*|*patch*|*scaff*) ylw "WARNING: filename suggests this is not the comprehensive CHR annotation" ;;
esac

# ---------------------------------------------------------------- 4. backup
step "4. Back up v1"
if [ -f "$PROJ/data/real_sequences_v1_chromosome_negatives.csv" ]; then
  grn "dataset backup already exists — leaving it alone"
elif [ -f "$PROJ/data/real_sequences.csv" ]; then
  cp "$PROJ/data/real_sequences.csv" "$PROJ/data/real_sequences_v1_chromosome_negatives.csv"
  grn "dataset backed up"
fi

if [ -d "$PROJ/outputs_v1" ]; then
  grn "outputs_v1/ already exists — leaving it alone"
else
  cp -R "$PROJ/outputs" "$PROJ/outputs_v1"
  grn "outputs/ backed up to outputs_v1/"
fi

# ---------------------------------------------------------------- 5. log
step "5. Clear the results log"
if [ -f "$PROJ/outputs/results_log.csv" ]; then
  if [ -f "$PROJ/outputs_v1/results_log.csv" ]; then
    rm -f "$PROJ/outputs/results_log.csv"
    [ -f "$PROJ/outputs/results_log.csv" ] && \
      fail "could not delete results_log.csv — remove it by hand, or v1 and v2 rows will interleave"
    grn "cleared (v1 copy safe in outputs_v1/)"
  else
    fail "refusing to clear results_log.csv — no backup found in outputs_v1/"
  fi
else
  grn "already clear"
fi

# ---------------------------------------------------------------- done
step "Ready"
cat <<EOF
Everything checked out. Next, from Terminal:

  cd $PROJ/src
  python real_data_prep.py \\
    --bed ../data/PUM2_K562.bed \\
    --genome $GENOME \\
    --gtf ../data/$(basename "$GTF") \\
    --out ../data/real_sequences.csv

Takes 20-40 min (the GTF parse is slow once, then cached).

Then STOP and send Claude the composition table it prints at the end,
before you start training.
EOF
