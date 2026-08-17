"""
Real data pipeline: BED peaks + reference genome -> sequence,label CSV.

Takes ENCODE eCLIP-seq peak coordinates (BED) and a reference genome FASTA and
produces a CSV in the same format as toy_sequences.csv, so everything downstream
(data_prep.py, train_model.py, run_multi_seed.py) works unchanged.

NEGATIVE SAMPLING MODES
-----------------------
--negative-mode region   (default, recommended)
    Each positive is classified by the transcript region its peak centre falls
    in -- 3'UTR, 5'UTR, CDS, non-coding exon, intron, or intergenic -- and its
    negative is drawn from a peak-free window of the SAME region class,
    preferentially in the same transcript. This is the strongest available
    control: positive and negative share transcription status, region type and
    local base composition, so the only systematic difference left for a
    classifier to exploit is the binding motif itself. Requires --gtf.

--negative-mode transcript
    Negatives from peak-free EXONIC regions of the host transcript. Controls for
    transcription but not for region type, so if positives are largely intronic
    or 3'UTR while negatives land in CDS, a compositional gap survives. Requires
    --gtf.

--negative-mode chromosome
    Original v1 behaviour: anywhere on the same chromosome, GC-matched, >=500 nt
    from the positive. Retained only to reproduce the v1 dataset.

Every mode prints a composition diagnostic table at the end. Read it before
training: near-zero differences in the dinucleotide and poly-A/T rows are the
evidence that no compositional shortcut remains.

USAGE
-----
    python real_data_prep.py \\
        --bed ../data/PUM2_K562.bed \\
        --genome ~/hg38.fa \\
        --gtf ../data/gencode.vXX.annotation.gtf.gz \\
        --out ../data/real_sequences.csv

Use the COMPREHENSIVE GENCODE annotation ("Comprehensive gene annotation", CHR),
not the `basic` subset -- basic keeps roughly one transcript per gene and gives
noticeably worse region coverage.
    https://www.gencodegenes.org/human/
The GTF may be passed gzipped. It is parsed once and cached alongside it as
<gtf>.regions.pkl, so later runs start in seconds.
"""

import argparse
import bisect
import gzip
import os
import pickle
import random
import re
import sys
from collections import defaultdict

import pandas as pd
from pyfaidx import Fasta

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

# region classes, in the priority order used to classify a peak centre
REGIONS = ("utr3", "utr5", "cds", "nc_exon", "intron")
INTERGENIC = "intergenic"

CACHE_VERSION = 2


# --------------------------------------------------------------------------
# sequence helpers
# --------------------------------------------------------------------------

def revcomp(seq):
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(seq))


def extract_window(genome, chrom, start, end, strand="+"):
    """Sequence for an explicit half-open interval, oriented 5'->3' on `strand`."""
    if chrom not in genome:
        return None
    if start < 0 or end > len(genome[chrom]):
        return None
    seq = genome[chrom][start:end].seq.upper()
    return revcomp(seq) if strand == "-" else seq


def extract_sequence(genome, chrom, center, seq_len, strand="+"):
    """Fixed-length genomic window centred on `center`, oriented 5'->3' on `strand`."""
    half = seq_len // 2
    return extract_window(genome, chrom, center - half, center - half + seq_len, strand)


def extract_peak_window(genome, chrom, pk_start, pk_end, seq_len, strand, rng,
                        randomize=True):
    """Window covering a peak, with the peak placed at a random offset.

    Following RBPsuite 2.0: a peak narrower than `seq_len` is padded to length
    with a random split of the padding between the two sides, so the binding
    site does not sit at a fixed position in every training example. Without
    this, peak-relative position is itself a learnable cue and any positional
    structure in the peaks (e.g. the eCLIP offset between crosslink site and
    binding element) leaks into the classifier -- and into anything that claims
    to explain it.

    Peaks wider than `seq_len` are cropped to a random `seq_len` sub-window.
    With randomize=False this reduces to centring on the peak midpoint.
    """
    width = pk_end - pk_start
    if not randomize:
        return extract_sequence(genome, chrom, (pk_start + pk_end) // 2,
                                seq_len, strand)
    if width >= seq_len:
        start = pk_start + rng.randint(0, width - seq_len)
    else:
        pad = seq_len - width
        start = pk_start - rng.randint(0, pad)
    return extract_window(genome, chrom, start, start + seq_len, strand)


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


# --------------------------------------------------------------------------
# interval helpers
# --------------------------------------------------------------------------

def merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def subtract_intervals(span, holes):
    """span (s, e) minus a list of merged intervals -> list of intervals."""
    s, e = span
    out = []
    for hs, he in holes:
        if he <= s:
            continue
        if hs >= e:
            break
        if hs > s:
            out.append((s, hs))
        s = max(s, he)
        if s >= e:
            break
    if s < e:
        out.append((s, e))
    return out


class IntervalIndex:
    """Sorted merged intervals per chromosome with O(log n) overlap queries."""

    def __init__(self):
        self._raw = defaultdict(list)
        self.starts, self.ends = {}, {}

    def add(self, chrom, start, end):
        self._raw[chrom].append((start, end))

    def build(self):
        for chrom, ivs in self._raw.items():
            merged = merge_intervals(ivs)
            self.starts[chrom] = [m[0] for m in merged]
            self.ends[chrom] = [m[1] for m in merged]
        self._raw.clear()
        return self

    def overlaps(self, chrom, start, end):
        starts = self.starts.get(chrom)
        if not starts:
            return False
        i = bisect.bisect_right(starts, end) - 1
        return i >= 0 and self.ends[chrom][i] > start

    def contains(self, chrom, pos):
        return self.overlaps(chrom, pos, pos + 1)


# --------------------------------------------------------------------------
# GTF parsing into per-transcript region sets
# --------------------------------------------------------------------------

TX_RE = re.compile(r'transcript_id "([^"]+)"')
GT_RE = re.compile(r'gene_type "([^"]+)"')


def load_transcript_regions(gtf_path, use_cache=True):
    """Parse a GENCODE GTF into per-transcript region intervals.

    Returns {transcript_id: {"chrom", "strand", "span", "utr3", "utr5", "cds",
                             "nc_exon", "intron"}}
    with all coordinates 0-based half-open.
    """
    cache = f"{gtf_path}.regions.pkl"
    if use_cache and os.path.exists(cache) and \
            os.path.getmtime(cache) > os.path.getmtime(gtf_path):
        with open(cache, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("version") == CACHE_VERSION:
            print(f"Loading cached region annotation: {cache}")
            return blob["transcripts"]
        print("Cached annotation is from an older version — reparsing")

    print(f"Parsing GTF (slow on first run, then cached): {gtf_path}")
    opener = gzip.open if gtf_path.endswith(".gz") else open
    exons, cds, utrs = defaultdict(list), defaultdict(list), defaultdict(list)
    meta = {}

    with opener(gtf_path, "rt") as fh:
        for n, line in enumerate(fh):
            if line[0] == "#":
                continue
            f = line.split("\t", 9)
            if len(f) < 9:
                continue
            kind = f[2]
            if kind not in ("exon", "CDS", "UTR",
                            "five_prime_UTR", "three_prime_UTR"):
                continue
            m = TX_RE.search(f[8])
            if m is None:
                continue
            tid = m.group(1)
            iv = (int(f[3]) - 1, int(f[4]))
            if tid not in meta:
                gt = GT_RE.search(f[8])
                meta[tid] = (f[0], f[6], gt.group(1) if gt else "unknown")
            if kind == "exon":
                exons[tid].append(iv)
            elif kind == "CDS":
                cds[tid].append(iv)
            elif kind == "five_prime_UTR":
                utrs[tid].append(("5", iv))
            elif kind == "three_prime_UTR":
                utrs[tid].append(("3", iv))
            else:                                  # bare "UTR" — resolve below
                utrs[tid].append(("?", iv))
            if n and n % 2_000_000 == 0:
                print(f"  ...{n:,} lines")

    transcripts = {}
    for tid, ex in exons.items():
        chrom, strand, gtype = meta[tid]
        ex = merge_intervals(ex)
        if not ex:
            continue
        span = (ex[0][0], ex[-1][1])
        c = merge_intervals(cds.get(tid, []))
        rec = {"chrom": chrom, "strand": strand, "span": span,
               "utr3": [], "utr5": [], "cds": c, "nc_exon": [],
               "intron": subtract_intervals(span, ex)}

        if c:
            cds_lo, cds_hi = c[0][0], c[-1][1]
            for tag, iv in utrs.get(tid, []):
                if tag == "?":
                    # GENCODE emits bare "UTR"; 3' vs 5' follows from position
                    # relative to the CDS, flipped on the minus strand
                    if iv[1] <= cds_lo:
                        tag = "5" if strand == "+" else "3"
                    elif iv[0] >= cds_hi:
                        tag = "3" if strand == "+" else "5"
                    else:
                        continue
                rec["utr3" if tag == "3" else "utr5"].append(iv)
            rec["utr3"] = merge_intervals(rec["utr3"])
            rec["utr5"] = merge_intervals(rec["utr5"])
        else:
            rec["nc_exon"] = ex

        transcripts[tid] = rec

    print(f"Parsed {len(transcripts):,} transcripts")
    if use_cache:
        try:
            with open(cache, "wb") as fh:
                pickle.dump({"version": CACHE_VERSION, "transcripts": transcripts},
                            fh, protocol=4)
            print(f"Cached to {cache}")
        except OSError as exc:
            print(f"  (could not write cache: {exc})")
    return transcripts


def build_indexes(transcripts):
    """Per-chromosome transcript spans, plus transcript pools per region class."""
    spans = defaultdict(list)
    pools = {r: defaultdict(list) for r in REGIONS}
    tx_cover = IntervalIndex()
    for tid, rec in transcripts.items():
        chrom = rec["chrom"]
        spans[chrom].append((rec["span"][0], rec["span"][1], tid))
        tx_cover.add(chrom, rec["span"][0], rec["span"][1])
        for r in REGIONS:
            if rec[r]:
                pools[r][chrom].append(tid)
    for chrom in spans:
        spans[chrom].sort()
    return spans, pools, tx_cover.build()


def find_host_transcripts(spans, chrom, pos, limit=24):
    hits = []
    lst = spans.get(chrom)
    if not lst:
        return hits
    i = bisect.bisect_right(lst, (pos, float("inf"), "")) - 1
    while i >= 0 and len(hits) < limit:
        s, e, tid = lst[i]
        if e > pos:
            hits.append(tid)
        if pos - s > 3_000_000:          # longest human gene ~2.4 Mb
            break
        i -= 1
    return hits


def classify_peak(transcripts, spans, chrom, pos):
    """Return (region_class, [transcript ids sharing that class])."""
    hosts = find_host_transcripts(spans, chrom, pos)
    if not hosts:
        return INTERGENIC, []
    for region in REGIONS:
        matching = [t for t in hosts
                    if any(s <= pos < e for s, e in transcripts[t][region])]
        if matching:
            return region, matching
    return INTERGENIC, hosts


# --------------------------------------------------------------------------
# negative sampling
# --------------------------------------------------------------------------

def sample_from_intervals(ivs, rng):
    total = sum(e - s for s, e in ivs)
    if total <= 0:
        return None
    off = rng.randrange(total)
    for s, e in ivs:
        if off < e - s:
            return s + off
        off -= (e - s)
    return None


def draw_negative(genome, transcripts, tids, region, peak_mask, seq_len,
                  pos_gc, gc_tol, rng, tries):
    for _ in range(tries):
        rec = transcripts[tids[rng.randrange(len(tids))]]
        ivs = rec[region]
        if not ivs:
            continue
        center = sample_from_intervals(ivs, rng)
        if center is None:
            continue
        half = seq_len // 2
        if peak_mask.overlaps(rec["chrom"], center - half - 500, center + half + 500):
            continue
        seq = extract_sequence(genome, rec["chrom"], center, seq_len, rec["strand"])
        if seq is None or "N" in seq:
            continue
        if abs(gc_content(seq) - pos_gc) <= gc_tol:
            return seq
    return None


def draw_intergenic(genome, chrom, chrom_len, tx_cover, peak_mask, seq_len,
                    pos_gc, gc_tol, rng, tries):
    half = seq_len // 2
    for _ in range(tries):
        center = rng.randint(half, max(half + 1, chrom_len - half))
        if tx_cover.overlaps(chrom, center - half, center + half):
            continue
        if peak_mask.overlaps(chrom, center - half - 500, center + half + 500):
            continue
        seq = extract_sequence(genome, chrom, center, seq_len,
                               "+" if rng.random() < 0.5 else "-")
        if seq is None or "N" in seq:
            continue
        if abs(gc_content(seq) - pos_gc) <= gc_tol:
            return seq
    return None


def draw_anywhere(genome, chrom, chrom_len, peak_mask, seq_len,
                  pos_gc, gc_tol, rng, tries):
    half = seq_len // 2
    for _ in range(tries):
        center = rng.randint(half, max(half + 1, chrom_len - half))
        if peak_mask.overlaps(chrom, center - half - 500, center + half + 500):
            continue
        seq = extract_sequence(genome, chrom, center, seq_len,
                               "+" if rng.random() < 0.5 else "-")
        if seq is None or "N" in seq:
            continue
        if abs(gc_content(seq) - pos_gc) <= gc_tol:
            return seq
    return None


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

POLY_RE = re.compile(r"(A{6,}|T{6,})")


def composition_report(df):
    def stats(seqs):
        n = max(len(seqs), 1)
        total_nt = sum(len(s) for s in seqs) or 1
        gc = sum(s.count("G") + s.count("C") for s in seqs) / total_nt
        di = defaultdict(int)
        tot = 0
        for s in seqs:
            for i in range(len(s) - 1):
                di[s[i:i + 2]] += 1
                tot += 1
        poly = sum(1 for s in seqs if POLY_RE.search(s)) / n
        return gc, {k: v / max(tot, 1) for k, v in di.items()}, poly

    gcp, dip, pp = stats(df.loc[df.label == 1, "sequence"].tolist())
    gcn, din, pn = stats(df.loc[df.label == 0, "sequence"].tolist())

    print("\n" + "=" * 64)
    print("COMPOSITION DIAGNOSTICS   positive / negative / difference")
    print("=" * 64)
    rows = [("GC content", gcp, gcn)]
    rows += [(f"dinucleotide {d}", dip.get(d, 0), din.get(d, 0))
             for d in ("AA", "TT", "AT", "TA", "GC", "CG")]
    rows += [("poly-A/T run >=6 nt", pp, pn)]
    worst = 0.0
    for label, a, b in rows:
        flag = ""
        if label != "GC content":
            worst = max(worst, abs(a - b) * 100)
            if abs(a - b) * 100 > 1.0:
                flag = "  <-- still separable"
        print(f"{label:<24}{a:>9.3%}{b:>11.3%}{a - b:>11.3%}{flag}")
    print("=" * 64)

    # Positional profile: a peak here means window position itself is predictive,
    # independently of sequence content. That is a shortcut no interpretability
    # method can be validated against.
    import numpy as np
    def profile(seqs):
        n = len(seqs[0])
        arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), n)
        return (arr == b"T").mean(axis=0)
    tp = profile(df.loc[df.label == 1, "sequence"].tolist())
    tn = profile(df.loc[df.label == 0, "sequence"].tolist())
    spread_p, spread_n = tp.max() - tp.min(), tn.max() - tn.min()
    print(f"\nPositional T% profile   positives: spread {spread_p:.3f} "
          f"(max at position {int(tp.argmax())})")
    print(f"                        negatives: spread {spread_n:.3f} "
          f"(max at position {int(tn.argmax())})")
    if spread_p > max(0.05, 2 * spread_n):
        print("  <-- WARNING: positives have strong positional structure. Window")
        print("      position is itself predictive. Re-run without --fixed-center.")
    else:
        print("  no strong positional cue in either class.")
    print("=" * 64)

    if worst > 1.0:
        print(f"Largest non-GC gap is {worst:.2f} points. A gap above ~1 point means a")
        print("classifier can still partly separate the classes on composition alone.")
    else:
        print(f"Largest non-GC gap is {worst:.2f} points — no usable compositional")
        print("shortcut remains. Safe to train.")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bed", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gtf")
    p.add_argument("--negative-mode",
                   choices=["region", "transcript", "chromosome"], default="region")
    p.add_argument("--seq_len", type=int, default=101)
    p.add_argument("--n_negatives_per_positive", type=int, default=1)
    p.add_argument("--gc-tolerance", type=float, default=0.05)
    p.add_argument("--gc-tolerance-relaxed", type=float, default=0.10)
    p.add_argument("--max-tries", type=int, default=60)
    p.add_argument("--fixed-center", action="store_true",
                   help="place every peak at the window centre (v1 behaviour). "
                        "Default is RBPsuite-style random padding, which removes "
                        "peak-relative position as a learnable cue.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.negative_mode in ("region", "transcript") and not args.gtf:
        p.error(f"--negative-mode {args.negative_mode} requires --gtf")
    if args.gtf and "basic" in os.path.basename(args.gtf):
        print("WARNING: this looks like the GENCODE 'basic' annotation. The "
              "comprehensive\n         annotation gives much better region "
              "coverage — consider re-downloading.\n")

    rng = random.Random(args.seed)

    print(f"Loading genome: {args.genome}")
    genome = Fasta(args.genome)
    chrom_len = {c: len(genome[c]) for c in genome.keys()}

    print(f"Loading peaks: {args.bed}")
    peaks = pd.read_csv(args.bed, sep="\t", header=None,
                        names=["chrom", "start", "end", "name", "score", "strand"],
                        usecols=range(6))
    print(f"Found {len(peaks):,} peaks")
    widths = (peaks.end - peaks.start)
    print(f"Peak width: median {int(widths.median())} nt, "
          f"IQR {int(widths.quantile(.25))}-{int(widths.quantile(.75))} nt")
    print("Positive windows: " + ("PEAK CENTRED (v1 behaviour)" if args.fixed_center
          else "random padding (peak position varies)"))

    peak_mask = IntervalIndex()
    for c, s, e in zip(peaks.chrom, peaks.start, peaks.end):
        peak_mask.add(c, int(s), int(e))
    peak_mask.build()

    transcripts = spans = pools = tx_cover = None
    if args.negative_mode in ("region", "transcript"):
        transcripts = load_transcript_regions(args.gtf)
        spans, pools, tx_cover = build_indexes(transcripts)

        peak_chroms = set(peaks.chrom.astype(str))
        gtf_chroms = {r["chrom"] for r in transcripts.values()}
        if not (peak_chroms & gtf_chroms & set(genome.keys())):
            sys.exit(
                "\nERROR: no chromosome names shared between BED, GTF and FASTA.\n"
                f"  BED   {sorted(peak_chroms)[:3]}\n"
                f"  GTF   {sorted(gtf_chroms)[:3]}\n"
                f"  FASTA {sorted(genome.keys())[:3]}\n"
                "Usually UCSC 'chr1' vs Ensembl '1'.")

    rows = []
    skipped = 0
    region_counts = defaultdict(int)
    tier = defaultdict(int)

    for i, pk in enumerate(peaks.itertuples(index=False)):
        chrom = pk.chrom
        center = (int(pk.start) + int(pk.end)) // 2
        strand = pk.strand if pk.strand in ("+", "-") else "+"

        pos_seq = extract_peak_window(genome, chrom, int(pk.start), int(pk.end),
                                      args.seq_len, strand, rng,
                                      randomize=not args.fixed_center)
        if pos_seq is None or "N" in pos_seq:
            skipped += 1
            continue
        rows.append({"sequence": pos_seq, "label": 1})
        pos_gc = gc_content(pos_seq)
        clen = chrom_len.get(chrom, 0)

        for _ in range(args.n_negatives_per_positive):
            neg = None

            if args.negative_mode == "region":
                region, host = classify_peak(transcripts, spans, chrom, center)
                region_counts[region] += 1
                if region == INTERGENIC:
                    neg = draw_intergenic(genome, chrom, clen, tx_cover, peak_mask,
                                          args.seq_len, pos_gc, args.gc_tolerance,
                                          rng, args.max_tries)
                    if neg is not None:
                        tier["intergenic match"] += 1
                else:
                    if host:
                        neg = draw_negative(genome, transcripts, host, region,
                                            peak_mask, args.seq_len, pos_gc,
                                            args.gc_tolerance, rng, args.max_tries)
                        if neg is not None:
                            tier["same transcript, same region"] += 1
                    if neg is None:
                        pool = pools[region].get(chrom)
                        if pool:
                            neg = draw_negative(genome, transcripts, pool, region,
                                                peak_mask, args.seq_len, pos_gc,
                                                args.gc_tolerance, rng, args.max_tries)
                            if neg is not None:
                                tier["other transcript, same region"] += 1
                    if neg is None:
                        pool = pools[region].get(chrom)
                        if pool:
                            neg = draw_negative(genome, transcripts, pool, region,
                                                peak_mask, args.seq_len, pos_gc,
                                                args.gc_tolerance_relaxed, rng,
                                                args.max_tries)
                            if neg is not None:
                                tier["same region, relaxed GC"] += 1

            elif args.negative_mode == "transcript":
                _, host = classify_peak(transcripts, spans, chrom, center)
                for region in ("utr3", "cds", "utr5", "nc_exon"):
                    if host:
                        neg = draw_negative(genome, transcripts, host, region,
                                            peak_mask, args.seq_len, pos_gc,
                                            args.gc_tolerance, rng, args.max_tries)
                        if neg is not None:
                            tier["host transcript exon"] += 1
                            break

            if neg is None:
                neg = draw_anywhere(genome, chrom, clen, peak_mask, args.seq_len,
                                    pos_gc, args.gc_tolerance_relaxed, rng,
                                    args.max_tries)
                if neg is not None:
                    tier["fallback: anywhere on chromosome"] += 1

            if neg is None:
                tier["FAILED"] += 1
            else:
                rows.append({"sequence": neg, "label": 0})

        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:,}/{len(peaks):,} peaks")

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"\nSkipped {skipped:,} peaks (chromosome edge or 'N')")
    if region_counts:
        print("\nPositives by transcript region:")
        tot = sum(region_counts.values())
        for r, n in sorted(region_counts.items(), key=lambda x: -x[1]):
            print(f"  {r:<34}{n:>9,}  {n / tot:6.1%}")
    print("\nHow each negative was drawn:")
    tot = sum(tier.values()) or 1
    for k, n in sorted(tier.items(), key=lambda x: -x[1]):
        print(f"  {k:<34}{n:>9,}  {n / tot:6.1%}")
    print(f"\nWrote {len(df):,} rows "
          f"({(df.label == 1).sum():,} positive, {(df.label == 0).sum():,} negative) "
          f"to {args.out}")

    composition_report(df)


if __name__ == "__main__":
    main()
