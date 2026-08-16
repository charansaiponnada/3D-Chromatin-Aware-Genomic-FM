#!/usr/bin/env python3
"""Build the S4 SEQUENCE-MATCHED control (architecture_spec.md 4.1.3, decision 6).

S4 answers the objection a chromatin reviewer will certainly raise:

    compartment_pc1 -- one of the eight phi coordinates -- correlates strongly
    with GC content and gene density, both computable from sequence alone. So a
    positive structural result may mean nothing more than "you handed the model
    a GC proxy".

S1 and S2 do not address this. S1 destroys everything; S2 destroys alignment,
and the objection specifically requires alignment to be PRESERVED -- the claim
is that an *aligned* 1D signal explains the benefit. S4 is that control: eight
aligned covariates, smooth on the same scales as phi, computed from the DNA
sequence and the GENCODE annotation with NO Hi-C anywhere.

Reading it:
    real phi beats S4  ->  structure carries signal beyond sequence composition
    S4 ~= real phi     ->  the benefit was a 1D covariate benefit; 3D claim dead

MATCHING REQUIREMENTS, and why each one matters
-----------------------------------------------
1. Same shape, same bins, same resolution as phi -- (27679, 8) at 5 kb.
2. Same feature_symmetry [1,1,1,-1,1,-1,1,1]. Two coordinates MUST flip sign
   under reverse-complement, or the reverse pass sees a control with different
   equivariance properties than the real thing and the comparison is confounded
   (failure mode F7).
3. Same standardisation -- zero mean, unit variance over usable bins,
   chromosome-wide, computed the same way phase1_features.py does it.
4. Same `usable` mask. S4 must be defined exactly where phi is defined, so the
   two runs see the same positions and the same number of them.

    ./3d-gen/bin/python scripts/phase4_build_s4.py

Writes data/processed/phi_chr9_5000bp_S4.npz and a validation report. CPU only,
no GPU, no network.
"""

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
PROCESSED = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
CHROM = "chr9"
RES = 5_000
GTF = RAW / "gencode.v47.annotation.gtf.gz"

# Token ids, from phase1_dataset.py. Kept as literals so a silent change there
# shows up as a failing assertion rather than as a quietly different control.
PAD, MASK, A, C, G, T, N_TOK = 0, 1, 2, 3, 4, 5, 6

# Deliberately mirrors phi's structure: three smoothing scales matching the three
# insulation windows, a density term matching log_contact_density, an
# antisymmetric directional term matching directionality_2Mb, an antisymmetric
# upstream-mass term matching upstream_mass_frac, a ratio term matching
# short_long_ratio, and a compartment-like term matching compartment_pc1.
S4_NAMES = [
    "gc_100kb",              # <-> insulation_100kb
    "gc_250kb",              # <-> insulation_250kb
    "gc_500kb",              # <-> insulation_500kb
    "gc_gradient_2Mb",       # <-> directionality_2Mb      ANTISYMMETRIC
    "gene_density_100kb",    # <-> log_contact_density
    "upstream_gene_frac",    # <-> upstream_mass_frac      ANTISYMMETRIC
    "cpg_over_gc",           # <-> short_long_ratio
    "gc_gene_composite",     # <-> compartment_pc1
]
S4_SYMMETRY = [1, 1, 1, -1, 1, -1, 1, 1]


def smooth(x: np.ndarray, span_bp: int) -> np.ndarray:
    """Centred moving average over `span_bp`, in bins. Edge-corrected by
    dividing by the count of real contributing bins, so the chromosome ends are
    not artificially pulled toward zero the way zero-padding would pull them."""
    w = max(1, span_bp // RES)
    if w % 2 == 0:
        w += 1                      # odd, so the window is genuinely centred
    k = np.ones(w)
    num = np.convolve(np.nan_to_num(x, nan=0.0), k, mode="same")
    den = np.convolve(np.isfinite(x).astype(float), k, mode="same")
    out = np.full_like(x, np.nan, dtype=np.float64)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def antisym(x: np.ndarray, span_bp: int) -> np.ndarray:
    """Downstream mean minus upstream mean, over +/- span_bp.

    This is the construction that makes a coordinate ANTISYMMETRIC under
    reverse-complement: reversing the chromosome swaps upstream and downstream,
    which negates the difference. phi's directionality_2Mb has exactly this
    property, which is why it carries symmetry -1.
    """
    w = max(1, span_bp // RES)
    xs = np.nan_to_num(x, nan=0.0)
    fin = np.isfinite(x).astype(float)
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(xs)])
    cf = np.concatenate([[0.0], np.cumsum(fin)])

    def mean_range(lo, hi):                       # half-open, clipped
        lo = np.clip(lo, 0, n); hi = np.clip(hi, 0, n)
        s = cs[hi] - cs[lo]; c = cf[hi] - cf[lo]
        out = np.full(n, np.nan)
        nz = c > 0
        out[nz] = s[nz] / c[nz]
        return out

    i = np.arange(n)
    down = mean_range(i + 1, i + 1 + w)
    up = mean_range(i - w, i)
    return down - up


def bin_sequence_features(tokens: np.ndarray, n_bins: int) -> dict:
    """Per-bin GC fraction, CpG dinucleotide rate and N fraction.

    Computed over resolved nucleotides only: a bin that is half N would
    otherwise report a GC fraction diluted by the unresolved half, which is a
    property of the assembly rather than of the sequence.
    """
    gc = np.full(n_bins, np.nan)
    cpg = np.full(n_bins, np.nan)
    nfrac = np.full(n_bins, np.nan)
    for b in range(n_bins):
        seg = tokens[b * RES:(b + 1) * RES]
        if seg.size == 0:
            continue
        acgt = (seg >= A) & (seg <= T)
        k = int(acgt.sum())
        nfrac[b] = 1.0 - k / seg.size
        if k < RES // 10:            # <10% resolved: no reliable composition
            continue
        gc[b] = float(((seg == C) | (seg == G)).sum()) / k
        # CpG = a C immediately followed by a G, both resolved
        cg = int(((seg[:-1] == C) & (seg[1:] == G)).sum())
        cpg[b] = cg / max(k - 1, 1)
    return {"gc": gc, "cpg": cpg, "n_frac": nfrac}


def gene_density(n_bins: int) -> np.ndarray:
    """Fraction of each bin covered by an annotated gene, from GENCODE.

    Coverage rather than gene count: a bin inside one very long gene and a bin
    containing five short ones are different things, and coverage is the one
    that tracks the active/inactive compartment distinction S4 is meant to
    imitate.
    """
    cov = np.zeros(n_bins, dtype=np.float64)
    kept = 0
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if len(f) < 8 or f[2] != "gene" or f[0] != CHROM:
                continue
            lo, hi = int(f[3]) - 1, int(f[4])       # GTF is 1-based inclusive
            b0, b1 = lo // RES, min((hi - 1) // RES, n_bins - 1)
            if b0 >= n_bins:
                continue
            kept += 1
            for b in range(max(b0, 0), b1 + 1):
                s, e = max(lo, b * RES), min(hi, (b + 1) * RES)
                if e > s:
                    cov[b] += (e - s) / RES
    print(f"  {kept:,} genes on {CHROM}")
    return np.clip(cov, 0.0, None)


def main() -> int:
    t0 = time.time()
    print("=== building S4 SEQUENCE-MATCHED control ===")
    print("no Hi-C is read by this script. That is the point of the control.\n")

    z = np.load(PROCESSED / f"phi_{CHROM}_{RES}bp.npz", allow_pickle=True)
    usable = z["usable"].astype(bool)
    n_bins = int(z["phi"].shape[0])
    phi_names = list(z["feature_names"])
    phi_sym = [int(v) for v in z["feature_symmetry"]]
    print(f"phi: {n_bins:,} bins, {int(usable.sum()):,} usable, "
          f"symmetry {phi_sym}")

    tokens = np.load(PROCESSED / f"tokens_{CHROM}.npy", mmap_mode="r")
    print(f"tokens: {len(tokens):,} bp\n")

    print("--- per-bin sequence composition ---")
    seq = bin_sequence_features(np.asarray(tokens), n_bins)
    for k, v in seq.items():
        print(f"  {k:<8} finite {np.isfinite(v).mean():.4f}")

    print("\n--- gene annotation ---")
    genes = gene_density(n_bins)

    print("\n--- assembling eight aligned covariates ---")
    gc = seq["gc"]
    raw = {
        "gc_100kb":           smooth(gc, 100_000),
        "gc_250kb":           smooth(gc, 250_000),
        "gc_500kb":           smooth(gc, 500_000),
        "gc_gradient_2Mb":    antisym(gc, 2_000_000),
        "gene_density_100kb": smooth(genes, 100_000),
        "upstream_gene_frac": antisym(genes, 100_000),
        "cpg_over_gc":        np.where(np.isfinite(gc) & (gc > 0),
                                       seq["cpg"] / np.maximum(gc, 1e-6), np.nan),
        # deliberate stand-in for compartment_pc1: the classic 1D correlates of
        # A/B compartment membership, combined and smoothed at compartment scale
        "gc_gene_composite":  smooth(gc, 500_000) + smooth(genes, 500_000),
    }
    for k in S4_NAMES:
        print(f"  {k:<20} finite {np.isfinite(raw[k]).mean():.4f}")

    s4 = np.stack([raw[k] for k in S4_NAMES], axis=1).astype(np.float32)

    # Defined exactly where phi is defined, so both runs see identical positions.
    s4_usable = usable & np.isfinite(s4).all(axis=1)
    print(f"\n  usable under phi:        {int(usable.sum()):,}")
    print(f"  usable under phi AND S4: {int(s4_usable.sum()):,}")
    if int(s4_usable.sum()) < 0.95 * int(usable.sum()):
        print("  WARNING: S4 loses >5% of phi's usable bins; the control would "
              "not see the same data as the real run")

    mu = s4[s4_usable].mean(axis=0)
    sd = s4[s4_usable].std(axis=0)
    sd[sd == 0] = 1.0
    s4_z = np.full_like(s4, np.nan)
    s4_z[s4_usable] = (s4[s4_usable] - mu) / sd
    print("\n  standardised chromosome-wide, exactly as phi is")
    for i, nm in enumerate(S4_NAMES):
        print(f"    {nm:<20} raw mu={mu[i]:+.4g} sd={sd[i]:.4g} "
              f"sym={'+' if S4_SYMMETRY[i] > 0 else '-'}")

    assert S4_SYMMETRY == phi_sym, (
        f"S4 symmetry {S4_SYMMETRY} != phi symmetry {phi_sym}; the control would "
        f"have different reverse-complement equivariance than the real features")
    assert s4_z.shape == z["phi"].shape

    # How much of phi does S4 actually explain? This is the number that tells
    # you whether the objection S4 exists to answer had any force to begin with.
    print("\n--- how much of phi is 1D-explainable? (Pearson, usable bins) ---")
    phi_z = z["phi"]
    both = s4_usable & usable
    corrs = {}
    for i, pn in enumerate(phi_names):
        best, bestn = 0.0, None
        for j, sn in enumerate(S4_NAMES):
            a, b = phi_z[both, i], s4_z[both, j]
            if a.std() == 0 or b.std() == 0:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if abs(r) > abs(best):
                best, bestn = r, sn
        corrs[pn] = {"best_r": round(best, 4), "with": bestn}
        print(f"  {pn:<22} best |r| = {abs(best):.4f}  ({bestn})")

    out = PROCESSED / f"phi_{CHROM}_{RES}bp_S4.npz"
    np.savez_compressed(
        out, phi=s4_z, phi_raw=s4, usable=s4_usable, valid=z["valid"],
        bin_start=z["bin_start"], bin_end=z["bin_end"],
        feature_names=np.array(S4_NAMES),
        feature_symmetry=np.array(S4_SYMMETRY),
        standardisation_mean=mu, standardisation_std=sd,
        resolution=np.int64(RES),
    )
    print(f"\nwrote {out.relative_to(REPO)} ({out.stat().st_size/1e6:.1f} MB)")

    rep = {
        "control": "S4 SEQUENCE-MATCHED",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hi_c_used": False,
        "n_bins": n_bins,
        "usable_bins_phi": int(usable.sum()),
        "usable_bins_s4": int(s4_usable.sum()),
        "feature_names": S4_NAMES,
        "feature_symmetry": S4_SYMMETRY,
        "phi_vs_s4_best_correlation": corrs,
        "elapsed_s": round(time.time() - t0, 1),
    }
    rp = PROCESSED / "s4_validation_report.json"
    rp.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {rp.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
