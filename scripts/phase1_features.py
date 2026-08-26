"""Phase 1, step 2 -- build the per-bin structural feature vector phi.

Consumes the pilot band + coarse matrix from phase1_acquire.py and emits
phi in R^8 per 5 kb bin, standardised chromosome-wide, plus a validity mask.

phi composition (this file OWNS the definition; architecture_spec.md 4.1 gave a
working sketch and explicitly deferred the choice to Phase 1):

  idx  feature                                   RC symmetry
  0    insulation, 100 kb diamond                symmetric
  1    insulation, 250 kb diamond                symmetric
  2    insulation, 500 kb diamond                symmetric
  3    directionality index, 2 Mb                ANTISYMMETRIC
  4    log contact density, 2 Mb                 symmetric
  5    upstream mass fraction                    ANTISYMMETRIC
  6    short/long range ratio (100 kb / 2 Mb)    symmetric
  7    A/B compartment eigenvector               symmetric

The symmetry column is not decoration: architecture_spec.md 4.1.0 needs it for
the reverse-complement sign-tying, and failure mode F7 is precisely the case
where the antisymmetric coordinates cancel between the forward and reverse
Mamba passes.

Run:  python scripts/phase1_features.py
      python scripts/phase1_features.py --chrom chr10

phi standardisation scope, decided 2026-08-26 for the multi-chromosome build
(docs/RESEARCH_PLAN_2026-08-26.md B1): PER-CHROMOSOME, unchanged from the
single-chromosome pilot. Each chromosome's phi is z-scored against its own
usable-bin mean/std (mu, sd below), independently of every other chromosome.
Reasoning: a global standardisation would require every chromosome's raw phi
in memory simultaneously and would make each chromosome's z-scores depend on
which OTHER chromosomes were included in a given build -- re-running with a
different chromosome set would silently change every existing chromosome's
phi values. Per-chromosome standardisation keeps each chromosome's phi build
fully independent and reproducible in isolation, at the cost that keep(phi)
and every phi-derived number are relative to that chromosome's OWN variance,
not on one absolute scale across chromosomes. This must travel with any
cross-chromosome phi comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"

CHROM = "chr9"
RES = 5_000
INSULATION_WINDOWS_BP = (100_000, 250_000, 500_000)
DI_WINDOW_BP = 2_000_000
SHORT_RANGE_BP = 100_000
MIN_VALID_FRAC = 0.5     # a diamond needs this fraction of usable cells

FEATURE_NAMES = [
    "insulation_100kb", "insulation_250kb", "insulation_500kb",
    "directionality_2Mb", "log_contact_density", "upstream_mass_frac",
    "short_long_ratio", "compartment_pc1",
]
# +1 symmetric, -1 antisymmetric under reverse-complement
FEATURE_SYMMETRY = [+1, +1, +1, -1, +1, -1, +1, +1]


def load_band() -> dict:
    z = np.load(INTERIM / f"hic_band_{CHROM}_{RES}bp.npz")
    n = int(z["n_bins"])
    w = int(z["band_bp"]) // RES
    weight = z["weight"]
    valid = np.isfinite(weight)

    # band[i, d] = balanced contact between bin i and bin i+d.
    # 0 means "measurable but no contact observed"; NaN means "not measurable".
    band = np.zeros((n, w + 1), dtype=np.float32)
    d = (z["col"] - z["row"]).astype(np.int64)
    keep = (d >= 0) & (d <= w)
    band[z["row"][keep].astype(np.int64), d[keep]] = z["val"][keep]

    ii = np.arange(n)[:, None]
    dd = np.arange(w + 1)[None, :]
    jj = ii + dd
    measurable = np.zeros((n, w + 1), dtype=bool)
    in_range = jj < n
    measurable[in_range] = valid[ii.repeat(w + 1, 1)[in_range]] & valid[jj[in_range]]
    band[~measurable] = np.nan

    print(f"  band {band.shape}, measurable cells {measurable.mean():.4f}, "
          f"valid bins {valid.sum():,}/{n:,}")
    return {"band": band, "valid": valid, "n": n, "w": w,
            "bin_start": z["bin_start"], "bin_end": z["bin_end"]}


def _rowcum(band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.nan_to_num(band, nan=0.0).astype(np.float64)
    m = np.isfinite(band).astype(np.float64)
    pad = ((0, 0), (1, 0))
    return (np.cumsum(np.pad(z, pad), axis=1), np.cumsum(np.pad(m, pad), axis=1))


def insulation(band: np.ndarray, n: int, w_bins: int,
               cum: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Diamond insulation: total contact in the w x w square upstream x downstream.

    insul(i) = sum_{a=1..w} sum_{b=1..w} M[i-a, i+b]
             = sum_{a=1..w} ( rowcum[i-a, a+w] - rowcum[i-a, a] )
    """
    tot = np.zeros(n)
    obs = np.zeros(n)
    idx = np.arange(n)
    for a in range(1, w_bins + 1):
        src = idx - a
        ok = src >= 0
        s = src[ok]
        tot[ok] += cum[s, a + w_bins] - cum[s, a]
        obs[ok] += cnt[s, a + w_bins] - cnt[s, a]
    frac = obs / (w_bins * w_bins)
    out = np.where(frac >= MIN_VALID_FRAC, tot / np.maximum(obs, 1), np.nan)
    med = np.nanmedian(out)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(out / med)


def upstream_downstream(band: np.ndarray, n: int, w_bins: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """A = contact mass to upstream bins, B = to downstream, within w_bins."""
    B = np.nansum(band[:, 1:w_bins + 1], axis=1)
    A = np.zeros(n)
    idx = np.arange(n)
    for d in range(1, w_bins + 1):
        src = idx - d
        ok = src >= 0
        A[ok] += np.nan_to_num(band[src[ok], d], nan=0.0)
    return A, B


def directionality_index(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Dixon et al. 2012 directionality index."""
    E = (A + B) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        di = np.sign(B - A) * (((A - E) ** 2) / E + ((B - E) ** 2) / E)
    di[~np.isfinite(di)] = np.nan
    return di


def compartments(bin_start: np.ndarray, n: int) -> np.ndarray:
    """PC1 of the O/E correlation matrix at coarse resolution, GC-oriented,
    then broadcast to 5 kb bins."""
    z = np.load(INTERIM / f"hic_coarse_{CHROM}_250000bp.npz")
    M = z["matrix"].astype(np.float64)
    cres = int(z["resolution"])
    k = M.shape[0]

    # observed / expected by diagonal
    oe = np.full_like(M, np.nan)
    for d in range(k):
        i = np.arange(k - d)
        v = M[i, i + d]
        mu = np.nanmean(v)
        if np.isfinite(mu) and mu > 0:
            oe[i, i + d] = v / mu
            oe[i + d, i] = v / mu

    good = np.isfinite(oe).mean(axis=0) > 0.5
    sub = oe[np.ix_(good, good)]
    sub = np.nan_to_num(sub, nan=1.0)
    C = np.corrcoef(sub)
    C = np.nan_to_num(C, nan=0.0)
    vals, vecs = np.linalg.eigh(C)
    pc1_sub = vecs[:, -1] * np.sqrt(max(vals[-1], 0.0))

    # orient so that A compartment (GC-rich) is positive
    gc = gc_content(cres, k)
    if np.corrcoef(pc1_sub, np.nan_to_num(gc[good], nan=np.nanmean(gc)))[0, 1] < 0:
        pc1_sub = -pc1_sub
    print(f"  compartments: {good.sum()}/{k} usable {cres//1000} kb bins, "
          f"eig1={vals[-1]:.2f}, GC-oriented")

    pc1 = np.full(k, np.nan)
    pc1[good] = pc1_sub
    coarse_idx = np.minimum(bin_start // cres, k - 1)
    return pc1[coarse_idx]


def gc_content(cres: int, k: int) -> np.ndarray:
    seq = read_chr9()
    gc = np.full(k, np.nan)
    for i in range(k):
        s = seq[i * cres:(i + 1) * cres]
        if not len(s):
            continue
        acgt = np.isin(s, [b"A", b"C", b"G", b"T"])
        if acgt.sum() < 0.5 * len(s):
            continue
        gc[i] = np.isin(s, [b"G", b"C"]).sum() / acgt.sum()
    return gc


_SEQ_CACHE: np.ndarray | None = None


def read_chr9() -> np.ndarray:
    global _SEQ_CACHE
    if _SEQ_CACHE is None:
        with (INTERIM / f"{CHROM}.fa").open() as f:
            f.readline()
            s = f.read().replace("\n", "").upper()
        _SEQ_CACHE = np.frombuffer(s.encode(), dtype="S1")
    return _SEQ_CACHE


def validate(phi_raw: dict, bin_start: np.ndarray, valid: np.ndarray) -> dict:
    """Cross-check against 4DN's own derived tracks."""
    import gzip
    import pybigtools

    report: dict = {}

    bw = pybigtools.open(str(RAW / "4DNFIBMOGOZC_insulation_bw.bw"))
    ref = np.array(bw.values(CHROM, 0, int(bin_start[-1]) + RES, bins=len(bin_start),
                             fillna=None), dtype=float)
    for name in ("insulation_100kb", "insulation_250kb", "insulation_500kb"):
        ours = phi_raw[name]
        m = np.isfinite(ours) & np.isfinite(ref) & valid
        if m.sum() > 1000:
            r = float(np.corrcoef(ours[m], ref[m])[0, 1])
            report[f"pearson_{name}_vs_4DN"] = round(r, 4)
            report[f"n_compared_{name}"] = int(m.sum())
            print(f"    {name:<20} vs 4DN insulation: r = {r:+.4f}  (n={m.sum():,})")

    bwc = pybigtools.open(str(RAW / "4DNFILYQ1PAY_compartments_bw.bw"))
    refc = np.array(bwc.values(CHROM, 0, int(bin_start[-1]) + RES,
                               bins=len(bin_start), fillna=None), dtype=float)
    ours = phi_raw["compartment_pc1"]
    m = np.isfinite(ours) & np.isfinite(refc)
    r = float(np.corrcoef(ours[m], refc[m])[0, 1])
    report["pearson_compartment_vs_4DN"] = round(r, 4)
    report["n_compared_compartment"] = int(m.sum())
    print(f"    {'compartment_pc1':<20} vs 4DN compartments: r = {r:+.4f}  (n={m.sum():,})")

    # boundary agreement: our insulation minima vs 4DN's calls
    bnds = []
    with gzip.open(RAW / "4DNFIVK5JOFU_boundaries_bed.bed.gz", "rt") as f:
        for line in f:
            p = line.split("\t")
            if p[0] == CHROM:
                bnds.append(int(p[1]) // RES)
    bnds = np.array(sorted(set(bnds)))
    ins = phi_raw["insulation_250kb"]
    fin = np.isfinite(ins)
    local_min = np.zeros(len(ins), dtype=bool)
    idx = np.arange(1, len(ins) - 1)
    ok = fin[idx - 1] & fin[idx] & fin[idx + 1]
    local_min[idx[ok]] = (ins[idx[ok]] < ins[idx[ok] - 1]) & (ins[idx[ok]] < ins[idx[ok] + 1])
    ours_b = np.flatnonzero(local_min & (ins < np.nanpercentile(ins, 25)))
    if len(ours_b) and len(bnds):
        d = np.abs(bnds[:, None] - ours_b[None, :]).min(axis=1)
        within2 = float((d <= 2).mean())
        report["n_4DN_boundaries_chr9"] = int(len(bnds))
        report["n_our_boundary_candidates"] = int(len(ours_b))
        report["frac_4DN_boundaries_within_10kb"] = round(within2, 4)
        report["median_distance_bins"] = float(np.median(d))
        print(f"    boundaries: {len(bnds):,} 4DN calls, {len(ours_b):,} ours; "
              f"{100*within2:.1f}% within 10 kb, median dist "
              f"{np.median(d):.1f} bins")
    return report


def main() -> None:
    global CHROM
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default=CHROM,
                    help="chromosome to build phi for, e.g. chr9, chr10 "
                         "(default: chr9). Must match a chromosome already "
                         "fetched by phase1_acquire.py --chrom.")
    args = ap.parse_args()
    CHROM = args.chrom

    t0 = time.time()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=== loading band ===")
    B = load_band()
    band, valid, n, w = B["band"], B["valid"], B["n"], B["w"]
    bin_start = B["bin_start"]

    print("=== insulation ===")
    cum, cnt = _rowcum(band)
    raw: dict[str, np.ndarray] = {}
    for bp in INSULATION_WINDOWS_BP:
        wb = bp // RES
        raw[f"insulation_{bp//1000}kb"] = insulation(band, n, wb, cum, cnt)
        v = raw[f"insulation_{bp//1000}kb"]
        print(f"  {bp//1000:>3} kb window ({wb} bins): finite {np.isfinite(v).mean():.4f}")

    print("=== directionality / mass ===")
    wdi = DI_WINDOW_BP // RES
    A, Bm = upstream_downstream(band, n, wdi)
    raw["directionality_2Mb"] = directionality_index(A, Bm)
    tot = A + Bm
    raw["log_contact_density"] = np.where(tot > 0, np.log1p(tot), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["upstream_mass_frac"] = np.where(tot > 0, A / tot, np.nan)
    wsr = SHORT_RANGE_BP // RES
    short = np.nansum(band[:, 1:wsr + 1], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["short_long_ratio"] = np.where(Bm > 0, short / Bm, np.nan)
    for k in ("directionality_2Mb", "log_contact_density",
              "upstream_mass_frac", "short_long_ratio"):
        print(f"  {k:<22} finite {np.isfinite(raw[k]).mean():.4f}")

    print("=== compartments ===")
    raw["compartment_pc1"] = compartments(bin_start, n)
    print(f"  compartment_pc1 finite {np.isfinite(raw['compartment_pc1']).mean():.4f}")

    print("=== validation against 4DN derived tracks ===")
    report = validate(raw, bin_start, valid)

    print("=== assembling phi ===")
    phi = np.stack([raw[k] for k in FEATURE_NAMES], axis=1).astype(np.float32)
    usable = valid & np.isfinite(phi).all(axis=1)
    print(f"  bins with complete phi: {usable.sum():,}/{n:,} ({100*usable.mean():.2f}%)")

    mu = phi[usable].mean(axis=0)
    sd = phi[usable].std(axis=0)
    sd[sd == 0] = 1.0
    phi_z = np.full_like(phi, np.nan)
    phi_z[usable] = (phi[usable] - mu) / sd
    print("  standardised chromosome-wide (zero mean, unit variance on usable bins)")
    for i, nm in enumerate(FEATURE_NAMES):
        print(f"    {nm:<22} raw mu={mu[i]:+.4g} sd={sd[i]:.4g} "
              f"sym={'+' if FEATURE_SYMMETRY[i] > 0 else '-'}")

    out = PROCESSED / f"phi_{CHROM}_{RES}bp.npz"
    np.savez_compressed(
        out, phi=phi_z, phi_raw=phi, usable=usable, valid=valid,
        bin_start=bin_start, bin_end=B["bin_end"],
        feature_names=np.array(FEATURE_NAMES),
        feature_symmetry=np.array(FEATURE_SYMMETRY),
        standardisation_mean=mu, standardisation_std=sd,
        resolution=np.int64(RES),
    )
    print(f"\nwrote {out.relative_to(REPO)} ({out.stat().st_size/1e6:.1f} MB)")

    report.update({
        "n_bins": int(n), "usable_bins": int(usable.sum()),
        "usable_frac": round(float(usable.mean()), 4),
        "feature_names": FEATURE_NAMES, "feature_symmetry": FEATURE_SYMMETRY,
        "elapsed_s": round(time.time() - t0, 1),
    })
    # Must END in "_validation_report.json" to match the .gitignore exception
    # `!data/processed/*_validation_report.json` -- these are results, not
    # regenerable data, and must be tracked. chrN goes BEFORE that suffix.
    report_name = ("phi_validation_report.json" if CHROM == "chr9"
                   else f"phi_{CHROM}_validation_report.json")
    rp = PROCESSED / report_name
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {rp.relative_to(REPO)}")
    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
