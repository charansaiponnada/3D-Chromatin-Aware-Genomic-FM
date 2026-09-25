"""Scratchpad QC: is the stored detrending biased at large separations?

    python qc_detrend.py
    python qc_detrend.py --cell-line GM12878 --json qc_detrend.json

NOT part of the repo. Not committed. This exists to answer one question before
the graphs are built:

    chromgraph.data.detrend() sets

        expected[s] = mean(raw) over the pixels *that were stored* at separation s

    Pixels that cooler never emitted -- a bin pair with no observed contact --
    are absent from that mean, not counted as zero. Sparsity grows with
    separation, so the further out you look the more the surviving pixels are
    the lucky nonzero ones, and the nonzero-only mean sits above the true mean.
    If that inflation grows with s, the stored P(s) is too flat, O/E at large s
    is systematically too small, and top-k distal edge selection is biased.

So compute the same slope both ways and print the gap. Every window here is a
measurement window passed on the command line; this script contains no
pass/fail threshold and reaches no verdict. It prints numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chromgraph.config import data_dir, load_config  # noqa: E402


def loglog_slope(sep_bins: np.ndarray, value: np.ndarray, bin_size: int,
                 lo_bp: float, hi_bp: float) -> dict:
    """Least-squares slope of log10(value) on log10(separation in bp).

    Only separations inside [lo_bp, hi_bp] with a strictly positive value are
    fitted; a zero or absent expectation has no logarithm and is dropped rather
    than floored, so the fit never silently invents a point.
    """
    sep_bp = sep_bins.astype(np.float64) * bin_size
    keep = (sep_bp >= lo_bp) & (sep_bp <= hi_bp) & (value > 0) & np.isfinite(value)
    n = int(keep.sum())
    if n < 2:
        return {"slope": None, "n_points": n,
                "note": "fewer than 2 usable separations in window"}

    x = np.log10(sep_bp[keep])
    y = np.log10(value[keep].astype(np.float64))
    slope, intercept = np.polyfit(x, y, 1)

    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": r2, "n_points": n,
            "sep_bp_range": [float(sep_bp[keep].min()), float(sep_bp[keep].max())]}


def usable_pairs_at_sep(usable: np.ndarray, max_sep: int) -> np.ndarray:
    """Number of bin pairs (i, i+s) with BOTH ends usable, for each s."""
    n = usable.size
    out = np.zeros(max_sep + 1, dtype=np.int64)
    for s in range(max_sep + 1):
        if s >= n:
            break
        out[s] = int(np.count_nonzero(usable[: n - s] & usable[s:]))
    return out


def analyse(d: Path, bin_size: int, lo_bp: float, hi_bp: float,
            tail_lo_bp: float) -> dict:
    pixels = np.load(d / "pixels.npz")
    bins = np.load(d / "bins.npz")
    qc = json.loads((d / "qc.json").read_text(encoding="utf-8"))

    row = pixels["row"].astype(np.int64)
    col = pixels["col"].astype(np.int64)
    raw = pixels["raw"].astype(np.float64)
    oe = pixels["oe"].astype(np.float64)
    stored_expected = pixels["expected"].astype(np.float64)
    usable = bins["usable"].astype(bool)

    sep = col - row
    max_sep = int(stored_expected.size - 1)
    n_bins = int(usable.size)
    seps = np.arange(max_sep + 1)

    # --- how many pixels actually exist at each separation -------------------
    obs_count = np.bincount(sep, minlength=max_sep + 1)[: max_sep + 1]
    obs_total = np.bincount(sep, weights=raw, minlength=max_sep + 1)[: max_sep + 1]

    # detrend() leaves expected[s] = 1.0 where nothing was observed. That is a
    # placeholder, not a measurement, so it must not enter a fit.
    stored_valid = np.where(obs_count > 0, stored_expected, np.nan)

    # --- denominators for the zero-filled expectation ------------------------
    # (a) every pair in the chromosome at this separation
    all_pairs = np.maximum(n_bins - seps, 0).astype(np.int64)
    # (b) only pairs whose two ends both survived ICE / the N-fraction filter.
    #     An ICE-dropped bin is structurally absent, not a zero contact, so this
    #     is the more defensible denominator; both are reported.
    usable_pairs = usable_pairs_at_sep(usable, max_sep)

    # Numerator for (b): restrict to pixels with both ends usable.
    both_usable = usable[row] & usable[col]
    obs_total_u = np.bincount(sep[both_usable], weights=raw[both_usable],
                              minlength=max_sep + 1)[: max_sep + 1]
    obs_count_u = np.bincount(sep[both_usable], minlength=max_sep + 1)[: max_sep + 1]

    with np.errstate(invalid="ignore", divide="ignore"):
        exp_zero_all = np.where(all_pairs > 0, obs_total / all_pairs, np.nan)
        exp_zero_usable = np.where(usable_pairs > 0, obs_total_u / usable_pairs, np.nan)
        fill_frac = np.where(usable_pairs > 0, obs_count_u / usable_pairs, np.nan)

    windows = {
        "primary": (lo_bp, hi_bp),
        "tail": (tail_lo_bp, hi_bp),
        "tail_to_band_edge": (tail_lo_bp, max_sep * bin_size),
    }

    slopes: dict = {}
    for wname, (wlo, whi) in windows.items():
        slopes[wname] = {
            "window_bp": [float(wlo), float(whi)],
            "stored_expected": loglog_slope(seps, stored_valid, bin_size, wlo, whi),
            "zerofilled_usable_pairs": loglog_slope(seps, exp_zero_usable, bin_size, wlo, whi),
            "zerofilled_all_pairs": loglog_slope(seps, exp_zero_all, bin_size, wlo, whi),
        }
        for key in ("zerofilled_usable_pairs", "zerofilled_all_pairs"):
            a = slopes[wname]["stored_expected"].get("slope")
            b = slopes[wname][key].get("slope")
            slopes[wname][key]["minus_stored"] = (
                float(b - a) if (a is not None and b is not None) else None
            )

    # --- rank correlation with separation ------------------------------------
    # O/E is built so the MEAN is 1 at every separation. That does not force the
    # rank correlation to zero: the spread can still drift with s. Raw counts are
    # the same statistic before detrending, as a scale for what "removed" means.
    rho_oe, p_oe = spearmanr(oe, sep)
    rho_raw, p_raw = spearmanr(raw, sep)

    # Sparsity profile -- the mechanism, if the two slopes disagree.
    def at(bp: float) -> float | None:
        s = int(round(bp / bin_size))
        return float(fill_frac[s]) if 0 <= s <= max_sep and np.isfinite(fill_frac[s]) else None

    return {
        "dir": str(d),
        "chrom": qc.get("chrom"),
        "cell_line": qc.get("cell_line"),
        "bin_size": bin_size,
        "n_bins": n_bins,
        "n_pixels": int(row.size),
        "max_separation_bins": max_sep,
        "frac_usable": qc.get("frac_usable"),
        "n_usable_bins": qc.get("n_usable_bins"),
        "checksums": qc.get("checksums"),
        "slopes": slopes,
        "spearman": {
            "oe_vs_separation": {"rho": float(rho_oe), "p": float(p_oe)},
            "raw_vs_separation": {"rho": float(rho_raw), "p": float(p_raw)},
            "n": int(row.size),
        },
        "observed_fraction_of_usable_pairs": {
            "at_10kb": at(10_000), "at_100kb": at(100_000),
            "at_500kb": at(500_000), "at_1Mb": at(1_000_000),
            "at_2Mb": at(2_000_000),
        },
        "mean_oe_by_separation_from_qc": qc.get("mean_oe_by_separation"),
    }


def fmt(v, nd=4, width=None):
    s = "n/a" if v is None else f"{v:.{nd}f}"
    return s if width is None else f"{s:>{width}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-line", default=None,
                    help="default: every cell line found under data/processed")
    ap.add_argument("--chroms", nargs="*", default=None,
                    help="default: every chromosome found")
    ap.add_argument("--lo-bp", type=float, default=10_000,
                    help="low edge of the P(s) fit window (default 10 kb)")
    ap.add_argument("--hi-bp", type=float, default=1_000_000,
                    help="high edge of the P(s) fit window (default 1 Mb)")
    ap.add_argument("--tail-lo-bp", type=float, default=500_000,
                    help="low edge of the distal sub-window (default 500 kb)")
    ap.add_argument("--json", default=None, help="also write the full report here")
    args = ap.parse_args()

    cfg = load_config()
    bin_size = cfg.data.bin_size
    root = data_dir() / "processed"
    if not root.exists():
        print(f"no processed data at {root}; run scripts/build_data.py first",
              file=sys.stderr)
        return 1

    cell_lines = [args.cell_line] if args.cell_line else sorted(
        p.name for p in root.iterdir() if p.is_dir())

    targets = []
    for cl in cell_lines:
        for d in sorted((root / cl).glob(f"*_{bin_size}")):
            chrom = d.name.rsplit("_", 1)[0]
            if args.chroms and chrom not in args.chroms:
                continue
            if (d / "qc.json").exists():
                targets.append(d)

    if not targets:
        print(f"nothing built under {root}", file=sys.stderr)
        return 1

    reports = [analyse(d, bin_size, args.lo_bp, args.hi_bp, args.tail_lo_bp)
               for d in targets]

    kb = lambda bp: f"{bp / 1000:.0f}kb" if bp < 1e6 else f"{bp / 1e6:.0f}Mb"  # noqa: E731
    print()
    print("P(s) log-log slope: stored expected (nonzero pixels only) vs the same")
    print("curve with unobserved pairs counted as zeros.")
    print()
    hdr = (f"{'chrom':<7}{'window':<20}{'stored':>9}{'zf-usable':>11}"
           f"{'delta':>8}{'zf-allpairs':>13}{'delta':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        for wname in ("primary", "tail", "tail_to_band_edge"):
            w = r["slopes"][wname]
            lo, hi = w["window_bp"]
            label = f"{kb(lo)}-{kb(hi)}"
            print(f"{r['chrom']:<7}{label:<20}"
                  f"{fmt(w['stored_expected']['slope'], 3, 9)}"
                  f"{fmt(w['zerofilled_usable_pairs']['slope'], 3, 11)}"
                  f"{fmt(w['zerofilled_usable_pairs']['minus_stored'], 3, 8)}"
                  f"{fmt(w['zerofilled_all_pairs']['slope'], 3, 13)}"
                  f"{fmt(w['zerofilled_all_pairs']['minus_stored'], 3, 8)}")
        print()

    print("Spearman rank correlation with separation")
    hdr2 = f"{'chrom':<7}{'rho(O/E, s)':>14}{'rho(raw, s)':>14}{'n_pixels':>12}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in reports:
        print(f"{r['chrom']:<7}"
              f"{fmt(r['spearman']['oe_vs_separation']['rho'], 5, 14)}"
              f"{fmt(r['spearman']['raw_vs_separation']['rho'], 5, 14)}"
              f"{r['spearman']['n']:>12,}")
    print()

    print("Observed fraction of usable bin pairs (the sparsity that drives the gap)")
    hdr3 = (f"{'chrom':<7}{'10kb':>9}{'100kb':>9}{'500kb':>9}{'1Mb':>9}{'2Mb':>9}")
    print(hdr3)
    print("-" * len(hdr3))
    for r in reports:
        f_ = r["observed_fraction_of_usable_pairs"]
        print(f"{r['chrom']:<7}" + "".join(
            fmt(f_[k], 3, 9) for k in ("at_10kb", "at_100kb", "at_500kb",
                                       "at_1Mb", "at_2Mb")))
    print()

    print("frac_usable and checksums")
    for r in reports:
        print(f"  {r['chrom']}  frac_usable={fmt(r['frac_usable'], 4)}  "
              f"n_bins={r['n_bins']}  n_usable={r['n_usable_bins']}  "
              f"n_pixels={r['n_pixels']:,}")
        for k, v in (r["checksums"] or {}).items():
            print(f"      {k:<12} {v}")
        print(f"      mean O/E by separation (from qc.json): "
              f"{r['mean_oe_by_separation_from_qc']}")
    print()

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
