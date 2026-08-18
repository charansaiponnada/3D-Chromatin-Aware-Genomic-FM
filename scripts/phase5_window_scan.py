#!/usr/bin/env python
"""How much within-window structural signal does a wider window actually buy?

D4 measured keep(phi) = 0.0573 at the current 32,768 bp window: only 5.7% of
phi's variance lies WITHIN a window, 94% is between windows. The delta-bias
mechanism acts per position, so that 5.7% is the entire budget it has to work
with, and it is a sufficient account of the Phase 4 null.

The obvious response is "use a wider window". Before spending 12-76 GPU-hours
finding out, this computes the payoff directly from phi alone -- no model, no
training, seconds of numpy:

    keep(phi, W) = mean_over_windows[ Var_t(phi | window of width W) ]
                   / Var_global(phi)

If keep rises steeply with W, widening the window is the experiment to run and
this quantifies how much signal it hands the mechanism. If keep is flat, a wider
window buys nothing and the Phase 4 null needs a different explanation --
which would be worth knowing for free rather than after three days of compute.

phi is delivered to the model interpolated to token resolution (phase1_dataset
4.1.4), so interpolation is applied here too rather than reading bin values
directly; a stepped read would report a different -- and wrong -- number.

Windows are drawn on the same grid the dataset builder uses and are required to
be >= MIN_STRUCT_FRAC valid, so the counts here are comparable to the training
windows rather than to arbitrary slices of the chromosome.

Reads data/processed only. Writes results/novel_model/p5_window_scan.json
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from phase1_dataset import (                                  # noqa: E402
    PROCESSED, CHROM, RES, MIN_STRUCT_FRAC, interpolate_phi)

OUT = REPO / "results" / "novel_model" / "p5_window_scan.json"
WIDTHS = [8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288]
N_WINDOWS = 300          # sampled per width
SCAN_SEED = 20260817


def main() -> int:
    z = np.load(PROCESSED / "dataset_index.npz", allow_pickle=True)
    phi = z["phi"].astype(np.float64)
    usable = z["usable"].astype(bool)
    names = [str(x) for x in z["feature_names"]]
    n_bins = phi.shape[0]
    chrom_len = n_bins * RES

    # Global variance over usable bins -- the denominator every keep() shares.
    gvar = phi[usable].var(axis=0)

    print("=" * 74)
    print("WINDOW SCAN -- how much of phi's variance is WITHIN a window?")
    print("=" * 74)
    print(f"  {CHROM}, {n_bins:,} bins at {RES:,} bp, "
          f"{usable.sum():,} usable ({usable.mean():.1%})")
    print(f"  phi interpolated to token resolution, as the model receives it")
    print(f"  {N_WINDOWS} windows sampled per width, "
          f"require >= {MIN_STRUCT_FRAC:.0%} valid")
    print()
    print(f"  {'window (bp)':>12s} {'bins':>7s} {'keep(phi)':>10s} "
          f"{'vs 32,768':>10s} {'n used':>7s}")
    print("  " + "-" * 52)

    rng = np.random.default_rng(SCAN_SEED)
    rows = {}
    base = None
    for W in WIDTHS:
        if W >= chrom_len:
            continue
        starts = rng.integers(0, chrom_len - W, size=N_WINDOWS * 4)
        within = []
        for s in starts:
            if len(within) >= N_WINDOWS:
                break
            pos = np.arange(s, s + W, dtype=np.int64)
            # subsample positions for speed; variance is unbiased under it
            if len(pos) > 4096:
                pos = pos[:: len(pos) // 4096]
            p, v = interpolate_phi(phi.astype(np.float32), usable, pos)
            if v.mean() < MIN_STRUCT_FRAC:
                continue
            within.append(p[v].astype(np.float64).var(axis=0))
        if not within:
            continue
        wvar = np.mean(np.stack(within), axis=0)
        keep = float(np.mean(wvar / gvar))
        if base is None and W == 32_768:
            base = keep
        rows[W] = {"keep": keep, "n_windows": len(within),
                   "bins_spanned": W / RES,
                   "per_feature_keep": {names[i]: float(wvar[i] / gvar[i])
                                        for i in range(len(names))}}
        rel = f"{keep / base:.2f}x" if base else "--"
        print(f"  {W:12,d} {W / RES:7.1f} {keep:10.4f} {rel:>10s} "
              f"{len(within):7d}")

    if 32_768 in rows:
        k32 = rows[32_768]["keep"]
        print()
        print(f"  Reference: D4 measured keep(phi) = 0.0573 on the 274 val")
        print(f"  windows at 32,768 bp. This scan gives {k32:.4f} on {N_WINDOWS}")
        print(f"  randomly placed windows -- different sample, same quantity.")

    print()
    print("=" * 74)
    print("READ")
    print("=" * 74)
    print("  keep(phi, W) is the share of structural variance a per-position")
    print("  mechanism can act on at width W. It is an UPPER BOUND on what")
    print("  widening the window can hand the mechanism -- not a prediction")
    print("  that the model will use it. No model is involved in this number.")

    OUT.write_text(json.dumps(
        {"widths": {str(k): v for k, v in rows.items()},
         "config": {"n_windows": N_WINDOWS, "seed": SCAN_SEED,
                    "res": RES, "chrom": CHROM,
                    "min_struct_frac": MIN_STRUCT_FRAC}}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
