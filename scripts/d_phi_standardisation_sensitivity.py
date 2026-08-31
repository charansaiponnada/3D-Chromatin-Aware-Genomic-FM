#!/usr/bin/env python
"""Is per-chromosome phi standardisation transductive enough to matter?

docs/PREREG_PHASE_D_2026-08-31.md 2 records this as OWED AND NOT YET RUN:

  "a global-standardisation sensitivity analysis, since per-chromosome
   z-scoring normalises held-out chromosomes with their own statistics
   (transductive)."

The concern is real and a reviewer will find it. phi is z-scored using each
chromosome's OWN mean and sd (phase1_features.py, data_card.md 7). For chr14,
chr15 and chr9 -- val and test -- that means the normalisation constants were
computed from data the model is not supposed to have seen. It is a mild form of
test-set leakage: not labels, but distributional information.

This script quantifies it rather than arguing about it. It re-standardises every
chromosome using TRAIN-ONLY statistics (chr10-13 pooled, matching the multichrom
split) and asks how much changes:

  1. How far apart are the two standardisations, per feature and per chromosome?
     Reported as the z-shift (difference in means, in train-sd units) and the
     scale ratio -- these are exactly the two things per-chromosome z-scoring
     removes and global z-scoring keeps.
  2. Does keep(phi) -- the quantity the whole Phase 4 null was diagnosed on --
     change? If keep(phi) is materially different under global standardisation,
     then every keep(phi) number in CLAUDE.md is scheme-dependent and must be
     labelled as such.

No GPU, no model, no training. Reads the phi npz files directly.

Run: ./3d-gen/bin/python -u scripts/d_phi_standardisation_sensitivity.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "d_phi_standardisation_sensitivity.json"

TRAIN = ["chr10", "chr11", "chr12", "chr13"]
VAL = ["chr14", "chr15"]
TEST = ["chr9"]
ALL = TRAIN + VAL + TEST
WINDOW = 65_536
RES = 5_000
N_WINDOWS = 400
SEED = 20260831


def load(c):
    return np.load(PROC / f"phi_{c}_5000bp.npz", allow_pickle=True)


def keep_phi(phi, usable, window, rng, n_windows):
    """within-window variance fraction, the CLAUDE.md 4 quantity."""
    bins = window // RES
    starts = np.flatnonzero(usable)
    starts = starts[starts + bins < len(usable)]
    if len(starts) == 0:
        return float("nan"), 0
    pick = rng.choice(starts, size=min(n_windows, len(starts)), replace=False)
    within, allv = [], []
    for s in pick:
        seg = phi[s:s + bins]
        m = usable[s:s + bins]
        if m.sum() < 2:
            continue
        seg = seg[m]
        within.append(seg.var(axis=0))
        allv.append(seg)
    if not within:
        return float("nan"), 0
    n_used = len(within)                   # windows, BEFORE the mean collapses
    within_mean = np.mean(within, axis=0)  # (n_features,)
    total = np.concatenate(allv, axis=0).var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(total > 0, within_mean / total, np.nan)
    return float(np.nanmean(k)), n_used


def main() -> int:
    Z = {c: load(c) for c in ALL}
    names = [str(x) for x in Z["chr9"]["feature_names"]]
    rng = np.random.default_rng(SEED)

    # TRAIN-ONLY statistics, pooled over chr10-13 usable bins.
    pool = np.concatenate([Z[c]["phi_raw"][Z[c]["usable"]] for c in TRAIN], 0)
    gmu, gsd = pool.mean(0), pool.std(0)
    gsd[gsd == 0] = 1.0

    print("=" * 78)
    print("phi standardisation sensitivity -- per-chromosome (current) vs")
    print("global TRAIN-ONLY (chr10-13 pooled)")
    print("=" * 78)
    print(f"  train pool: {len(pool):,} usable bins over {', '.join(TRAIN)}")
    print()

    res = {"train_chroms": TRAIN, "val_chroms": VAL, "test_chroms": TEST,
           "window": WINDOW, "n_windows_sampled": N_WINDOWS,
           "feature_names": names,
           "global_mean": gmu.tolist(), "global_sd": gsd.tolist(),
           "per_chrom": {}, "keep_phi": {}}

    print("  How far is each chromosome's own standardisation from the train-global one?")
    print("  z-shift = (chrom_mean - train_mean)/train_sd ; scale = chrom_sd/train_sd")
    print(f"  {'chrom':7s} {'role':6s} {'max|z-shift|':>13s} {'mean|z-shift|':>14s} "
          f"{'scale min':>10s} {'scale max':>10s}")
    for c in ALL:
        raw = Z[c]["phi_raw"][Z[c]["usable"]]
        mu, sd = raw.mean(0), raw.std(0)
        sd = np.where(sd == 0, 1.0, sd)
        zsh = (mu - gmu) / gsd
        sc = sd / gsd
        role = "train" if c in TRAIN else ("val" if c in VAL else "TEST")
        res["per_chrom"][c] = {
            "role": role, "n_usable": int(Z[c]["usable"].sum()),
            "z_shift": zsh.tolist(), "scale_ratio": sc.tolist(),
            "max_abs_z_shift": float(np.abs(zsh).max()),
            "mean_abs_z_shift": float(np.abs(zsh).mean()),
        }
        print(f"  {c:7s} {role:6s} {np.abs(zsh).max():13.4f} {np.abs(zsh).mean():14.4f} "
              f"{sc.min():10.4f} {sc.max():10.4f}")

    print()
    print(f"  keep(phi) at {WINDOW:,} bp under both schemes "
          f"({N_WINDOWS} windows/chrom, mean over 8 features):")
    print(f"  {'chrom':7s} {'role':6s} {'per-chrom':>10s} {'global':>10s} "
          f"{'diff':>9s} {'n_win':>6s}")
    for c in ALL:
        raw, us = Z[c]["phi_raw"], Z[c]["usable"]
        per = (raw - raw[us].mean(0)) / np.where(raw[us].std(0) == 0, 1, raw[us].std(0))
        glo = (raw - gmu) / gsd
        r1 = np.random.default_rng(SEED)
        r2 = np.random.default_rng(SEED)          # identical window sample
        kp, n = keep_phi(per, us, WINDOW, r1, N_WINDOWS)
        kg, _ = keep_phi(glo, us, WINDOW, r2, N_WINDOWS)
        role = res["per_chrom"][c]["role"]
        res["keep_phi"][c] = {"per_chrom": kp, "global": kg,
                              "diff": kg - kp, "n_windows": n}
        print(f"  {c:7s} {role:6s} {kp:10.4f} {kg:10.4f} {kg-kp:+9.4f} {n:6d}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
