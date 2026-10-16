"""Per-window 3D labels for the frozen-checkpoint probes.

Implements Section 2 of docs/preregistration_window_probes.md exactly:
  T1 insulation  -- diamond insulation, 5 kb bins, 20-bin square, log2 over chrom mean
  T2 compartment -- E1 of the cis O/E correlation at 100 kb, GC-oriented, broadcast to 5 kb
  T3 boundary    -- insulation local minimum (+-5 bins), strength >= 0.1, label +-1 bin

Also stores per-bin GC and N fraction for the reference baseline.

    python scripts/window_labels.py            # all autosomes in the config splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chromgraph.config import load_config
from chromgraph.data import load_chromosome, load_manifest, processed_root

INS_WINDOW = 20        # bins (100 kb at 5 kb)
COMP_RES = 100_000
BOUND_RADIUS = 5
BOUND_MIN_STRENGTH = 0.1
LABEL_RADIUS = 1


def insulation(clr, chrom: str, w: int) -> np.ndarray:
    """Mean balanced contact in the w x w square just off the diagonal at each bin."""
    n = clr.extent(chrom)[1] - clr.offset(chrom)
    out = np.full(n, np.nan)
    m = clr.matrix(balance=True, sparse=True).fetch(chrom).tocsr()
    step = 2000
    for s in range(w, n - w, step):
        e = min(s + step, n - w)
        # block covering every square needed for bins s..e-1
        blk = m[s - w:e + w, s - w:e + w].toarray()
        for i in range(s, e):
            k = i - (s - w)
            sq = blk[k - w:k, k + 1:k + 1 + w]
            if np.isnan(sq).mean() <= 0.5:
                out[i] = np.nanmean(sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log2(out / np.nanmean(out))
    out[~np.isfinite(out)] = np.nan
    return out


def boundaries(ins: np.ndarray, r: int, min_strength: float, label_r: int) -> np.ndarray:
    n = ins.size
    lab = np.zeros(n, dtype=np.float32)
    for i in range(r, n - r):
        v = ins[i]
        if not np.isfinite(v):
            continue
        win = ins[i - r:i + r + 1]
        if np.nanmin(win) < v:
            continue
        left, right = ins[i - r:i], ins[i + 1:i + r + 1]
        if np.all(np.isnan(left)) or np.all(np.isnan(right)):
            continue
        strength = 0.5 * (np.nanmax(left) + np.nanmax(right)) - v
        if strength >= min_strength:
            lab[max(0, i - label_r):i + label_r + 1] = 1.0
    lab[~np.isfinite(ins)] = np.nan
    return lab


def compartment(clr100, chrom: str, gc100: np.ndarray) -> np.ndarray:
    a = clr100.matrix(balance=True).fetch(chrom)
    n = a.shape[0]
    good = np.isfinite(a).sum(1) > n * 0.5
    a = a[np.ix_(good, good)]
    a = np.nan_to_num(a)
    exp = np.array([np.diagonal(a, k).mean() for k in range(a.shape[0])])
    idx = np.abs(np.subtract.outer(np.arange(a.shape[0]), np.arange(a.shape[0])))
    with np.errstate(divide="ignore", invalid="ignore"):
        oe = a / exp[idx]
    oe = np.nan_to_num(oe, nan=0.0, posinf=0.0)
    corr = np.corrcoef(oe)
    corr = np.nan_to_num(corr)
    vals, vecs = np.linalg.eigh(corr)
    e1 = vecs[:, -1]
    full = np.full(n, np.nan)
    full[good] = e1
    ok = np.isfinite(full) & np.isfinite(gc100)
    if np.corrcoef(full[ok], gc100[ok])[0, 1] < 0:
        full = -full
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    import cooler

    cfg = load_config(args.config)
    root = processed_root(cfg)
    mcool = Path("data/raw") / Path(load_manifest()["cell_lines"]["GM12878"]["mcool"]["url"]).name
    clr5 = cooler.Cooler(f"{mcool}::resolutions/{cfg.data.bin_size}")
    clr100 = cooler.Cooler(f"{mcool}::resolutions/{COMP_RES}")
    ratio = COMP_RES // cfg.data.bin_size

    chroms = cfg.data.chroms_train + cfg.data.chroms_val + cfg.data.chroms_test
    summary = {}
    for chrom in chroms:
        cd = load_chromosome(root, "GM12878", chrom, cfg.data.bin_size)
        seq = np.asarray(cd["seq"])
        n = seq.shape[0]
        acgt = (seq < 4).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gc = ((seq == 1) | (seq == 2)).sum(1) / acgt
        nfrac = 1.0 - acgt / seq.shape[1]

        ins = insulation(clr5, chrom, INS_WINDOW)[:n]
        bnd = boundaries(ins, BOUND_RADIUS, BOUND_MIN_STRENGTH, LABEL_RADIUS)
        n100 = clr100.extent(chrom)[1] - clr100.offset(chrom)
        gc100 = np.array([np.nanmean(gc[k * ratio:(k + 1) * ratio]) if k * ratio < n else np.nan
                          for k in range(n100)])
        e1 = np.repeat(compartment(clr100, chrom, gc100), ratio)[:n]
        if e1.size < n:
            e1 = np.concatenate([e1, np.full(n - e1.size, np.nan)])

        out = cd["dir"] / "window_labels.npz"
        np.savez_compressed(out, insulation=ins.astype(np.float32),
                            compartment=e1.astype(np.float32), boundary=bnd,
                            gc=gc.astype(np.float32), n_frac=nfrac.astype(np.float32))
        summary[chrom] = {"n": int(n), "ins_finite": float(np.isfinite(ins).mean()),
                          "e1_finite": float(np.isfinite(e1).mean()),
                          "boundary_pos_rate": float(np.nanmean(bnd)),
                          "frac_A": float(np.nanmean(e1 > 0))}
        print(chrom, summary[chrom], flush=True)
    Path("results/window_probes").mkdir(parents=True, exist_ok=True)
    Path("results/window_probes/labels_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
