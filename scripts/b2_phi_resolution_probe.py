#!/usr/bin/env python
"""B2 / workplan T1 -- phi resolution probe. STOP GATE, do not skip.

docs/WORKPLAN_2026-08-25.md T1.3 / docs/RESEARCH_PLAN_2026-08-26.md B2:
before committing to a full-chromosome refetch at 1 kb (projected ~25x the
entries of the 5 kb band -- a real cost, not a small one), fetch ONE ~10 Mb
tile at each candidate resolution and measure, on that tile alone:

  1. usable-bin fraction, against the 5 kb reference 77.7% (chr9, full
     chromosome -- this tile-only number is not directly comparable in scale,
     but a collapse here is still a bad sign)
  2. keep(phi) at 32,768 and 65,536 bp, against the 5 kb reference
     0.0490 / 0.1099 (results/novel_model/p5_window_scan.json)
  3. wall time and bytes for the tile, so a full-chromosome cost is a
     measured extrapolation rather than a guess

DECISION RULE, fixed in advance (do not move it after seeing the numbers):
proceed to a full rebuild only if keep(phi) at 65,536 at least DOUBLES the
5 kb value (0.1099) while the usable fraction stays above ~60%. Otherwise
fall back to 2 kb and repeat, or stop and report -- a negative here is a
publishable measurement of Hi-C's intrinsic spatial autocorrelation, not a
pipeline failure.

Uses the SAME insulation/directionality machinery as the real pipeline
(imported from phase1_features.py, not reimplemented) on a small in-memory
band built directly from cooler -- no npz is written, nothing touches
data/interim or data/processed. Safe to run repeatedly.

Run:  ./3d-gen/bin/python -u scripts/b2_phi_resolution_probe.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from phase1_acquire import fdn_record, CELL_LINES              # noqa: E402
from phase1_features import (                                   # noqa: E402
    insulation, upstream_downstream, directionality_index, _rowcum,
    INSULATION_WINDOWS_BP, DI_WINDOW_BP, SHORT_RANGE_BP, MIN_VALID_FRAC)

OUT = REPO / "results" / "b2_phi_resolution_probe.json"
CHROM = "chr9"
TILE_BP = 10_000_000          # matches phase1_acquire.py's own TILE_BP
BAND_BP = 2_000_000
CANDIDATE_RES = [1_000, 2_000]
REFERENCE_RES = 5_000
REFERENCE_USABLE = 0.7774                              # data_card.md, full chr9
REFERENCE_KEEP = {32_768: 0.0490, 65_536: 0.1099}       # p5_window_scan.json
DOUBLING_TARGET = REFERENCE_KEEP[65_536] * 2            # the pre-fixed gate
USABLE_FLOOR = 0.60
WINDOW_WIDTHS = [32_768, 65_536]
N_WINDOWS_PER_WIDTH = 60          # tile is small; fewer non-overlapping fits
SCAN_SEED = 20260826


def fetch_tile(mcool_url: str, resolution: int) -> dict:
    """One TILE_BP tile of CHROM at `resolution`, balanced, banded -- the
    exact per-tile body of phase1_acquire.fetch_hic_band, run once."""
    import cooler, fsspec, h5py

    t0 = time.time()
    fh = fsspec.open(mcool_url, mode="rb", block_size=4 << 20).open()
    h5 = h5py.File(fh, "r")
    clr = cooler.Cooler(h5[f"resolutions/{resolution}"])
    region = f"{CHROM}:0-{TILE_BP}"
    m = clr.matrix(balance=True, sparse=True).fetch(region).tocoo()
    band_bins = BAND_BP // resolution
    keep = (np.abs(m.row - m.col) <= band_bins) & np.isfinite(m.data) & (m.row <= m.col)
    row, col, val = m.row[keep], m.col[keep], m.data[keep].astype(np.float32)
    bins = clr.bins().fetch(region)
    n = len(bins)
    weight = (bins["weight"].to_numpy(dtype=np.float32)
              if "weight" in bins.columns else np.full(n, np.nan, np.float32))
    dt = time.time() - t0
    nbytes = int(val.nbytes + row.nbytes + col.nbytes + weight.nbytes)
    return {"row": row, "col": col, "val": val, "n": n, "weight": weight,
            "band_bins": band_bins, "elapsed_s": dt, "bytes": nbytes,
            "bin_start": bins["start"].to_numpy(np.int64)}


def band_matrix(tile: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce phase1_features.load_band()'s (n, w+1) banded array from
    COO entries, for a tile that is not a saved npz."""
    n, w = tile["n"], tile["band_bins"]
    weight = tile["weight"]
    valid = np.isfinite(weight)
    band = np.zeros((n, w + 1), dtype=np.float32)
    d = (tile["col"] - tile["row"]).astype(np.int64)
    ok = (d >= 0) & (d <= w) & (tile["row"] < n) & (tile["col"] < n)
    band[tile["row"][ok], d[ok]] = tile["val"][ok]

    ii = np.arange(n)[:, None]
    dd = np.arange(w + 1)[None, :]
    jj = ii + dd
    measurable = np.zeros((n, w + 1), dtype=bool)
    in_range = jj < n
    measurable[in_range] = valid[ii.repeat(w + 1, 1)[in_range]] & valid[jj[in_range]]
    band[~measurable] = np.nan
    return band, valid


def build_phi_tile(tile: dict, resolution: int) -> dict:
    band, valid = band_matrix(tile)
    n = tile["n"]
    cum, cnt = _rowcum(band)
    raw = {}
    for bp in INSULATION_WINDOWS_BP:
        wb = bp // resolution
        if wb < 1 or wb > tile["band_bins"]:
            continue
        raw[f"insulation_{bp//1000}kb"] = insulation(band, n, wb, cum, cnt)

    wdi = min(DI_WINDOW_BP // resolution, tile["band_bins"])
    A, Bm = upstream_downstream(band, n, wdi)
    raw["directionality_2Mb"] = directionality_index(A, Bm)
    tot = A + Bm
    raw["log_contact_density"] = np.where(tot > 0, np.log1p(tot), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["upstream_mass_frac"] = np.where(tot > 0, A / tot, np.nan)
    wsr = min(SHORT_RANGE_BP // resolution, tile["band_bins"])
    short = np.nansum(band[:, 1:wsr + 1], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["short_long_ratio"] = np.where(Bm > 0, short / Bm, np.nan)

    # No compartment_pc1 here: it needs the COARSE matrix (250 kb bins), which
    # a 10 Mb tile cannot support (40 coarse bins at most) -- out of scope
    # for this probe, which is about insulation/directionality resolution
    # sensitivity, the features that actually depend on RES.
    names = list(raw.keys())
    phi = np.stack([raw[k] for k in names], axis=1).astype(np.float32)
    usable = valid & np.isfinite(phi).all(axis=1)
    return {"phi": phi, "usable": usable, "names": names, "n": n}


def interp(phi: np.ndarray, usable: np.ndarray, positions: np.ndarray,
           resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """phase1_dataset.interpolate_phi, parameterised on resolution instead of
    reading it from that module's global (this probe runs multiple
    resolutions in one process and must not mutate shared module state)."""
    n_bins = phi.shape[0]
    u = (positions.astype(np.float64) - resolution / 2.0) / resolution
    i0 = np.floor(u).astype(np.int64)
    frac = (u - i0).astype(np.float32)
    i1 = i0 + 1
    np.clip(i0, 0, n_bins - 1, out=i0)
    np.clip(i1, 0, n_bins - 1, out=i1)
    out = (1.0 - frac)[:, None] * phi[i0] + frac[:, None] * phi[i1]
    valid = usable[i0] & usable[i1]
    out[~valid] = 0.0
    return out.astype(np.float32), valid


def keep_phi(phi: np.ndarray, usable: np.ndarray, resolution: int,
             chrom_len_bp: int, width: int, rng: np.random.Generator) -> dict:
    gvar = phi[usable].var(axis=0)
    starts = rng.integers(0, max(chrom_len_bp - width, 1),
                          size=N_WINDOWS_PER_WIDTH * 4)
    within = []
    for s in starts:
        if len(within) >= N_WINDOWS_PER_WIDTH:
            break
        pos = np.arange(s, s + width, dtype=np.int64)
        pos = pos[pos < chrom_len_bp]
        if len(pos) < width * 0.9:
            continue
        p, v = interp(phi, usable, pos, resolution)
        if v.mean() < MIN_VALID_FRAC:
            continue
        within.append(p[v].astype(np.float64).var(axis=0))
    if not within:
        return {"keep": None, "n_windows": 0}
    wvar = np.mean(np.stack(within), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        keep = float(np.nanmean(wvar / np.where(gvar > 0, gvar, np.nan)))
    return {"keep": keep, "n_windows": len(within)}


def main() -> int:
    print("=" * 74)
    print("B2 STOP GATE -- phi resolution probe, ONE 10 Mb tile, no full fetch")
    print("=" * 74)
    print(f"  reference (5 kb, full chr9): usable {REFERENCE_USABLE:.4f}, "
          f"keep(phi) 32,768={REFERENCE_KEEP[32_768]:.4f} "
          f"65,536={REFERENCE_KEEP[65_536]:.4f}")
    print(f"  decision rule: proceed only if keep(phi)@65,536 >= "
          f"{DOUBLING_TARGET:.4f} (2x reference) AND usable >= {USABLE_FLOOR:.0%}")
    print()

    mcool_acc = next(acc for acc, (short, _) in CELL_LINES["GM12878"]["files"].items()
                     if short == "mcool")
    rec = fdn_record(mcool_acc)
    mcool_url = rec.get("open_data_url")
    print(f"  mcool: {mcool_url}\n")

    rng = np.random.default_rng(SCAN_SEED)
    results = {}
    for res in CANDIDATE_RES:
        print("-" * 74)
        print(f"resolution {res:,} bp")
        print("-" * 74)
        tile = fetch_tile(mcool_url, res)
        print(f"  fetched {TILE_BP/1e6:.0f} Mb tile: {len(tile['val']):,} band "
              f"entries, {tile['n']:,} bins, {tile['elapsed_s']:.1f}s, "
              f"{tile['bytes']/1e6:.1f} MB in-memory")
        pt = build_phi_tile(tile, res)
        usable_frac = float(pt["usable"].mean())
        print(f"  usable-bin fraction on this tile: {usable_frac:.4f} "
              f"(full-chr9 5 kb reference: {REFERENCE_USABLE:.4f})")

        keeps = {}
        for w in WINDOW_WIDTHS:
            if w > TILE_BP * 0.5:
                print(f"  window {w:,} skipped: too large relative to the "
                      f"{TILE_BP:,} bp tile to sample non-degenerate windows")
                continue
            kp = keep_phi(pt["phi"], pt["usable"], res, TILE_BP, w, rng)
            keeps[str(w)] = kp
            if kp["keep"] is not None:
                ref = REFERENCE_KEEP.get(w)
                rel = f"{kp['keep']/ref:.2f}x 5kb-ref" if ref else ""
                print(f"  keep(phi) @ {w:>7,d} bp: {kp['keep']:.4f}  "
                      f"({kp['n_windows']} windows)  {rel}")
            else:
                print(f"  keep(phi) @ {w:>7,d} bp: no valid windows on this tile")

        results[str(res)] = {
            "tile_entries": int(len(tile["val"])), "n_bins": int(tile["n"]),
            "elapsed_s": tile["elapsed_s"], "bytes": tile["bytes"],
            "usable_frac": usable_frac, "keep_phi": keeps,
            "feature_names": pt["names"],
        }
        print()

    print("=" * 74)
    print("GATE -- fixed BEFORE these numbers existed, not adjusted after")
    print("=" * 74)
    verdicts = {}
    for res in CANDIDATE_RES:
        r = results[str(res)]
        k65 = r["keep_phi"].get("65536", {}).get("keep")
        passed = (k65 is not None and k65 >= DOUBLING_TARGET
                 and r["usable_frac"] >= USABLE_FLOOR)
        verdicts[str(res)] = bool(passed)
        k65s = f"{k65:.4f}" if k65 is not None else "n/a"
        print(f"  {res:>6,} bp: keep(phi)@65,536 = {k65s}  "
              f"(need >= {DOUBLING_TARGET:.4f}), usable = "
              f"{r['usable_frac']:.4f} (need >= {USABLE_FLOOR:.0%})  "
              f"=> {'PASS -- proceed to full rebuild' if passed else 'FAIL'}")

    OUT.write_text(json.dumps({
        "purpose": "B2 STOP GATE probe -- ONE 10 Mb tile per resolution, no full fetch",
        "chrom": CHROM, "tile_bp": TILE_BP, "band_bp": BAND_BP,
        "reference_5kb": {"usable_frac": REFERENCE_USABLE,
                          "keep_phi": REFERENCE_KEEP},
        "decision_rule": {"doubling_target_65536": DOUBLING_TARGET,
                          "usable_floor": USABLE_FLOOR},
        "results": results, "verdicts": verdicts,
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
