#!/usr/bin/env python
"""P5-POS -- the positional probe, and the within-window test the pooled probe
could not do.

WHY THE POOLED PROBE WAS NOT ENOUGH
------------------------------------
phase5_structure_probe.py mean-pooled the final hidden state over all 32,768
positions and predicted a window-level target. Insulation is a LOCAL property --
a dip at a boundary -- so averaging across the whole window is close to the
worst available way to detect it. A model could localise boundaries perfectly
and still score zero. That probe put structural r = +0.1208 against a GC floor
of +0.1905, and the pooling is a live suspect for the gap.

TWO PROBES HERE, AND THE SECOND IS THE POINT
---------------------------------------------
A. GLOBAL positional
       predict  phi_t          from  h_t
   Removes the input-side pooling. But note what D4 measured: keep(phi) =
   0.0573, i.e. only 5.7% of phi's variance is within-window. So this target is
   ~94% determined by WHICH window the position sits in, and a good score here
   is mostly window identification again -- a fairer version of the pooled
   probe, not a different question.

B. WITHIN-WINDOW (centred)  <-- the sharp test
       predict  phi_t - mean_w(phi)   from   h_t - mean_w(h)
   Both sides have their window mean removed, so window identity is gone by
   construction. What remains is exactly the 5.7% the delta-bias mechanism was
   designed to act on: does the representation track how structure VARIES
   ACROSS a window, not merely which window it is?

   B is the honest test of local structural sensitivity. A is the headline
   number people will quote. Report both; they answer different questions.

phi IS WITHHELD FROM THE MODEL (S0 = zeros), as in the pooled probe. phi is the
target, sequence is the input. Feeding the structural arm real phi and then
probing for phi would be circular.

FLOOR. Local composition: GC + dinucleotide frequencies computed in a +/-2,560
bp neighbourhood of each sampled position, put through the identical ridge and
the identical centring. For probe B the floor is centred too, so it measures
what LOCAL composition alone says about LOCAL structural variation.

Positions are sampled every --stride bp, so rows within a window are not
adjacent and are far enough apart to be less redundant than neighbours.

Inference only. No training. Writes
results/novel_model/p5_positional_probe.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from chromfm.model import use_scan                           # noqa: E402
from phase1_dataset import WindowDataset                     # noqa: E402
from train import collate                                    # noqa: E402
from phase5_structure_probe import (                         # noqa: E402
    load_any, ridge_fit_eval, TARGETS, EXPECTED_NAMES, PROBE_SEED)

NOVEL = REPO / "results" / "novel_model"
BASE = REPO / "results" / "baselines"
OUT = NOVEL / "p5_positional_probe.json"
CACHE = NOVEL / "p5_pos_cache"
BATCH = 2
GC_HALF = 2560           # +/- bp neighbourhood for the local composition floor


@torch.no_grad()
def represent_positional(model, cfg, ds, idx, device, stride, tag=""):
    """Per-position hidden states at every `stride`-th position.

    Returns (X, W) where X is (n_rows, d_model) and W is the window id of each
    row -- W is what the centring in probe B groups by.
    """
    grab = {}

    def hook(_m, _i, out):
        grab["h"] = out.detach()

    h = model.norm_f.register_forward_hook(hook)
    feats, wid = [], []
    t0 = time.time()
    try:
        for b0 in range(0, len(idx), BATCH):
            if b0 and b0 % 100 == 0:
                el = time.time() - t0
                print(f"    {tag} {b0}/{len(idx)} windows, {el:.0f}s, "
                      f"eta {(len(idx) - b0) * el / b0:.0f}s", flush=True)
            items = [ds[int(i)] for i in idx[b0:b0 + BATCH]]
            batch = collate(items)
            tokens = batch["tokens"].to(device)
            kw = {}
            if cfg.structural:
                b, l = tokens.shape
                kw = {"phi": torch.zeros(b, l, cfg.d_struct_raw,
                                         dtype=torch.float32, device=device),
                      "phi_valid": torch.zeros(b, l, dtype=torch.bool,
                                               device=device),
                      "symmetry": None}
            model(tokens, **kw)
            hs = grab["h"][:, ::stride, :].float().cpu().numpy()
            feats.append(hs.reshape(-1, hs.shape[-1]))
            for j in range(hs.shape[0]):
                wid.append(np.full(hs.shape[1], b0 + j, dtype=np.int64))
    finally:
        h.remove()
    return np.concatenate(feats), np.concatenate(wid)


def positional_targets(ds, idx, stride):
    """phi at each sampled position, plus window id and local composition."""
    A, C, G, T = 2, 3, 4, 5
    ys = {k: [] for k in TARGETS}
    comp, wid, valid = [], [], []
    for j, i in enumerate(idx):
        item = ds[int(i)]
        phi, ok = item["phi"], item["phi_valid"].astype(bool)
        tok = item["tokens"].astype(np.int64)
        sel = np.arange(0, len(tok), stride)
        for k, ch in TARGETS.items():
            ys[k].append(phi[sel, ch])
        valid.append(ok[sel])
        wid.append(np.full(len(sel), j, dtype=np.int64))
        for p in sel:
            lo, hi = max(0, p - GC_HALF), min(len(tok), p + GC_HALF)
            loc = tok[lo:hi]
            acgt = loc[(loc >= A) & (loc <= T)]
            n = max(len(acgt), 1)
            gc = float(((acgt == C) | (acgt == G)).sum()) / n
            dd = acgt[:-1] * 4 + acgt[1:] - (A * 4 + A)
            dd = dd[(dd >= 0) & (dd < 16)]
            di = np.bincount(dd, minlength=16).astype(np.float64)
            di /= max(di.sum(), 1.0)
            comp.append(np.concatenate([[gc], di]))
    return ({k: np.concatenate(v) for k, v in ys.items()},
            np.asarray(comp), np.concatenate(wid), np.concatenate(valid))


def centre_by_window(X, wid):
    """Subtract each window's own mean. Removes window identity entirely."""
    out = X.astype(np.float64, copy=True)
    order = np.argsort(wid, kind="stable")
    w_sorted = wid[order]
    bounds = np.flatnonzero(np.diff(w_sorted)) + 1
    for grp in np.split(order, bounds):
        out[grp] -= out[grp].mean(axis=0, keepdims=True)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device.")
        return 2
    device = torch.device("cuda", 0)
    use_scan("triton")

    ds_tr = WindowDataset("train", structural=True, rc_augment=False)
    ds_va = WindowDataset("val", structural=True, rc_augment=False)
    z = np.load(REPO / "data" / "processed" / "dataset_index.npz",
                allow_pickle=True)
    if [str(x) for x in z["feature_names"]] != EXPECTED_NAMES:
        print("phi channel order changed")
        return 2

    rng = np.random.default_rng(PROBE_SEED)
    idx_tr = np.arange(len(ds_tr))
    if args.n_train and args.n_train < len(idx_tr):
        idx_tr = np.sort(rng.choice(idx_tr, args.n_train, replace=False))
    idx_va = np.arange(len(ds_va))

    print("=" * 74)
    print("P5-POS -- positional probe (A global, B within-window)")
    print("=" * 74)
    print(f"  train {len(idx_tr):,} windows   val {len(idx_va):,} windows"
          f"   stride {args.stride} bp")
    print(f"  rows per window {32768 // args.stride}")
    print("  phi WITHHELD from the model (S0 zeros)")
    print()

    ytr, ctr, wtr, vtr = positional_targets(ds_tr, idx_tr, args.stride)
    yva, cva, wva, vva = positional_targets(ds_va, idx_va, args.stride)
    print(f"  rows: train {len(wtr):,}  val {len(wva):,}")
    print(f"  valid phi: train {vtr.sum():,}  val {vva.sum():,}")
    print()

    results = {"config": {
        "n_train_windows": int(len(idx_tr)), "n_val_windows": int(len(idx_va)),
        "stride": args.stride, "gc_half": GC_HALF,
        "phi_at_probe": "S0 zeros (withheld)",
        "probe_A": "predict phi_t from h_t",
        "probe_B": "predict phi_t - mean_w(phi) from h_t - mean_w(h)",
    }, "targets": {}}

    def run_pair(Xtr, Xva, label, store):
        for k in TARGETS:
            mt = vtr & np.isfinite(ytr[k])
            mv = vva & np.isfinite(yva[k])
            a = ridge_fit_eval(Xtr[mt], ytr[k][mt], Xva[mv], yva[k][mv])
            bt = centre_by_window(Xtr[mt], wtr[mt])
            bv = centre_by_window(Xva[mv], wva[mv])
            yb_t = centre_by_window(ytr[k][mt].reshape(-1, 1), wtr[mt]).ravel()
            yb_v = centre_by_window(yva[k][mv].reshape(-1, 1), wva[mv]).ravel()
            b = ridge_fit_eval(bt, yb_t, bv, yb_v)
            store.setdefault(k, {})[label] = {"A_global": a, "B_within": b}
            print(f"  {k:20s} A r={a['r']:+.4f}   B r={b['r']:+.4f}")

    print("-" * 74)
    print("FLOOR -- local composition (GC + dinucleotide, +/-2,560 bp)")
    run_pair(ctr, cva, "composition_floor", results["targets"])
    print()

    runs = [("structural", d) for d in sorted(NOVEL.glob("structural_seed*"))
            if (d / "checkpoint.pt").exists()]
    runs += [("baseline_v2", d) for d in sorted(BASE.glob("baseline_v2_seed*"))
             if (d / "checkpoint.pt").exists()]

    CACHE.mkdir(parents=True, exist_ok=True)
    per_run = {}
    for arm, d in runs:
        t0 = time.time()
        print("-" * 74)
        ktr = CACHE / f"{d.name}_tr_{len(idx_tr)}_{args.stride}.npy"
        kva = CACHE / f"{d.name}_va_{len(idx_va)}_{args.stride}.npy"
        if ktr.exists() and kva.exists():
            Xtr, Xva = np.load(ktr), np.load(kva)
            print(f"{d.name}  [{arm}]  cached", flush=True)
        else:
            model, cfg, step = load_any(d, device)
            print(f"{d.name}  [{arm}]  step {step:,}", flush=True)
            Xtr, _ = represent_positional(model, cfg, ds_tr, idx_tr, device,
                                          args.stride, "train")
            np.save(ktr, Xtr)
            Xva, _ = represent_positional(model, cfg, ds_va, idx_va, device,
                                          args.stride, "val")
            np.save(kva, Xva)
            del model
            torch.cuda.empty_cache()
        store = {}
        run_pair(Xtr, Xva, "model", store)
        per_run[d.name] = {"arm": arm,
                           "targets": {k: v["model"] for k, v in store.items()},
                           "seconds": round(time.time() - t0, 1)}
        print(f"  ({per_run[d.name]['seconds']:.0f} s)")

    results["runs"] = per_run

    print()
    print("=" * 74)
    print("ARM COMPARISON")
    print("=" * 74)
    for k in TARGETS:
        fl = results["targets"][k]["composition_floor"]
        print(f"\n  {k}")
        print(f"    floor        A {fl['A_global']['r']:+.4f}   "
              f"B {fl['B_within']['r']:+.4f}")
        for arm in ("structural", "baseline_v2"):
            ra = [v["targets"][k]["A_global"]["r"] for v in per_run.values()
                  if v["arm"] == arm]
            rb = [v["targets"][k]["B_within"]["r"] for v in per_run.values()
                  if v["arm"] == arm]
            if not ra:
                continue
            print(f"    {arm:12s} A {np.mean(ra):+.4f} +/- "
                  f"{np.std(ra, ddof=1) if len(ra) > 1 else 0:.4f}   "
                  f"B {np.mean(rb):+.4f} +/- "
                  f"{np.std(rb, ddof=1) if len(rb) > 1 else 0:.4f}")
            results["targets"][k][arm] = {"A_r": ra, "B_r": rb}

    print()
    print("  A = includes window identity (94% of phi variance is between")
    print("      windows, so a high A is mostly 'which window is this').")
    print("  B = window identity removed. This is the mechanism's own target.")
    print("  n=3 per arm: minimum attainable two-sided permutation p = 0.10.")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
