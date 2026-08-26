#!/usr/bin/env python
"""T2/A1 -- common-alpha confound check on the phi-variance stratification.

THE PROBLEM THIS ADDRESSES
--------------------------
phase5_repower.py's stratification result (rho = +0.121 between within-window
phi variance and the structural-minus-baseline advantage) fits each of the six
runs' ridge probes with its OWN LOO-GCV-selected alpha. The six selected alphas
are not equal across arms (structural ~3e5; baseline_v2_seed2 rails to the grid
maximum). A probe with a different effective regularisation per arm can shift
"who wins where" for reasons that have nothing to do with structure: heavier
regularisation shrinks predictions toward the mean, which changes how a probe's
r responds to a window's own target variance -- exactly the covariate (keep(phi))
this stratification claims to explain. If the alpha differences are large
enough, they alone could produce a spurious rho.

This script refits BOTH arms, all six runs, at each of three FIXED, SHARED
alpha values -- no per-run LOO-GCV -- and re-derives everything that depends on
the per-window structural/baseline difference: the quartile table, the
frac-positive monotone trend, and the Spearman rho with its block-length sweep
and CI-width reliability check. If the pattern survives a shared alpha, the
alpha confound is not the explanation. If it does not, it was.

Sweep is the geometric mean of the six LOO-GCV-selected alphas from
phase5_repower.py's own probe (recomputed here, not copied from any file: they
were printed to stdout there but never written to p5_repower.json), one decade
below and one decade above.

Writes results/novel_model/p5_common_alpha.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from phase1_dataset import interpolate_phi                    # noqa: E402
from phase5_structure_probe import (                          # noqa: E402
    ridge_fit_eval, TARGETS, EXPECTED_NAMES, PROBE_SEED)
from phase5_repower import (                                   # noqa: E402
    positional_targets, centre_by_window, block_resample_windows)

NOVEL = REPO / "results" / "novel_model"
CACHE = NOVEL / "p5_pos_cache"
OUT = NOVEL / "p5_common_alpha.json"
INDEX32 = REPO / "data" / "processed" / "dataset_index_w32768.npz"

STRIDE = 512
N_TRAIN = 500
WINDOW = 32_768
BOOT_SEED = 20260826
MIN_BLOCKS = 20
N_BOOT = 10_000
N_BOOT_RHO = 4_000
STRUCT = ["structural_seed0", "structural_seed1", "structural_seed2"]
BASELN = ["baseline_v2_seed0", "baseline_v2_seed1", "baseline_v2_seed2"]
TARGET = "insulation_100kb"
L_SWEEP = (1, 4, 8, 15, 31, 61)


def ridge_fixed_alpha(Xtr, ytr, Xva, yva, alpha):
    """Ridge at a FIXED alpha -- no LOO-GCV selection, same standardisation
    and closed form as ridge_fit_eval otherwise, so the only thing that
    changes between this and the original probe is whether alpha is chosen
    per-run or imposed."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = sd.copy()
    sd[sd < 1e-8] = 1.0
    Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
    ym = ytr.mean()
    yc = ytr - ym
    G = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1])
    w = np.linalg.solve(G, Ztr.T @ yc)
    pred = Zva @ w + ym
    if pred.std() < 1e-12 or yva.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(pred, yva)[0, 1])
    return {"r": r, "pred": pred}


def per_window_r(pred, yva, wv, n_win):
    rw = np.full(n_win, np.nan)
    for j in np.unique(wv):
        m = wv == j
        if m.sum() >= 8 and pred[m].std() > 1e-12 and yva[m].std() > 1e-12:
            rw[j] = np.corrcoef(pred[m], yva[m])[0, 1]
    return rw


def quartile_table(kk, d):
    qs = np.quantile(kk, [0.25, 0.5, 0.75])
    bins = np.digitize(kk, qs)
    strat = {}
    for q in range(4):
        m = bins == q
        if m.sum() == 0:
            continue
        dq = d[m]
        strat[f"Q{q + 1}"] = {
            "n": int(m.sum()),
            "phi_var_lo": float(kk[m].min()), "phi_var_hi": float(kk[m].max()),
            "mean_d": float(dq.mean()), "sd_d": float(dq.std(ddof=1)),
            "frac_positive": float((dq > 0).mean())}
    return strat, bins


def bootstrap_frac_monotone(d, bins, n_boot, seed):
    """Block bootstrap of the four quartiles' frac-positive, swept over block
    length. bins is a fixed per-window covariate (keep(phi) rank), so
    resampling contiguous blocks of window POSITIONS and reading off each
    resampled position's already-assigned bin preserves that covariate
    exactly -- only which window (and hence which d value) lands at a given
    position is resampled."""
    n = len(d)
    out = {}
    for L in L_SWEEP:
        if L > n:
            continue
        take = block_resample_windows(n, L, n_boot, np.random.default_rng(seed))
        fracs = np.full((n_boot, 4), np.nan)
        for q in range(4):
            qm = bins == q
            if qm.sum() == 0:
                continue
            for b in range(n_boot):
                idx = take[b]
                sel = qm[idx]
                if sel.sum() > 0:
                    fracs[b, q] = (d[idx][sel] > 0).mean()
        ci = {}
        for q in range(4):
            col = fracs[:, q]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            lo, hi = np.percentile(col, [2.5, 97.5])
            ci[f"Q{q + 1}"] = {"lo": float(lo), "hi": float(hi),
                               "point": float((d[bins == q] > 0).mean())}
        mono = np.nanmean([
            (fracs[b, 0] <= fracs[b, 1] <= fracs[b, 2] <= fracs[b, 3])
            for b in range(n_boot)
            if np.isfinite(fracs[b]).all()
        ]) if np.isfinite(fracs).all(axis=1).any() else float("nan")
        nb = int(np.ceil(n / L))
        out[str(L)] = {"block_len_windows": L, "n_blocks": nb,
                       "reliable": bool(nb >= MIN_BLOCKS),
                       "quartile_ci95": ci,
                       "frac_boot_replicates_monotone": (
                           float(mono) if mono == mono else None)}
    return out


def rho_sweep(kk, d, seed):
    n = len(d)
    out = {}
    for L in L_SWEEP:
        if L > n:
            continue
        take = block_resample_windows(n, L, N_BOOT_RHO, np.random.default_rng(seed))
        rb = np.array([spearmanr(kk[s], d[s]).statistic for s in take])
        lo, hi = np.percentile(rb, [2.5, 97.5])
        pp = float(min(2.0 * min((rb <= 0).mean(), (rb >= 0).mean()), 1.0))
        nb = int(np.ceil(n / L))
        out[str(L)] = {"block_len_windows": L, "n_blocks": nb,
                       "p_two_sided": pp, "ci95_lo": float(lo), "ci95_hi": float(hi),
                       "ci_width": float(hi - lo), "reliable": bool(nb >= MIN_BLOCKS)}
    return out


def main() -> int:
    z = np.load(INDEX32, allow_pickle=True)
    assert int(z["window"]) == WINDOW
    names = [str(x) for x in z["feature_names"]]
    assert names == EXPECTED_NAMES, "phi channel order changed"
    phi, usable = z["phi"], z["usable"].astype(bool)
    starts_tr_all, starts_va = z["train"], z["val"]
    n_win = len(starts_va)

    rng = np.random.default_rng(PROBE_SEED)
    idx_tr = np.arange(len(starts_tr_all))
    if N_TRAIN < len(idx_tr):
        idx_tr = np.sort(rng.choice(idx_tr, N_TRAIN, replace=False))
    starts_tr = starts_tr_all[idx_tr]

    ytr, wtr, vtr = positional_targets(starts_tr, phi, usable, STRIDE)
    yva, wva, vva = positional_targets(starts_va, phi, usable, STRIDE)
    ch_tr, ch_va = ytr[TARGET], yva[TARGET]
    mt = vtr & np.isfinite(ch_tr)
    mv = vva & np.isfinite(ch_va)
    ybt = centre_by_window(ch_tr[mt].reshape(-1, 1), wtr[mt]).ravel()
    ybv = centre_by_window(ch_va[mv].reshape(-1, 1), wva[mv]).ravel()
    wv = wva[mv]

    # per-window keep(phi) -- a fixed covariate of the TARGET only, independent
    # of which alpha (or which arm) predicts it
    wvar = np.array([np.var(ybv[wv == j]) for j in range(n_win)])
    gvar = float(np.var(ybv))
    keep_w = wvar / gvar if gvar > 0 else np.full(n_win, np.nan)

    # ---- step 1: reproduce the six LOO-GCV-selected alphas (not persisted by
    # phase5_repower.py -- printed to stdout only)
    print("=" * 74)
    print("STEP 1 -- recover the six LOO-GCV-selected alphas")
    print("=" * 74)
    Xtr_cache, Xva_cache = {}, {}
    selected_alpha = {}
    for name in STRUCT + BASELN:
        ktr = CACHE / f"{name}_tr_{len(idx_tr)}_{STRIDE}.npy"
        kva = CACHE / f"{name}_va_{len(starts_va)}_{STRIDE}.npy"
        Xtr, Xva = np.load(ktr), np.load(kva)
        bt = centre_by_window(Xtr[mt], wtr[mt])
        bv = centre_by_window(Xva[mv], wv)
        Xtr_cache[name], Xva_cache[name] = bt, bv
        res = ridge_fit_eval(bt, ybt, bv, ybv, with_pred=False)
        selected_alpha[name] = res["alpha"]
        print(f"  {name:20s} alpha = {res['alpha']:.6g}"
              f"  (grid edge: {res['alpha_at_grid_edge']})")

    logs = np.log10(np.array(list(selected_alpha.values())))
    geo_mean = float(10 ** logs.mean())
    sweep_alphas = {"geo_mean/10": geo_mean / 10.0,
                    "geo_mean": geo_mean,
                    "geo_mean*10": geo_mean * 10.0}
    print(f"\n  geometric mean of the six selected alphas = {geo_mean:.6g}")
    print(f"  sweep: {[f'{v:.6g}' for v in sweep_alphas.values()]}")

    results = {}
    print()
    print("=" * 74)
    print("STEP 2 -- refit at a SHARED alpha, all six runs, and re-derive")
    print("=" * 74)
    for label, alpha in sweep_alphas.items():
        print("-" * 74)
        print(f"alpha = {alpha:.6g}  ({label})")
        print("-" * 74)
        rw = {}
        pooled_r = {}
        for name in STRUCT + BASELN:
            fit = ridge_fixed_alpha(Xtr_cache[name], ybt, Xva_cache[name], ybv, alpha)
            rw[name] = per_window_r(fit["pred"], ybv, wv, n_win)
            pooled_r[name] = fit["r"]
            print(f"  {name:20s} pooled r = {fit['r']:+.4f}")

        S = np.vstack([rw[n] for n in STRUCT])
        B = np.vstack([rw[n] for n in BASELN])
        ok = np.isfinite(S).all(0) & np.isfinite(B).all(0)
        d = S[:, ok].mean(0) - B[:, ok].mean(0)
        kk = keep_w[ok]
        print(f"  windows scored: {ok.sum()}/{n_win}")
        print(f"  d_w: mean {d.mean():+.5f} sd {d.std(ddof=1):.5f} "
              f"{(d > 0).sum()}/{len(d)} positive")

        strat, bins = quartile_table(kk, d)
        print(f"  {'quartile':>10s} {'phi var range':>22s} {'n':>5s} "
              f"{'mean d_w':>10s} {'frac > 0':>9s}")
        for q in range(1, 5):
            s = strat.get(f"Q{q}")
            if s:
                print(f"  {'Q' + str(q):>10s} {s['phi_var_lo']:9.4f} - "
                      f"{s['phi_var_hi']:9.4f} {s['n']:5d} "
                      f"{s['mean_d']:+10.5f} {s['frac_positive']:9.3f}")

        fracs = [strat[f"Q{i}"]["frac_positive"] for i in range(1, 5) if f"Q{i}" in strat]
        monotone = all(fracs[i] <= fracs[i + 1] for i in range(len(fracs) - 1))
        q1_negative = strat.get("Q1", {}).get("mean_d", 0.0) < 0
        print(f"  Q1 mean_d negative: {q1_negative}   frac>0 monotone: {monotone}")

        rho, p_naive = spearmanr(kk, d)
        rsweep = rho_sweep(kk, d, BOOT_SEED)
        rel_Ls = [int(k) for k, v in rsweep.items() if v["reliable"]]
        L_rho = max(rel_Ls) if rel_Ls else None
        print(f"  Spearman rho = {rho:+.4f} (naive p {p_naive:.4f}, NOT valid)")
        if L_rho is not None:
            br = rsweep[str(L_rho)]
            print(f"  block bootstrap, largest reliable L = {L_rho} "
                  f"({L_rho * WINDOW / 1e6:.3f} Mb, {br['n_blocks']} blocks): "
                  f"p = {br['p_two_sided']:.4f}  CI [{br['ci95_lo']:+.4f},{br['ci95_hi']:+.4f}]")
        for L in L_SWEEP:
            if str(L) in rsweep:
                v = rsweep[str(L)]
                flag = "" if v["reliable"] else "  (too few blocks)"
                print(f"    L={L:3d} ({L*WINDOW/1e6:.3f} Mb) n_blocks={v['n_blocks']:3d} "
                      f"p={v['p_two_sided']:.4f} CI[{v['ci95_lo']:+.4f},{v['ci95_hi']:+.4f}] "
                      f"width={v['ci_width']:.4f}{flag}")

        fboot = bootstrap_frac_monotone(d, bins, N_BOOT, BOOT_SEED)

        results[label] = {
            "alpha": alpha,
            "per_run_pooled_r": pooled_r,
            "n_scored": int(ok.sum()),
            "d": {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                  "n_positive": int((d > 0).sum()), "n": int(len(d))},
            "quartiles": strat,
            "gate_q1_negative": bool(q1_negative),
            "gate_frac_monotone": bool(monotone),
            "spearman_rho": float(rho),
            "spearman_p_naive_INVALID": float(p_naive),
            "spearman_block_sweep": rsweep,
            "spearman_reliable_max_L": L_rho,
            "frac_positive_block_bootstrap": fboot,
        }

    overall_pass = all(results[k]["gate_q1_negative"] and results[k]["gate_frac_monotone"]
                       for k in sweep_alphas)
    print()
    print("=" * 74)
    print("GATE -- does the pattern survive a shared, non-per-arm alpha?")
    print("=" * 74)
    for label in sweep_alphas:
        r = results[label]
        print(f"  {label:12s} alpha={r['alpha']:.4g}  "
              f"Q1<0: {r['gate_q1_negative']}   frac>0 monotone: {r['gate_frac_monotone']}   "
              f"rho={r['spearman_rho']:+.4f}")
    print(f"\n  OVERALL: {'SURVIVES' if overall_pass else 'FAILS'} at all three "
          f"alpha settings in the sweep.")

    OUT.write_text(json.dumps({
        "purpose": "A1 common-alpha confound check on the phi-variance stratification",
        "selected_alphas_recovered": selected_alpha,
        "geo_mean_alpha": geo_mean,
        "sweep": results,
        "overall_gate_pass": bool(overall_pass),
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
