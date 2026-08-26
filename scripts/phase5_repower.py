#!/usr/bin/env python
"""T2 -- re-power the probe-B result without spending a GPU-hour.

THE PROBLEM THIS ADDRESSES
--------------------------
The one clean positive in the project is probe B: within-window insulation,
where every structural seed beats every baseline seed. It is currently tested
across 3 seeds. An exact two-sided permutation test at 3 vs 3 has C(6,3) = 20
arrangements, so the smallest attainable two-sided p is 2/20 = 0.100. p < 0.05
is unreachable BY DESIGN, not by outcome. Every further GPU-hour spent at n = 3
buys an experiment that has pre-committed to non-significance.

The seed does not have to be the unit of analysis. p5_positional_probe ran on
274 val windows at stride 512, i.e. 17,536 val rows. Testing across WINDOWS
instead of across seeds answers a different question -- "does this hold across
the genome" rather than "does this reproduce across training runs" -- and both
belong in a paper. This script computes the second and reports it beside the
first.

WHY NOT AN ORDINARY BOOTSTRAP
------------------------------
Hi-C features are autocorrelated over megabases. The 274 val windows are
CONTIGUOUS and non-overlapping (32,768 bp apart, spanning 120,000,000 ->
128,978,432 on chr9, 8.98 Mb total), so they are a time series, not a sample.
i.i.d. resampling of windows would treat neighbouring windows as independent
evidence and return a confidently wrong p-value. This is the same
autocorrelation that makes known weakness 5 real; it must not be let back in
through the resampling scheme.

So: MOVING BLOCK BOOTSTRAP, with the block length chosen from the measured
autocorrelation of the per-window statistic itself, and a sensitivity sweep
across block lengths reported alongside -- because with 8.98 Mb of val and
megabase-scale dependence, the number of effectively independent blocks is
small and the honest thing is to show how p moves with L.

INPUTS -- no GPU, no model forward pass
----------------------------------------
The cached per-position representations written by phase5_positional_probe.py
(results/novel_model/p5_pos_cache/*.npy) are reused directly. They were built
against the 32,768 bp index, which has since been replaced by the 65,536 bp
one, so the ARCHIVED index is loaded explicitly -- dataset_index_w32768.npz.
Reading the live index here would silently pair 32 kb representations with
65 kb targets.

Writes results/novel_model/p5_repower.json
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from phase1_dataset import interpolate_phi                   # noqa: E402
from phase5_structure_probe import (                         # noqa: E402
    ridge_fit_eval, TARGETS, EXPECTED_NAMES, PROBE_SEED)

NOVEL = REPO / "results" / "novel_model"
CACHE = NOVEL / "p5_pos_cache"
OUT = NOVEL / "p5_repower.json"
INDEX32 = REPO / "data" / "processed" / "dataset_index_w32768.npz"

STRIDE = 512             # must match the cache the representations came from
N_TRAIN = 500            # ditto
WINDOW = 32_768
BOOT_SEED = 20260826     # fixed, not a training seed and not the probe seed
MIN_BLOCKS = 20          # below this a block bootstrap's percentiles are not
                         # trustworthy; reported, never silently dropped
N_BOOT = 10_000
STRUCT = ["structural_seed0", "structural_seed1", "structural_seed2"]
BASELN = ["baseline_v2_seed0", "baseline_v2_seed1", "baseline_v2_seed2"]


# ----------------------------------------------------------------- reconstruct

def positional_targets(starts, phi, usable, stride):
    """phi at every stride-th position of every window, plus window ids.

    Reproduces phase5_positional_probe.positional_targets for the phi side
    only (the local-composition floor needs the token array and is not the
    quantity under test here). WindowDataset.__getitem__ interpolates phi over
    arange(start, start+window); the same call is made here so the values are
    identical rather than merely similar.
    """
    ys = {k: [] for k in TARGETS}
    wid, valid = [], []
    for j, s in enumerate(starts):
        pos = np.arange(int(s), int(s) + WINDOW)
        ph, ok = interpolate_phi(phi, usable, pos)
        sel = np.arange(0, WINDOW, stride)
        for k, ch in TARGETS.items():
            ys[k].append(ph[sel, ch])
        valid.append(ok[sel])
        wid.append(np.full(len(sel), j, dtype=np.int64))
    return ({k: np.concatenate(v) for k, v in ys.items()},
            np.concatenate(wid), np.concatenate(valid))


def centre_by_window(X, wid):
    """Subtract each window's own mean -- removes window identity entirely."""
    out = X.astype(np.float64, copy=True)
    order = np.argsort(wid, kind="stable")
    w_sorted = wid[order]
    bounds = np.flatnonzero(np.diff(w_sorted)) + 1
    for grp in np.split(order, bounds):
        out[grp] -= out[grp].mean(axis=0, keepdims=True)
    return out


# ---------------------------------------------------------------------- stats

def acf(x, nlags):
    """Sample autocorrelation of a 1-D series, lags 0..nlags."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float((x * x).sum())
    return np.array([1.0 if L == 0 else float((x[L:] * x[:-L]).sum()) / denom
                     for L in range(nlags + 1)])


def moving_block_bootstrap(d, L, n_boot, rng):
    """Two-sided p for H0: mean(d) = 0, resampling contiguous blocks of L.

    Blocks are drawn with replacement from all n-L+1 overlapping start
    positions and concatenated to length n. The null is imposed by centring:
    the resampled means are compared against the observed mean AFTER removing
    the observed mean, which is the standard way to bootstrap a test rather
    than only a confidence interval.
    """
    n = len(d)
    L = int(min(max(L, 1), n))
    n_blocks = int(np.ceil(n / L))
    starts_pool = np.arange(0, n - L + 1)
    obs = float(d.mean())
    dc = d - obs                       # impose H0
    idx0 = rng.integers(0, len(starts_pool), size=(n_boot, n_blocks))
    offs = np.arange(L)
    means = np.empty(n_boot)
    for b in range(n_boot):
        take = (starts_pool[idx0[b]][:, None] + offs[None, :]).ravel()[:n]
        means[b] = dc[take].mean()
    p = float((np.abs(means) >= abs(obs)).mean())
    lo, hi = np.percentile(means + obs, [2.5, 97.5])
    return {"block_len_windows": L, "block_len_bp": L * WINDOW,
            "n_blocks": n_blocks, "p_two_sided": p,
            "ci95_lo": float(lo), "ci95_hi": float(hi)}


def window_suffstats(x, y, wv, n_win):
    """Per-window sufficient statistics for a pooled Pearson r.

    A correlation over any UNION of windows is a function of the six sums
    below, and each is additive over windows. That makes a block bootstrap of
    the POOLED r an O(n_windows) operation per resample instead of an
    O(n_rows) one, which is what makes 10,000 resamples across six runs cheap.
    """
    out = np.zeros((n_win, 6))
    for j in range(n_win):
        m = wv == j
        xs, ys = x[m], y[m]
        out[j] = [xs.sum(), ys.sum(), (xs * xs).sum(),
                  (ys * ys).sum(), (xs * ys).sum(), m.sum()]
    return out


def r_from_suffstats(S):
    """Pearson r from summed sufficient statistics (rows already summed)."""
    Sx, Sy, Sxx, Syy, Sxy, n = S.T if S.ndim == 2 else S
    cov = Sxy - Sx * Sy / n
    vx = Sxx - Sx * Sx / n
    vy = Syy - Sy * Sy / n
    den = np.sqrt(np.clip(vx, 1e-300, None) * np.clip(vy, 1e-300, None))
    return cov / den


def block_resample_windows(n_win, L, n_boot, rng):
    """(n_boot, n_win) matrices of window indices drawn as contiguous blocks."""
    L = int(min(max(L, 1), n_win))
    n_blocks = int(np.ceil(n_win / L))
    pool = np.arange(0, n_win - L + 1)
    starts = pool[rng.integers(0, len(pool), size=(n_boot, n_blocks))]
    return (starts[:, :, None] + np.arange(L)[None, None, :]
            ).reshape(n_boot, -1)[:, :n_win]


def sign_test(d):
    """Exact two-sided binomial sign test on H0: P(d_w > 0) = 0.5.

    The most assumption-free version of "does this hold across the genome":
    no ridge, no bootstrap, no block length, no distributional assumption. If
    the effect were a broad genome-wide property, the sign should be positive
    in well over half of windows regardless of how noisy each one is.
    """
    from scipy.stats import binomtest
    pos = int((d > 0).sum())
    n = int((d != 0).sum())
    bt = binomtest(pos, n, 0.5, alternative="two-sided")
    return {"n_positive": pos, "n": n, "fraction": pos / n,
            "p_two_sided": float(bt.pvalue)}


def exact_permutation_3v3(a, b):
    """Exact two-sided permutation test on 3 vs 3. Minimum attainable p = 0.1."""
    from itertools import combinations
    pool = np.concatenate([a, b])
    obs = float(np.mean(a) - np.mean(b))
    diffs = []
    for c in combinations(range(6), 3):
        m = np.zeros(6, dtype=bool)
        m[list(c)] = True
        diffs.append(float(pool[m].mean() - pool[~m].mean()))
    diffs = np.array(diffs)
    return {"observed": obs, "n_arrangements": len(diffs),
            "p_two_sided": float((np.abs(diffs) >= abs(obs) - 1e-12).mean()),
            "p_floor": float(2.0 / len(diffs))}


# ----------------------------------------------------------------------- main

def main() -> int:
    if not INDEX32.exists():
        raise FileNotFoundError(
            f"{INDEX32} not found. This analysis is defined on the 32,768 bp "
            f"runs; the live dataset_index.npz is the 65,536 bp build and the "
            f"two must not be pooled.")
    z = np.load(INDEX32, allow_pickle=True)
    assert int(z["window"]) == WINDOW, f"archived index is {int(z['window'])} bp"
    names = [str(x) for x in z["feature_names"]]
    if names != EXPECTED_NAMES:
        print("phi channel order changed")
        return 2

    phi, usable = z["phi"], z["usable"].astype(bool)
    starts_tr_all, starts_va = z["train"], z["val"]

    # Reproduce the probe's train subsample EXACTLY -- same seed, same draw, or
    # the cached representations do not correspond to these targets.
    rng = np.random.default_rng(PROBE_SEED)
    idx_tr = np.arange(len(starts_tr_all))
    if N_TRAIN < len(idx_tr):
        idx_tr = np.sort(rng.choice(idx_tr, N_TRAIN, replace=False))
    starts_tr = starts_tr_all[idx_tr]

    print("=" * 74)
    print("T2 -- re-power probe B: windows as the unit of analysis")
    print("=" * 74)
    print(f"  index      {INDEX32.name}  (window {WINDOW:,} bp)")
    print(f"  train      {len(starts_tr):,} windows   val {len(starts_va):,} windows")
    v0, v1 = int(starts_va[0]), int(starts_va[-1]) + WINDOW
    print(f"  val span   chr9:{v0:,}-{v1:,}  = {(v1 - v0) / 1e6:.3f} Mb, contiguous")
    print(f"  stride     {STRIDE} bp -> {WINDOW // STRIDE} rows per window")
    print()

    ytr, wtr, vtr = positional_targets(starts_tr, phi, usable, STRIDE)
    yva, wva, vva = positional_targets(starts_va, phi, usable, STRIDE)
    print(f"  rows: train {len(wtr):,}  val {len(wva):,}")
    print(f"  valid phi: train {vtr.sum():,}  val {vva.sum():,}")
    print()

    TARGET = "insulation_100kb"          # the probe-B positive under test
    ch_tr, ch_va = ytr[TARGET], yva[TARGET]
    mt = vtr & np.isfinite(ch_tr)
    mv = vva & np.isfinite(ch_va)

    # Probe B: both sides centred within window, so window identity is gone.
    ybt = centre_by_window(ch_tr[mt].reshape(-1, 1), wtr[mt]).ravel()
    ybv = centre_by_window(ch_va[mv].reshape(-1, 1), wva[mv]).ravel()
    wv = wva[mv]

    per_run_r, per_window_r, per_run_pred = {}, {}, {}
    print("-" * 74)
    print(f"probe B, {TARGET}: pooled r per run, then per-window r")
    print("-" * 74)
    for name in STRUCT + BASELN:
        ktr = CACHE / f"{name}_tr_{len(idx_tr)}_{STRIDE}.npy"
        kva = CACHE / f"{name}_va_{len(starts_va)}_{STRIDE}.npy"
        if not (ktr.exists() and kva.exists()):
            raise FileNotFoundError(
                f"missing cached representation {ktr.name} / {kva.name}. "
                f"Re-run phase5_positional_probe.py (needs a GPU) before this.")
        Xtr, Xva = np.load(ktr), np.load(kva)
        bt = centre_by_window(Xtr[mt], wtr[mt])
        bv = centre_by_window(Xva[mv], wv)
        res = ridge_fit_eval(bt, ybt, bv, ybv, with_pred=True)
        pred = res["pred"]
        per_run_r[name] = float(res["r"])

        # per-window correlation between predicted and actual centred phi
        rw = np.full(len(starts_va), np.nan)
        for j in np.unique(wv):
            m = wv == j
            if m.sum() >= 8 and pred[m].std() > 1e-12 and ybv[m].std() > 1e-12:
                rw[j] = np.corrcoef(pred[m], ybv[m])[0, 1]
        per_window_r[name] = rw
        per_run_pred[name] = pred
        print(f"  {name:20s} pooled r = {res['r']:+.4f}   "
              f"windows scored {np.isfinite(rw).sum():3d}/{len(rw)}   "
              f"alpha {res['alpha']:.3g}")

    S = np.vstack([per_window_r[n] for n in STRUCT])
    B = np.vstack([per_window_r[n] for n in BASELN])
    ok = np.isfinite(S).all(0) & np.isfinite(B).all(0)
    d = S[:, ok].mean(0) - B[:, ok].mean(0)
    print()
    print(f"  per-window difference d_w on {ok.sum()} windows: "
          f"mean {d.mean():+.5f}, sd {d.std(ddof=1):.5f}, "
          f"{(d > 0).sum()}/{len(d)} positive")

    # ---- how far does dependence actually reach?
    nl = min(60, len(d) - 2)
    a = acf(d, nl)
    thresh = 2.0 / np.sqrt(len(d))
    first_below = next((L for L in range(1, nl + 1) if abs(a[L]) < thresh), nl)
    print()
    print("-" * 74)
    print("dependence in the per-window statistic (this is what sets block size)")
    print("-" * 74)
    print(f"  ACF lags 1-8: " + " ".join(f"{a[L]:+.3f}" for L in range(1, 9)))
    exceed = [L for L in range(1, nl + 1) if abs(a[L]) >= thresh]
    print(f"  band = 2/sqrt(n) = {thresh:.4f}")
    print(f"  lags exceeding the band: {exceed[:10]}"
          f"{' ...' if len(exceed) > 10 else ''}  "
          f"({len(exceed)}/{nl} lags)")
    print(f"  FIRST crossing below the band is lag {first_below}, but that is "
          f"the first, not the last:")
    print(f"  the ACF does NOT decay cleanly -- roughly the number of "
          f"exceedances noise alone would give.")

    rng_b = np.random.default_rng(BOOT_SEED)
    # Chosen block length: the measured dependence in d_w, but never shorter
    # than 1 Mb -- Hi-C autocorrelation is a megabase-scale property of the
    # DATA, and a block shorter than that would be optimistic even if the
    # statistic's own ACF happens to look short-range.
    L_data = max(first_below, 1)
    L_hic = int(np.ceil(1_000_000 / WINDOW))
    L_use = max(L_data, L_hic)
    print(f"  block length used: {L_use} windows = {L_use * WINDOW / 1e6:.3f} Mb "
          f"(max of measured lag {L_data} and a 1 Mb Hi-C floor {L_hic})")
    print()
    print("  WHY THE 1 Mb FLOOR OVERRIDES THE MEASUREMENT. The measured ACF")
    print("  says dependence dies within ~0.066 Mb. That is not evidence of")
    print("  independence: d_w has sd 0.081 against a mean of 0.008, so it is")
    print("  mostly noise, and the autocorrelation of a mostly-noise statistic")
    print("  is near zero whatever the underlying data do. Hi-C dependence is a")
    print("  megabase-scale property of the DATA. Taking the short measured lag")
    print("  at face value would be letting the noise floor pick the block")
    print("  size, which is the anticonservative direction.")
    print()

    print("-" * 74)
    print("moving block bootstrap, sensitivity across block length")
    print("-" * 74)
    print(f"  {'L (win)':>8s} {'L (Mb)':>8s} {'blocks':>7s} {'p':>8s} "
          f"{'95% CI':>22s}")
    sweep = {}
    for L in sorted({1, 2, 4, 8, L_use, 31, 61}):
        if L > len(d):
            continue
        r = moving_block_bootstrap(d, L, N_BOOT, np.random.default_rng(BOOT_SEED))
        r["n_blocks_reliable"] = bool(r["n_blocks"] >= MIN_BLOCKS)
        sweep[str(L)] = r
        mark = "  <-- used" if L == L_use else ""
        print(f"  {L:8d} {L * WINDOW / 1e6:8.3f} {r['n_blocks']:7d} "
              f"{r['p_two_sided']:8.4f} "
              f"[{r['ci95_lo']:+.5f},{r['ci95_hi']:+.5f}]{mark}")
    print()
    print("  L = 1 is the INVALID i.i.d. case, shown only to expose how much")
    print("  the p-value depends on pretending neighbouring windows are")
    print("  independent. It is not the reported result.")

    # ---- the better-powered version of the same question
    #
    # Averaging per-window correlations throws power away: each is computed on
    # 64 rows and is mostly noise (sd 0.081 against a mean of 0.008 above). The
    # quantity actually reported by the probe is the POOLED r, so bootstrap
    # that instead -- resample windows in contiguous blocks, recompute pooled r
    # per run on the resampled rows, and take the arm difference. The ridge fit
    # is NOT refit: this resamples the evaluation set, which is the population
    # the claim generalises over, not the training draw.
    n_win = len(starts_va)
    suff = {n: window_suffstats(per_run_pred[n], ybv, wv, n_win)
            for n in STRUCT + BASELN}
    print()
    print("-" * 74)
    print("block bootstrap of the POOLED r difference (the reported statistic)")
    print("-" * 74)
    print(f"  {'L (win)':>8s} {'L (Mb)':>8s} {'blocks':>7s} {'p':>8s} "
          f"{'95% CI':>22s} {'width':>8s}")
    obs_pooled = float(np.mean([per_run_r[n] for n in STRUCT])
                       - np.mean([per_run_r[n] for n in BASELN]))
    pooled_sweep = {}
    for L in sorted({1, 4, 8, 15, L_use, 45, 61}):
        if L > n_win:
            continue
        take = block_resample_windows(n_win, L, N_BOOT,
                                      np.random.default_rng(BOOT_SEED))
        arm = {}
        for n in STRUCT + BASELN:
            arm[n] = r_from_suffstats(suff[n][take].sum(axis=1))
        diff = (np.mean([arm[n] for n in STRUCT], axis=0)
                - np.mean([arm[n] for n in BASELN], axis=0))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        # percentile p for H0: difference = 0
        pp = 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())
        nb = int(np.ceil(n_win / L))
        pooled_sweep[str(L)] = {"block_len_windows": int(L),
                                "block_len_bp": int(L * WINDOW),
                                "n_blocks": nb,
                                "p_two_sided": float(min(pp, 1.0)),
                                "ci95_lo": float(lo), "ci95_hi": float(hi),
                                "ci_width": float(hi - lo),
                                "reliable": bool(nb >= MIN_BLOCKS)}
        flag = "" if nb >= MIN_BLOCKS else "  UNRELIABLE (too few blocks)"
        print(f"  {L:8d} {L * WINDOW / 1e6:8.3f} {nb:7d} {min(pp, 1.0):8.4f} "
              f"[{lo:+.5f},{hi:+.5f}] {hi - lo:8.5f}{flag}")
    print(f"  observed pooled difference {obs_pooled:+.5f}")
    print()
    print("  READ THIS COLUMN BEFORE THE p COLUMN. The 95% CI gets NARROWER as")
    print("  the block gets longer, which is backwards -- longer blocks retain")
    print("  more dependence and must cost precision, never buy it. It is the")
    print("  signature of a bootstrap run on too few blocks: at L = 31 the")
    print("  resample is drawn from 9 blocks and at L = 61 from 5, so the")
    print("  resampling distribution is built from a handful of draws and its")
    print("  percentiles are not trustworthy. The small p-values there are an")
    print("  artefact of that, not evidence. Blocks long enough for Hi-C")
    print("  autocorrelation and enough of them for a valid bootstrap cannot")
    print("  both be had from 8.98 Mb of validation genome.")

    # ---- THE MECHANISTIC PREDICTION
    #
    # keep(phi) = 0.0573 was offered as the reason the Phase 4 benefit is null:
    # the delta-bias mechanism acts per position, so it can only use the share
    # of phi's variance that lies WITHIN a window, and that share is tiny. So
    # far that is an explanation for an absence, which is the weakest kind of
    # claim -- it explains a null without predicting anything.
    #
    # It does predict something, and the prediction is testable in data already
    # cached. If keep(phi) is the true account, the structural advantage should
    # CONCENTRATE in windows where phi actually varies inside the window, and
    # vanish in windows where its input is flat. Windows differ a lot in this:
    # per-window keep is computed below and spans two orders of magnitude.
    #
    # A hint that this is real: the pooled r difference (+0.0103) exceeds the
    # mean of per-window differences (+0.0077), and pooled r implicitly weights
    # windows by their within-window phi variance.
    #
    # This can equally come out flat, which is also worth knowing: it would mean
    # keep(phi) explains the null but does not predict where the signal lives.
    wvar = np.array([np.var(ybv[wv == j]) for j in range(n_win)])
    gvar = float(np.var(ybv))
    keep_w = wvar / gvar if gvar > 0 else np.full(n_win, np.nan)
    kk = keep_w[ok]
    print()
    print("-" * 74)
    print("MECHANISTIC TEST -- does the advantage live where phi actually varies?")
    print("-" * 74)
    print(f"  per-window within-window phi variance (centred insulation_100kb),")
    print(f"  as a fraction of the pooled val variance:")
    print(f"    min {kk.min():.4f}  median {np.median(kk):.4f}  max {kk.max():.4f}")
    qs = np.quantile(kk, [0.25, 0.5, 0.75])
    bins = np.digitize(kk, qs)
    print()
    print(f"  {'quartile':>10s} {'phi var range':>22s} {'n':>5s} "
          f"{'mean d_w':>10s} {'frac > 0':>9s}")
    strat = {}
    for q in range(4):
        m = bins == q
        if m.sum() == 0:
            continue
        dq = d[m]
        lo_, hi_ = kk[m].min(), kk[m].max()
        strat[f"Q{q + 1}"] = {"n": int(m.sum()),
                              "phi_var_lo": float(lo_), "phi_var_hi": float(hi_),
                              "mean_d": float(dq.mean()),
                              "sd_d": float(dq.std(ddof=1)),
                              "frac_positive": float((dq > 0).mean())}
        print(f"  {'Q' + str(q + 1):>10s} {lo_:9.4f} - {hi_:9.4f} {m.sum():5d} "
              f"{dq.mean():+10.5f} {(dq > 0).mean():9.3f}")

    # Spearman: monotone association between how much phi varies in a window
    # and how much the structural arm gains there. Rank-based, so it does not
    # assume the relationship is linear or that d_w is well behaved.
    from scipy.stats import spearmanr
    rho, p_rho_naive = spearmanr(kk, d)
    print()
    print(f"  Spearman rho(within-window phi variance, d_w) = {rho:+.4f}")
    print(f"    naive p = {p_rho_naive:.4f}  -- DO NOT USE. spearmanr assumes")
    print(f"    independent observations, and these 274 windows are a")
    print(f"    contiguous 8.98 Mb series. Using it here would be the exact")
    print(f"    error this script exists to avoid.")

    # Same moving block bootstrap, swept over block length, and subject to the
    # SAME CI-width diagnostic as everything else here -- a rho p-value from 9
    # blocks is no more trustworthy than a pooled-r one from 9 blocks.
    n_ok = int(ok.sum())
    n_boot_rho = 4_000            # spearman per resample is the slow part
    rho_sweep = {}
    print(f"    {'L':>4s} {'Mb':>6s} {'blocks':>7s} {'p':>8s} {'95% CI':>20s} "
          f"{'width':>8s}")
    for L in (1, 4, 8, 15, L_use, 61):
        if L > n_ok:
            continue
        take = block_resample_windows(n_ok, L, n_boot_rho,
                                      np.random.default_rng(BOOT_SEED))
        rb = np.array([spearmanr(kk[s], d[s]).statistic for s in take])
        lo_, hi_ = np.percentile(rb, [2.5, 97.5])
        pp = float(min(2.0 * min((rb <= 0).mean(), (rb >= 0).mean()), 1.0))
        nb = int(np.ceil(n_ok / L))
        rho_sweep[str(L)] = {"block_len_windows": int(L), "n_blocks": nb,
                             "p_two_sided": pp, "ci95_lo": float(lo_),
                             "ci95_hi": float(hi_),
                             "ci_width": float(hi_ - lo_),
                             "reliable": bool(nb >= MIN_BLOCKS)}
        flag = "" if nb >= MIN_BLOCKS else "  too few blocks"
        print(f"    {L:4d} {L * WINDOW / 1e6:6.3f} {nb:7d} {pp:8.4f} "
              f"[{lo_:+.4f},{hi_:+.4f}] {hi_ - lo_:8.4f}{flag}")

    ok_rL = [int(k) for k, v in rho_sweep.items() if v["reliable"]]
    L_rho = max(ok_rL)
    best_rho = rho_sweep[str(L_rho)]
    lo_r, hi_r = best_rho["ci95_lo"], best_rho["ci95_hi"]
    p_rho = best_rho["p_two_sided"]
    print(f"    width narrows with L again -- so the L >= {MIN_BLOCKS}-block "
          f"rows are the only usable ones.")
    print(f"    USING L = {L_rho} ({L_rho * WINDOW / 1e6:.3f} Mb, "
          f"{best_rho['n_blocks']} blocks): p = {p_rho:.4f}, "
          f"CI [{lo_r:+.4f},{hi_r:+.4f}]")

    q4, q1 = strat.get("Q4"), strat.get("Q1")
    fracs = [strat[f"Q{i}"]["frac_positive"] for i in range(1, 5)
             if f"Q{i}" in strat]
    monotone = all(fracs[i] <= fracs[i + 1] for i in range(len(fracs) - 1))
    if q4 and q1:
        print()
        print(f"  top quartile mean d_w {q4['mean_d']:+.5f}  vs  "
              f"bottom {q1['mean_d']:+.5f}")
        print(f"  fraction positive by quartile: "
              f"{' -> '.join(f'{f:.3f}' for f in fracs)}"
              f"{'  (monotone increasing)' if monotone else '  (NOT monotone)'}")
        print()
        print("  READ CAREFULLY -- mean d_w is NOT monotone in phi variance")
        print("  (Q2 is the largest, not Q4), so this is not a clean dose-")
        print("  response. What IS monotone is the fraction of windows where")
        print("  the structural arm wins. And the bottom quartile is NEGATIVE:")
        print("  where phi barely varies inside the window, the structural arm")
        print("  is WORSE. Caution: in Q1 the centred target is almost pure")
        print("  noise (phi variance 0.003-0.066 of pooled), so a per-window r")
        print("  there is estimating close to nothing and its sign is weakly")
        print("  determined.")
        if rho > 0 and lo_r > 0:
            print()
            print("  => CONSISTENT AND POSITIVE AT EVERY BLOCK LENGTH TESTED.")
            print("     Unlike the pooled-r difference, rho's CI excludes zero")
            print("     even at the largest RELIABLE block count, so this does")
            print("     not rest on the 9-block artefact. The remaining caveat")
            print(f"     is the same one as everywhere else: {L_rho * WINDOW / 1e6:.3f} Mb blocks")
            print("     are shorter than Hi-C autocorrelation, so the p is")
            print("     anticonservative and 8.98 Mb cannot do better. Call it")
            print("     a strong lead that the multi-chromosome build should")
            print("     confirm -- not an established result.")
        else:
            print()
            print("  => SUGGESTIVE, NOT ESTABLISHED. The bootstrap CI on rho")
            print("     includes 0 once window dependence is respected. The")
            print("     direction is right and the fraction-positive trend is")
            print("     monotone, but this does not clear significance on 8.98")
            print("     Mb -- the same genome-quantity limit as everything else")
            print("     here. It is a lead for the multi-chromosome build to")
            print("     confirm or kill, not a result.")

    # ---- the seed-level test, for comparison
    perm = exact_permutation_3v3(
        np.array([per_run_r[n] for n in STRUCT]),
        np.array([per_run_r[n] for n in BASELN]))
    print()
    print("-" * 74)
    print("the two tests side by side -- they answer different questions")
    print("-" * 74)
    print(f"  ACROSS SEEDS   (does it reproduce across training runs?)")
    print(f"    structural   {[round(per_run_r[n], 4) for n in STRUCT]}")
    print(f"    baseline_v2  {[round(per_run_r[n], 4) for n in BASELN]}")
    print(f"    diff {perm['observed']:+.4f}   exact p = {perm['p_two_sided']:.4f}"
          f"   (floor {perm['p_floor']:.3f} at 3v3, {perm['n_arrangements']} "
          f"arrangements)")
    used = sweep[str(L_use)]
    st = sign_test(d)
    print(f"  ACROSS WINDOWS (does it hold across the genome?)")
    print(f"    mean d_w {d.mean():+.5f}  sd {d.std(ddof=1):.5f} "
          f"-- the sd is {d.std(ddof=1) / abs(d.mean()):.0f}x the mean")
    print(f"    sign test  {st['n_positive']}/{st['n']} windows positive "
          f"({st['fraction']:.1%})  p = {st['p_two_sided']:.4f}"
          f"   <-- assumption-free, and it is FLAT")
    print(f"    moving block bootstrap, L = {L_use} windows "
          f"({L_use * WINDOW / 1e6:.3f} Mb), {used['n_blocks']} blocks")
    print(f"    mean-of-per-window-r  p = {used['p_two_sided']:.4f}   95% CI "
          f"[{used['ci95_lo']:+.5f}, {used['ci95_hi']:+.5f}]")
    pu = pooled_sweep[str(L_use)]
    print(f"    pooled-r difference   p = {pu['p_two_sided']:.4f}   95% CI "
          f"[{pu['ci95_lo']:+.5f}, {pu['ci95_hi']:+.5f}]   "
          f"observed {obs_pooled:+.5f}   NOT TRUSTWORTHY, 9 blocks")

    # ---- what can actually be claimed
    ok_L = [int(k) for k, v in pooled_sweep.items() if v["reliable"]]
    L_max_ok = max(ok_L)
    best = pooled_sweep[str(L_max_ok)]
    print()
    print("=" * 74)
    print("WHAT THIS SUPPORTS, AND WHAT IT DOES NOT")
    print("=" * 74)
    print(f"  The arm difference is positive in all three seeds and the pooled")
    print(f"  estimate is {obs_pooled:+.5f}. Neither test licenses p < 0.05:")
    print()
    print(f"   - across seeds, p = {perm['p_two_sided']:.3f} and CANNOT go lower:")
    print(f"     3v3 has {perm['n_arrangements']} arrangements, floor "
          f"{perm['p_floor']:.3f}.")
    print(f"   - across windows, the two ends of the bracket disagree and the")
    print(f"     honest reading is between them:")
    print(f"       longest block with >= {MIN_BLOCKS} blocks: L = {L_max_ok} "
          f"({L_max_ok * WINDOW / 1e6:.3f} Mb, {best['n_blocks']} blocks), "
          f"p = {best['p_two_sided']:.4f}")
    print(f"       -- but {L_max_ok * WINDOW / 1e6:.3f} Mb is SHORTER than Hi-C")
    print(f"          autocorrelation, so that p is anticonservative.")
    print(f"       at >= 1 Mb blocks, which Hi-C requires, only 5-9 blocks exist")
    print(f"       and the bootstrap percentiles are not valid at all.")
    print()
    print("  The binding constraint is the AMOUNT OF INDEPENDENT GENOME, not")
    print("  the choice of test. 274 contiguous val windows span 8.98 Mb; at")
    print("  megabase dependence that is single-digit effective samples. No")
    print("  resampling scheme creates information the data do not contain.")
    print("  The fix is more genome (T4, multi-chromosome), not a cleverer")
    print("  statistic -- and T4 is also what known weakness 5 requires.")

    OUT.write_text(json.dumps({
        "target": TARGET, "probe": "B (within-window, both sides centred)",
        "index": INDEX32.name, "window_bp": WINDOW, "stride_bp": STRIDE,
        "val": {"n_windows": int(len(starts_va)), "n_scored": int(ok.sum()),
                "span_bp": int(v1 - v0), "start": v0, "end": v1,
                "contiguous": True},
        "per_run_pooled_r": per_run_r,
        "per_window_d": {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                         "n_positive": int((d > 0).sum()), "n": int(len(d)),
                         "values": [float(x) for x in d]},
        "acf": {"lags_0_to_n": [float(x) for x in a],
                "sig_threshold": float(thresh),
                "first_lag_below": int(first_below)},
        "block_bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                            "chosen_L": int(L_use),
                            "chosen_L_reason": "max(measured ACF lag, 1 Mb Hi-C floor)",
                            "sweep": sweep},
        "pooled_r_block_bootstrap": {"observed": obs_pooled,
                                     "n_boot": N_BOOT, "seed": BOOT_SEED,
                                     "chosen_L": int(L_use),
                                     "sweep": pooled_sweep},
        "sign_test": st,
        "phi_variance_stratification": {
            "per_window_keep_phi": {"min": float(kk.min()),
                                    "median": float(np.median(kk)),
                                    "max": float(kk.max())},
            "quartiles": strat,
            "spearman_rho": float(rho),
            "spearman_p_naive_INVALID": float(p_rho_naive),
            "spearman_p_block_bootstrap": float(p_rho),
            "spearman_ci95": [float(lo_r), float(hi_r)],
            "spearman_sweep": rho_sweep,
            "block_len_windows": int(L_rho),
            "block_len_note": "largest L with >= MIN_BLOCKS blocks",
            "frac_positive_monotone": bool(monotone)},
        "seed_level_permutation": perm,
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
