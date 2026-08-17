#!/usr/bin/env python
"""P5-PROBE -- can the pretrained representation predict 3D organisation from
sequence alone, and does structure-conditioned pretraining help?

THE QUESTION THIS ASKS, AND WHY IT IS NOT THE PHASE 4 QUESTION
--------------------------------------------------------------
Phase 4 asked whether conditioning on phi lowers masked-LM bits/nt. It does not
(+0.0020 bits, +0.80sigma, wrong sign). But nucleotide prediction inside a
32,768 bp window is dominated by local k-mer statistics, and TAD-scale
organisation lives at 100 kb - 1 Mb. There is little reason for structure to
carry marginal information about the MLM objective at this window size, so that
null is weak evidence about the representation.

This script asks the transfer question instead: does the pretrained
representation encode 3D organisation, and does the structural arm encode MORE
of it? Pretraining loss and downstream transfer routinely dissociate; a flat
loss with a transfer gap is a real and reportable result, and so is a flat loss
with no transfer gap.

phi IS WITHHELD AT PROBE TIME. Both arms are run with phi = zeros, which is
exactly the pre-registered S0 control (architecture_spec.md 4.1.3; measured
Delta = +0.0000 bits, KL = 2.5e-05, so the structural model at S0 is
behaviourally the baseline). If the structural arm were fed real phi and then
probed for insulation, it would be copying its own input and every number here
would be circular. It is not fed phi. The targets are phi; the inputs are
sequence.

So a structural-arm win here means: conditioning on structure DURING pretraining
taught the model to infer structure FROM SEQUENCE. That is the claim the project
is actually about, and it is a claim the baseline can win.

TARGETS (window-level, from the same standardised phi used in training)
    insulation_100kb    ch 0   TAD boundary strength, r = 0.9969 vs 4DN
    compartment_pc1     ch 7   A/B compartment, r = 0.9759 vs 4DN
    directionality_2Mb  ch 3   antisymmetric, the hardest of the three
Each is the mean over the window's valid positions. Windows with < 50% valid
phi are already excluded by the dataset builder.

REFERENCE FLOOR. A ridge on GC content + dinucleotide frequencies is fitted on
the identical split. Neither arm's representation is interesting unless it beats
that floor -- compartment_pc1 in particular is known to track GC, and a
representation that merely recovers GC has learned nothing about 3D.

PROTOCOL
    fit    ridge on the TRAIN split representations (subsampled, --n-train)
    select alpha by generalised cross-validation within train only
    report Pearson r and R^2 on the VAL split, never touched during fitting
    3 structural seeds vs 3 baseline_v2 seeds, same windows, same order

Inference only. No training. No checkpoint is written. Reads:
    results/novel_model/structural_seed*/checkpoint.pt
    results/baselines/baseline_v2_seed*/checkpoint.pt
Writes:
    results/novel_model/p5_structure_probe.json
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

from chromfm.model import BiMambaLM, ModelConfig, use_scan   # noqa: E402
from phase1_dataset import WindowDataset                     # noqa: E402
from train import collate                                    # noqa: E402

NOVEL = REPO / "results" / "novel_model"
BASE = REPO / "results" / "baselines"
OUT = NOVEL / "p5_structure_probe.json"

# The JupyterHub idle culler kills the whole user cgroup after 10 minutes with
# no browser activity (CLAUDE.md 3). A full pass over six checkpoints takes
# longer than that, so representations are cached to disk per (run, split) and
# a relaunch skips whatever already landed. A kill then costs one model, not
# the run. The cache key includes n_windows because a different subsample is a
# different matrix -- reusing it silently would compare models on unlike rows.
CACHE = NOVEL / "p5_cache"

# phi channel index -> reporting name. Indices are FEATURE_NAMES order in
# scripts/phase1_features.py; asserted against the dataset at run time.
TARGETS = {
    "insulation_100kb": 0,
    "directionality_2Mb": 3,
    "compartment_pc1": 7,
}
EXPECTED_NAMES = [
    "insulation_100kb", "insulation_250kb", "insulation_500kb",
    "directionality_2Mb", "log_contact_density", "upstream_mass_frac",
    "short_long_ratio", "compartment_pc1",
]

ALPHAS = np.logspace(-3, 6, 40)
BATCH = 2
PROBE_SEED = 20260817          # fixed, not a training seed


# ----------------------------------------------------------------- checkpoints

def load_any(run_dir: Path, device):
    """Load either arm. Unlike phase4_d1_diagnostic.load_model this does not
    require structural=True -- the whole point is to run both arms."""
    import yaml
    doc = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    m = doc["model"]
    cfg = ModelConfig(
        d_model=m["d_model"], n_layer=m["n_layer"], d_state=m["d_state"],
        d_conv=m["d_conv"], expand=m["expand"], vocab_size=m["vocab_size"],
        # The baseline run_config.yaml omits the structural keys entirely --
        # those modules are never built when structural=False -- so these fall
        # back to the ModelConfig defaults rather than KeyError.
        structural=m["structural"],
        d_struct=m.get("d_struct", ModelConfig.d_struct),
        d_struct_raw=m.get("d_struct_raw", ModelConfig.d_struct_raw),
        d_struct_hidden=m.get("d_struct_hidden", ModelConfig.d_struct_hidden),
        use_permeability=m.get("use_permeability", False),
        dt_min=m["dt_min"], dt_max=m["dt_max"], dt_floor=m["dt_floor"],
    )
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
    model = BiMambaLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, int(ck["step"])


# ----------------------------------------------------------- representations

@torch.no_grad()
def represent(model, cfg, ds, idx, device, tag=""):
    """Mean-pooled final hidden state, phi withheld (S0 = zeros).

    The representation is norm_f(u), i.e. the input to lm_head -- the last point
    at which the model holds a per-position representation rather than logits.
    Captured by hook so this does not depend on forward() returning it.
    """
    grab = {}

    def hook(_mod, _inp, out):
        grab["h"] = out.detach()

    h = model.norm_f.register_forward_hook(hook)
    feats = []
    t0 = time.time()
    try:
        for b0 in range(0, len(idx), BATCH):
            if b0 and b0 % 100 == 0:
                el = time.time() - t0
                rate = b0 / el
                print(f"    {tag} {b0}/{len(idx)} windows, {el:.0f}s elapsed, "
                      f"eta {(len(idx) - b0) / max(rate, 1e-9):.0f}s", flush=True)
            items = [ds[int(i)] for i in idx[b0:b0 + BATCH]]
            batch = collate(items)
            tokens = batch["tokens"].to(device)
            kw = {}
            if cfg.structural:
                # S0: structure removed. Zeros are the standardised mean, and
                # phi_valid = False additionally zeroes s inside forward().
                b, l = tokens.shape
                kw = {
                    "phi": torch.zeros(b, l, cfg.d_struct_raw,
                                       dtype=torch.float32, device=device),
                    "phi_valid": torch.zeros(b, l, dtype=torch.bool,
                                             device=device),
                    "symmetry": None,
                }
            model(tokens, **kw)
            feats.append(grab["h"].mean(dim=1).float().cpu())
    finally:
        h.remove()
    return torch.cat(feats).numpy()


def seq_features(ds, idx):
    """GC content + 16 dinucleotide frequencies. The floor both arms must beat.

    Computed from tokens, so it is exactly the information a model could get
    from local composition alone.
    """
    A, C, G, T = 2, 3, 4, 5
    rows = []
    for i in idx:
        tok = ds[int(i)]["tokens"].astype(np.int64)
        acgt = tok[(tok >= A) & (tok <= T)]
        n = max(len(acgt), 1)
        gc = float(((acgt == C) | (acgt == G)).sum()) / n
        d = acgt[:-1] * 4 + acgt[1:] - (A * 4 + A)
        d = d[(d >= 0) & (d < 16)]
        di = np.bincount(d, minlength=16).astype(np.float64)
        di /= max(di.sum(), 1.0)
        rows.append(np.concatenate([[gc], di]))
    return np.asarray(rows, dtype=np.float64)


def targets(ds, idx):
    """Window-mean of each target phi channel over valid positions."""
    out = {k: [] for k in TARGETS}
    for i in idx:
        item = ds[int(i)]
        phi, valid = item["phi"], item["phi_valid"].astype(bool)
        if valid.sum() == 0:
            for k in TARGETS:
                out[k].append(np.nan)
            continue
        for k, ch in TARGETS.items():
            out[k].append(float(phi[valid, ch].mean()))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ------------------------------------------------------------------- the ridge

def ridge_fit_eval(Xtr, ytr, Xva, yva):
    """Ridge with alpha chosen by leave-one-out GCV inside train only.

    Closed form via SVD: for each alpha, LOO residual is
    r_i / (1 - h_ii) with h from the hat matrix, so all 40 alphas cost one SVD.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
    ym = ytr.mean()
    yc = ytr - ym

    U, S, _ = np.linalg.svd(Ztr, full_matrices=False)
    UT_y = U.T @ yc

    best = (np.inf, None)
    for a in ALPHAS:
        d = S ** 2 / (S ** 2 + a)
        yhat = U @ (d * UT_y)
        hii = (U ** 2 * d).sum(1)
        denom = np.clip(1.0 - hii, 1e-6, None)
        loo = float(np.mean(((yc - yhat) / denom) ** 2))
        if loo < best[0]:
            best = (loo, a)
    alpha = best[1]

    # refit at the chosen alpha, then predict val
    G = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1])
    w = np.linalg.solve(G, Ztr.T @ yc)
    pred = Zva @ w + ym

    ss_res = float(((yva - pred) ** 2).sum())
    ss_tot = float(((yva - yva.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if pred.std() < 1e-12 or yva.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(pred, yva)[0, 1])
    return {"r": r, "r2": r2, "alpha": float(alpha), "loo_mse": float(best[0])}


# ------------------------------------------------------------------------ main

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600,
                    help="train windows to fit the probe on (0 = all)")
    ap.add_argument("--n-val", type=int, default=0,
                    help="val windows to evaluate on (0 = all)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device.")
        return 2
    device = torch.device("cuda", 0)
    use_scan("triton")

    # Datasets are built structural=True so phi is available AS A TARGET. It is
    # never passed to the model -- represent() substitutes zeros.
    ds_tr = WindowDataset("train", structural=True, rc_augment=False)
    ds_va = WindowDataset("val", structural=True, rc_augment=False)

    z = np.load(REPO / "data" / "processed" / "dataset_index.npz",
                allow_pickle=True)
    names = [str(x) for x in z["feature_names"]]
    if names != EXPECTED_NAMES:
        print(f"phi channel order changed: {names}")
        return 2
    for k, ch in TARGETS.items():
        assert names[ch] == k, f"channel {ch} is {names[ch]}, expected {k}"

    rng = np.random.default_rng(PROBE_SEED)
    idx_tr = np.arange(len(ds_tr))
    if args.n_train and args.n_train < len(idx_tr):
        idx_tr = np.sort(rng.choice(idx_tr, args.n_train, replace=False))
    idx_va = np.arange(len(ds_va))
    if args.n_val and args.n_val < len(idx_va):
        idx_va = np.sort(rng.choice(idx_va, args.n_val, replace=False))

    print("=" * 74)
    print("P5-PROBE -- 3D organisation decoded from sequence-only representation")
    print("=" * 74)
    print(f"  train windows {len(idx_tr):,} of {len(ds_tr):,}")
    print(f"  val   windows {len(idx_va):,} of {len(ds_va):,}")
    print("  phi WITHHELD from the model at probe time (S0, zeros)")
    print()

    y_tr, y_va = targets(ds_tr, idx_tr), targets(ds_va, idx_va)
    for k in TARGETS:
        ok_tr = np.isfinite(y_tr[k]).sum()
        ok_va = np.isfinite(y_va[k]).sum()
        print(f"  target {k:20s} train {ok_tr:,} val {ok_va:,} "
              f"sd(val) {np.nanstd(y_va[k]):.4f}")
    print()

    results = {"targets": {}, "config": {
        "n_train": int(len(idx_tr)), "n_val": int(len(idx_va)),
        "probe_seed": PROBE_SEED, "phi_at_probe": "S0 zeros (withheld)",
        "representation": "mean-pooled norm_f(u), d_model=256",
    }}

    # ---- reference floor: composition only
    print("-" * 74)
    print("REFERENCE FLOOR -- GC + dinucleotide composition (no model)")
    Xtr_s, Xva_s = seq_features(ds_tr, idx_tr), seq_features(ds_va, idx_va)
    for k in TARGETS:
        m = np.isfinite(y_tr[k])
        mv = np.isfinite(y_va[k])
        res = ridge_fit_eval(Xtr_s[m], y_tr[k][m], Xva_s[mv], y_va[k][mv])
        results["targets"].setdefault(k, {})["composition_floor"] = res
        print(f"  {k:20s} r = {res['r']:+.4f}   R2 = {res['r2']:+.4f}")
    print()

    # ---- both arms
    runs = []
    for d in sorted(NOVEL.glob("structural_seed*")):
        if (d / "checkpoint.pt").exists():
            runs.append(("structural", d))
    for d in sorted(BASE.glob("baseline_v2_seed*")):
        if (d / "checkpoint.pt").exists():
            runs.append(("baseline_v2", d))
    if not runs:
        print("no checkpoints found")
        return 2

    CACHE.mkdir(parents=True, exist_ok=True)
    per_run = {}
    for arm, d in runs:
        t0 = time.time()
        print("-" * 74, flush=True)
        ktr = CACHE / f"{d.name}_train_{len(idx_tr)}.npy"
        kva = CACHE / f"{d.name}_val_{len(idx_va)}.npy"
        kstep = CACHE / f"{d.name}_step.txt"
        if ktr.exists() and kva.exists():
            Xtr, Xva = np.load(ktr), np.load(kva)
            step = int(kstep.read_text()) if kstep.exists() else -1
            print(f"{d.name}  [{arm}]  step {step:,}  cached", flush=True)
        else:
            model, cfg, step = load_any(d, device)
            kstep.write_text(str(step))
            print(f"{d.name}  [{arm}]  step {step:,}  "
                  f"structural={cfg.structural}", flush=True)
            Xtr = represent(model, cfg, ds_tr, idx_tr, device, "train")
            np.save(ktr, Xtr)
            Xva = represent(model, cfg, ds_va, idx_va, device, "val")
            np.save(kva, Xva)
            del model
            torch.cuda.empty_cache()

        row = {"arm": arm, "step": step, "targets": {}}
        for k in TARGETS:
            m = np.isfinite(y_tr[k])
            mv = np.isfinite(y_va[k])
            res = ridge_fit_eval(Xtr[m], y_tr[k][m], Xva[mv], y_va[k][mv])
            row["targets"][k] = res
            print(f"  {k:20s} r = {res['r']:+.4f}   R2 = {res['r2']:+.4f}")
        row["seconds"] = round(time.time() - t0, 1)
        per_run[d.name] = row
        print(f"  ({row['seconds']:.0f} s)")

    results["runs"] = per_run

    # ---- arm comparison
    print()
    print("=" * 74)
    print("ARM COMPARISON -- val Pearson r, mean +/- sd over seeds")
    print("=" * 74)
    for k in TARGETS:
        st = [v["targets"][k]["r"] for v in per_run.values()
              if v["arm"] == "structural"]
        bl = [v["targets"][k]["r"] for v in per_run.values()
              if v["arm"] == "baseline_v2"]
        floor = results["targets"][k]["composition_floor"]["r"]
        summ = {
            "structural_r": st, "baseline_v2_r": bl,
            "structural_mean": float(np.mean(st)) if st else None,
            "baseline_v2_mean": float(np.mean(bl)) if bl else None,
            "structural_sd": float(np.std(st, ddof=1)) if len(st) > 1 else None,
            "baseline_v2_sd": float(np.std(bl, ddof=1)) if len(bl) > 1 else None,
            "delta_r": (float(np.mean(st) - np.mean(bl))
                        if st and bl else None),
            "composition_floor_r": floor,
        }
        results["targets"][k].update(summ)
        print(f"\n  {k}")
        print(f"    composition floor   r = {floor:+.4f}")
        if st:
            print(f"    structural   {np.mean(st):+.4f} "
                  f"+/- {np.std(st, ddof=1) if len(st) > 1 else 0:.4f}  "
                  f"{['%+.4f' % x for x in st]}")
        if bl:
            print(f"    baseline_v2  {np.mean(bl):+.4f} "
                  f"+/- {np.std(bl, ddof=1) if len(bl) > 1 else 0:.4f}  "
                  f"{['%+.4f' % x for x in bl]}")
        if st and bl:
            print(f"    delta        {np.mean(st) - np.mean(bl):+.4f}")

    print()
    print("READ THIS BEFORE QUOTING ANY NUMBER ABOVE:")
    print("  * A delta is not a result at n=3. Exact permutation p has a")
    print("    minimum of 0.1000 at 3v3 -- p<0.05 is unreachable by design.")
    print("    Treat this run as a SIGNAL CHECK that decides whether the")
    print("    5-seed paired version is worth the compute.")
    print("  * An arm that does not beat the composition floor has not")
    print("    demonstrated anything about 3D organisation.")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
