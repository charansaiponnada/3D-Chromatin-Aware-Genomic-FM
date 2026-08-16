#!/usr/bin/env python3
"""D1-D3 -- mechanistic diagnostics on the trained structural checkpoints.

    ./3d-gen/bin/python scripts/phase4_d1_diagnostic.py [structural_seed0 ...]

architecture_spec.md 4.1.3 specifies four cheap diagnostics that answer "is the
structural pathway alive?" from INSIDE the model, independently of P1's
input-swap evidence. The spec's own words (4.1.3): "D1 < 0.05 and D_S1 ~ 0
should agree, and disagreement between them is itself informative."

This script runs the three that read off a finished checkpoint:

  D1  Var_t(W_dstruct s_t) / Var_t(dt_proj delta'_t), per layer per direction.
      < 0.05  =>  pathway inert (failure mode F1: the structural term is
      effectively constant across positions and is absorbed into b_dt, leaving
      a re-parameterised baseline).

      Both quantities are (b, l, d_inner). "Var_t" is taken per channel and then
      averaged over channels, i.e. total variance / d_inner. Two decompositions
      are reported because they answer different objections:

        pooled  variance over every position in the split. This is Var_t as the
                spec writes it, and it is the number the 0.05 threshold applies
                to.
        within  variance over positions inside a window, averaged over windows.
                A structural term that is constant WITHIN a window but varies
                BETWEEN windows is still not absorbable into a bias, so a large
                pooled/within gap says the mechanism operates at window scale
                rather than at nucleotide scale. That distinction matters for
                reading the result and it is invisible in the pooled number.

  D2  histogram of the permeability gate p_t = softplus(w_gate . s_t + b_gate).
      Mass concentrated at ~0 => gate unused (F5). b_gate initialises at -4, so
      p_t starts at softplus(-4) = 0.0181; anything at that value has not moved.

  D3  ||W_dstruct||_F per layer. The spec asks for a TRAJECTORY (flat near zero
      => no gradient pressure, F6). Only the final checkpoint is retained per
      run -- train.py overwrites checkpoint.pt -- so what is reported here is
      the endpoint, against the exact zero it was initialised to. A non-zero
      endpoint proves gradient pressure existed; it cannot show when.

D4 (probe R^2 predicting s_t from the BASELINE model's hidden states) is not run
here. It needs a fitted probe on a second model and is a different kind of
experiment; it is the F3 test, not the F1 test.

Inference only. No training, no checkpoint is written.
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
from train import collate, mask_tokens                       # noqa: E402

NOVEL = REPO / "results" / "novel_model"
OUT = NOVEL / "d1_diagnostic.json"
EVAL_SEED = 1234
D1_INERT = 0.05          # architecture_spec.md 4.1.3
P_INIT = float(torch.nn.functional.softplus(torch.tensor(-4.0)))


class Accum:
    """Per-channel sum / sumsq in float64, plus a within-window variance mean."""

    def __init__(self):
        self.n = 0
        self.s = None
        self.ss = None
        self.within_sum = 0.0     # sum over windows of mean-over-channels var_t
        self.within_n = 0

    def add(self, x: torch.Tensor):
        # x: (b, l, d)
        f = x.detach().double()
        b, l, d = f.shape
        flat = f.reshape(-1, d)
        if self.s is None:
            self.s = flat.sum(0)
            self.ss = (flat * flat).sum(0)
        else:
            self.s += flat.sum(0)
            self.ss += (flat * flat).sum(0)
        self.n += flat.shape[0]
        if l > 1:
            self.within_sum += float(f.var(dim=1, unbiased=True).mean(-1).sum())
            self.within_n += b

    def pooled_var(self) -> float:
        if self.n < 2:
            return float("nan")
        mean = self.s / self.n
        var = self.ss / self.n - mean * mean
        return float(var.clamp_min(0).mean())

    def within_var(self) -> float:
        if self.within_n == 0:
            return float("nan")
        return self.within_sum / self.within_n


def load_model(run_dir: Path, device):
    import yaml
    doc = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    m = doc["model"]
    cfg = ModelConfig(
        d_model=m["d_model"], n_layer=m["n_layer"], d_state=m["d_state"],
        d_conv=m["d_conv"], expand=m["expand"], vocab_size=m["vocab_size"],
        structural=m["structural"], d_struct=m["d_struct"],
        d_struct_raw=m["d_struct_raw"], d_struct_hidden=m["d_struct_hidden"],
        use_permeability=m["use_permeability"],
        dt_min=m["dt_min"], dt_max=m["dt_max"], dt_floor=m["dt_floor"],
    )
    if not cfg.structural:
        raise ValueError(f"{run_dir.name} is not a structural run")
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
    model = BiMambaLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, int(ck["step"]), doc


@torch.no_grad()
def diagnose(model, cfg, ds, device, symmetry, max_windows=None):
    """Hook every direction's dt_proj / W_dstruct / w_gate and accumulate."""
    acc = {}
    hooks = []
    for li, layer in enumerate(model.layers):
        for dname in ("fwd", "rev"):
            d = getattr(layer, dname)
            key = f"L{li:02d}.{dname}"
            acc[key] = {"dt": Accum(), "struct": Accum()}

            def mk(k, slot):
                def h(_mod, _inp, out):
                    acc[k][slot].add(out)
                return h
            hooks.append(d.dt_proj.register_forward_hook(mk(key, "dt")))
            hooks.append(d.W_dstruct.register_forward_hook(mk(key, "struct")))

    # D2: permeability gate, captured on the first layer's forward direction
    # (w_gate is per-direction; the histogram is over the whole model)
    p_hist = torch.zeros(64, dtype=torch.float64)
    p_edges = torch.linspace(0.0, 2.0, 65)
    p_stats = {"n": 0, "sum": 0.0, "min": float("inf"), "max": -float("inf"),
               "n_at_init": 0}

    if cfg.use_permeability:
        def gate_hook(_mod, _inp, out):
            p = torch.nn.functional.softplus(out.squeeze(-1)).detach().float().cpu()
            f = p.reshape(-1)
            p_hist.add_(torch.histc(f.clamp(0, 2.0), bins=64, min=0.0,
                                    max=2.0).double())
            p_stats["n"] += f.numel()
            p_stats["sum"] += float(f.sum())
            p_stats["min"] = min(p_stats["min"], float(f.min()))
            p_stats["max"] = max(p_stats["max"], float(f.max()))
            p_stats["n_at_init"] += int((f - P_INIT).abs().lt(1e-6).sum())
        for layer in model.layers:
            for dname in ("fwd", "rev"):
                hooks.append(getattr(layer, dname).w_gate
                             .register_forward_hook(gate_hook))

    gen = torch.Generator(device=device).manual_seed(EVAL_SEED)
    n = len(ds) if max_windows is None else min(len(ds), max_windows)
    for b0 in range(0, n, 2):
        items = [ds[i] for i in range(b0, min(b0 + 2, n))]
        batch = collate(items)
        tokens = batch["tokens"].to(device)
        inputs, _ = mask_tokens(tokens, 0.15, gen, 0.8, 0.1)
        model(inputs, phi=batch["phi"].to(device),
              phi_valid=batch["phi_valid"].to(device), symmetry=symmetry)

    for h in hooks:
        h.remove()
    return acc, p_hist, p_edges, p_stats, n


def main() -> int:
    t0 = time.time()
    if not torch.cuda.is_available():
        print("No CUDA device.")
        return 2
    device = torch.device("cuda", 0)
    use_scan("triton")

    names = sys.argv[1:] or sorted(
        d.name for d in NOVEL.glob("structural_seed*") if d.is_dir())
    runs = []
    for nm in names:
        d = NOVEL / nm
        if not (d / "checkpoint.pt").exists() or not (d / "run_config.yaml").exists():
            print(f"skip {nm}: no checkpoint")
            continue
        if "status: COMPLETED" not in (d / "run_config.yaml").read_text():
            print(f"skip {nm}: not COMPLETED")
            continue
        runs.append(d)
    if not runs:
        print("No COMPLETED structural runs.")
        return 1

    ds = WindowDataset("val", structural=True, rc_augment=False, seed=0,
                       phi_control="none")
    symmetry = torch.from_numpy(
        np.asarray(ds.symmetry, dtype=np.float32)).to(device)

    print("=" * 78)
    print("D1-D3 MECHANISTIC DIAGNOSTICS -- structural arm, trained checkpoints")
    print("=" * 78)
    print(f"real phi, val split, {len(ds)} windows, masking seed {EVAL_SEED}")
    print(f"D1 inert threshold (architecture_spec 4.1.3): {D1_INERT}")
    print()

    out = {"threshold_d1_inert": D1_INERT, "eval_seed": EVAL_SEED,
           "n_val_windows": len(ds), "runs": {}}

    for run_dir in runs:
        model, cfg, step, doc = load_model(run_dir, device)
        acc, p_hist, p_edges, p_stats, nwin = diagnose(
            model, cfg, ds, device, symmetry)

        print(f"--- {run_dir.name}  (step {step}) ---")
        print()
        print("D1  ratio Var_t(W_dstruct s) / Var_t(dt_proj delta')")
        print("     layer      pooled       within      ||W_dstruct||_F")
        rows, pooled_all, within_all = [], [], []
        for key in sorted(acc):
            a = acc[key]
            vd_p, vs_p = a["dt"].pooled_var(), a["struct"].pooled_var()
            vd_w, vs_w = a["dt"].within_var(), a["struct"].within_var()
            rp = vs_p / vd_p if vd_p > 0 else float("nan")
            rw = vs_w / vd_w if vd_w > 0 else float("nan")
            li, dname = key.split(".")
            W = getattr(model.layers[int(li[1:])], dname).W_dstruct.weight
            fro = float(W.detach().norm())
            rows.append({"layer": key, "d1_pooled": rp, "d1_within": rw,
                         "var_dt_pooled": vd_p, "var_struct_pooled": vs_p,
                         "var_dt_within": vd_w, "var_struct_within": vs_w,
                         "w_dstruct_fro": fro})
            pooled_all.append(rp)
            within_all.append(rw)
            print(f"     {key:<10} {rp:10.4f}   {rw:10.4f}      {fro:10.4f}")

        med_p = float(np.median(pooled_all))
        med_w = float(np.median(within_all))
        n_inert = int(sum(1 for r in pooled_all if r < D1_INERT))
        print()
        print(f"     median pooled D1  {med_p:.4f}    "
              f"median within-window D1  {med_w:.4f}")
        print(f"     directions below the {D1_INERT} inert threshold (pooled): "
              f"{n_inert}/{len(pooled_all)}")
        print(f"     VERDICT: {'INERT' if med_p < D1_INERT else 'LIVE'} "
              f"by the pre-registered D1 rule")
        print()

        pd2 = None
        if cfg.use_permeability and p_stats["n"]:
            mean_p = p_stats["sum"] / p_stats["n"]
            frac_init = p_stats["n_at_init"] / p_stats["n"]
            hi = p_hist / max(float(p_hist.sum()), 1.0)
            print("D2  permeability gate p_t = softplus(w_gate . s + b_gate)")
            print(f"     p_t init value (b_gate=-4)  {P_INIT:.6f}")
            print(f"     mean {mean_p:.6f}   min {p_stats['min']:.6f}   "
                  f"max {p_stats['max']:.6f}   n {p_stats['n']:,}")
            print(f"     fraction still exactly at init  {frac_init:.6f}")
            top = torch.topk(hi, 5)
            print("     five heaviest bins (p range -> mass):")
            for v, i in zip(top.values.tolist(), top.indices.tolist()):
                print(f"       [{p_edges[i]:.4f}, {p_edges[i+1]:.4f})  {v:.4f}")
            pd2 = {"p_init": P_INIT, "mean": mean_p, "min": p_stats["min"],
                   "max": p_stats["max"], "n": p_stats["n"],
                   "frac_at_init": frac_init,
                   "hist_edges": p_edges.tolist(),
                   "hist_mass": (p_hist / max(float(p_hist.sum()), 1.0)).tolist()}
            print()

        fros = [r["w_dstruct_fro"] for r in rows]
        print("D3  ||W_dstruct||_F endpoint vs the exact 0.0 it was initialised to")
        print(f"     min {min(fros):.4f}   median {float(np.median(fros)):.4f}   "
              f"max {max(fros):.4f}   directions at exactly 0: "
              f"{sum(1 for f in fros if f == 0.0)}/{len(fros)}")
        print("     Endpoint only -- checkpoint.pt is overwritten each save, so")
        print("     the trajectory the spec asks for is not recoverable.")
        print()

        out["runs"][run_dir.name] = {
            "step": step, "layers": rows,
            "d1_median_pooled": med_p, "d1_median_within": med_w,
            "d1_n_below_threshold": n_inert, "d1_n_directions": len(pooled_all),
            "d1_verdict": "INERT" if med_p < D1_INERT else "LIVE",
            "d2": pd2,
            "d3_w_dstruct_fro": {"min": min(fros), "max": max(fros),
                                 "median": float(np.median(fros))},
        }
        del model
        torch.cuda.empty_cache()

    print("=" * 78)
    print("ACROSS SEEDS")
    print("=" * 78)
    meds = [v["d1_median_pooled"] for v in out["runs"].values()]
    print(f"  median pooled D1 per seed: "
          f"{', '.join(f'{m:.4f}' for m in meds)}")
    print(f"  all seeds {'LIVE' if min(meds) >= D1_INERT else 'MIXED/INERT'} "
          f"against the {D1_INERT} threshold")
    print()
    print("  D1 measures whether the structural term MOVES delta across")
    print("  positions. It does not measure whether that movement helps. A LIVE")
    print("  D1 with a null loss delta is exactly the 'used but not expressible")
    print("  in MLM' cell of the 4.1.3 gate, not evidence of benefit.")
    print()
    print(f"  elapsed {time.time()-t0:.1f} s")

    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
