#!/usr/bin/env python
"""D4 -- is the structural ENCODER the bottleneck, or is W_dstruct nulling a
healthy signal?

WHY THIS EXISTS
---------------
D1 found the structural pathway inert in layers 0-11 and live only in 12-15.
D3 found ||W_dstruct||_F non-zero in 32/32 directions in every seed, against
the exact 0 it was initialised to -- so gradient DID reach the dead layers.
Those two facts are in tension: the dead layers learned something, and what
they produce is still near-constant across positions.

There is exactly one reconciliation that costs no training. d_struct = 2 is a
hard 8 -> 2 bottleneck. If the encoder output s is itself near-constant WITHIN
a window, then W_dstruct @ s is near-constant no matter what W_dstruct learned.
The deadness would then be the encoder's fault, not the layer's.

    Var_t(s) collapsed   => encoder is the bottleneck
                            => the d_struct=8 no-encoder ablation is justified
    Var_t(s) healthy     => early layers actively null a varying signal
                            => dropping the encoder will NOT fix it, and the
                               ~9 GPU-h ablation would confirm nothing

THE MEASUREMENT, AND THE CONFOUND IT MUST NOT FALL INTO
-------------------------------------------------------
"Var_t(s) is small" is meaningless on its own, because the INPUT barely varies
within a window either. The window is 32,768 bp and phi lives on 5 kb bins, so
a window spans only ~6.5 bins -- phi is interpolated across a handful of knots
and is intrinsically smooth at this scale. A small Var_t(s) could just be a
faithful encoding of a nearly-constant input.

So the reported quantity is a RATIO, not a variance:

    keep(x) = mean_over_windows[ Var_t(x) ] / Var_global(x)

the fraction of x's total variance that lives WITHIN a window rather than
between windows. keep(phi) is the input's own within-window budget, measured on
the same windows. The diagnostic is keep(s) vs keep(phi):

    keep(s) << keep(phi)   the encoder specifically destroys within-window
                           variation -- bottleneck confirmed
    keep(s) ~= keep(phi)   the encoder passes what it was given; the input
                           simply has little within-window structure to pass

Note the second outcome is NOT a clean bill of health for the design: it would
mean the mechanism is starved by the WINDOW SIZE, not by the encoder, and the
fix is a wider window rather than a wider d_struct.

Real phi is used here (unlike the P5 probe, which withholds it). This measures
what the encoder does to its input, so it must be given its input.

Inference only. No training. Reads structural checkpoints. Writes
results/novel_model/p5_vars_diagnostic.json
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
from phase5_structure_probe import (                         # noqa: E402
    load_any, NOVEL_RUNS)

NOVEL = REPO / "results" / "novel_model"
OUT = NOVEL / "p5_vars_diagnostic.json"
BATCH = 2
D1_INERT = 0.05          # architecture_spec.md 4.1.3, same threshold as D1


class VarAccum:
    """Within-window variance and global variance for a (b, l, d) stream.

    Global variance is accumulated from raw sums so it never needs the whole
    tensor in memory; within-window variance is averaged per window.
    """

    def __init__(self):
        self.n = 0
        self.s = None
        self.ss = None
        self.within = 0.0
        self.n_win = 0

    def add(self, x):                       # x: (b, l, d) float32 on gpu
        x = x.detach().double()
        b, l, d = x.shape
        flat = x.reshape(-1, d)
        if self.s is None:
            self.s = torch.zeros(d, dtype=torch.float64, device=x.device)
            self.ss = torch.zeros(d, dtype=torch.float64, device=x.device)
        self.s += flat.sum(0)
        self.ss += (flat * flat).sum(0)
        self.n += flat.shape[0]
        # per-window variance across positions, averaged over d then over b
        self.within += float(x.var(dim=1, unbiased=False).mean(dim=-1).sum())
        self.n_win += b

    def result(self):
        if self.n == 0 or self.n_win == 0:
            return {"global_var": float("nan"), "within_var": float("nan"),
                    "keep": float("nan")}
        mean = self.s / self.n
        gvar = float((self.ss / self.n - mean * mean).clamp_min(0).mean())
        wvar = self.within / self.n_win
        return {"global_var": gvar, "within_var": wvar,
                "keep": wvar / gvar if gvar > 0 else float("nan")}


@torch.no_grad()
def measure(model, cfg, ds, device, symmetry, max_windows):
    """keep() for phi (input), s (encoder output), and W_dstruct@s per layer."""
    enc = VarAccum()
    per_dir = {}
    hooks = []

    def enc_hook(_m, _i, out):
        enc.add(out)
    hooks.append(model.struct_encoder.register_forward_hook(enc_hook))

    for li, layer in enumerate(model.layers):
        for dname in ("fwd", "rev"):
            key = f"L{li:02d}.{dname}"
            per_dir[key] = VarAccum()

            def mk(k):
                def h(_m, _i, out):
                    per_dir[k].add(out)
                return h
            hooks.append(getattr(layer, dname).W_dstruct
                         .register_forward_hook(mk(key)))

    phi_acc = VarAccum()
    n = min(len(ds), max_windows)
    try:
        for b0 in range(0, n, BATCH):
            items = [ds[i] for i in range(b0, min(b0 + BATCH, n))]
            batch = collate(items)
            phi = batch["phi"].to(device)
            phi_acc.add(phi)
            model(batch["tokens"].to(device), phi=phi,
                  phi_valid=batch["phi_valid"].to(device), symmetry=symmetry)
            if b0 and b0 % 50 == 0:
                print(f"    {b0}/{n} windows", flush=True)
    finally:
        for h in hooks:
            h.remove()

    return {"phi": phi_acc.result(), "s": enc.result(),
            "per_direction": {k: v.result() for k, v in per_dir.items()},
            "n_windows": n}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=120)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device.")
        return 2
    device = torch.device("cuda", 0)
    use_scan("triton")

    ds = WindowDataset("val", structural=True, rc_augment=False)
    symmetry = torch.from_numpy(ds.symmetry.astype(np.float32)).to(device)

    print("=" * 74)
    print("D4 -- ENCODER BOTTLENECK TEST   keep(x) = within-window var / global var")
    print("=" * 74)
    print("  real phi IS supplied here (this measures what the encoder does to it)")
    print()

    out = {"runs": {}, "config": {"n_windows": args.n_windows,
                                  "d1_inert_threshold": D1_INERT}}
    for d in sorted(NOVEL_RUNS.glob("structural_seed*")):
        if not (d / "checkpoint.pt").exists():
            continue
        t0 = time.time()
        model, cfg, step = load_any(d, device)
        print("-" * 74)
        print(f"{d.name}  step {step:,}  d_struct={cfg.d_struct}")
        r = measure(model, cfg, ds, device, symmetry, args.n_windows)
        del model
        torch.cuda.empty_cache()

        kp, ks = r["phi"]["keep"], r["s"]["keep"]
        print(f"  keep(phi)  = {kp:.4f}   "
              f"(within {r['phi']['within_var']:.4e} / global {r['phi']['global_var']:.4e})")
        print(f"  keep(s)    = {ks:.4f}   "
              f"(within {r['s']['within_var']:.4e} / global {r['s']['global_var']:.4e})")
        print(f"  ratio keep(s)/keep(phi) = {ks / kp:.4f}" if kp > 0 else "")

        early = [v["keep"] for k, v in r["per_direction"].items()
                 if int(k[1:3]) <= 11]
        late = [v["keep"] for k, v in r["per_direction"].items()
                if int(k[1:3]) >= 12]
        print(f"  keep(W_dstruct@s)  L00-L11 median {np.nanmedian(early):.4f}"
              f"   L12-L15 median {np.nanmedian(late):.4f}")
        r["seconds"] = round(time.time() - t0, 1)
        r["step"] = step
        out["runs"][d.name] = r
        print(f"  ({r['seconds']:.0f} s)")

    if not out["runs"]:
        print("no structural checkpoints found")
        return 2

    ks = [v["s"]["keep"] for v in out["runs"].values()]
    kp = [v["phi"]["keep"] for v in out["runs"].values()]
    ratio = [a / b for a, b in zip(ks, kp) if b > 0]
    out["summary"] = {
        "keep_phi_mean": float(np.mean(kp)), "keep_s_mean": float(np.mean(ks)),
        "ratio_mean": float(np.mean(ratio)) if ratio else None,
    }

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  keep(phi) mean = {np.mean(kp):.4f}")
    print(f"  keep(s)   mean = {np.mean(ks):.4f}")
    if ratio:
        rm = float(np.mean(ratio))
        print(f"  keep(s)/keep(phi) = {rm:.4f}")
        print()
        if rm < 0.5:
            print("  => The encoder DESTROYS within-window variation.")
            print("     The d_struct=8 no-encoder ablation is justified.")
        else:
            print("  => The encoder PASSES what it was given. The early-layer")
            print("     deadness is NOT an encoder bottleneck, so the")
            print("     no-encoder ablation would not fix it. Suspect the")
            print("     32,768 bp window instead: phi lives on 5 kb bins, so a")
            print("     window spans only ~6.5 bins and has little")
            print("     within-window structure to encode in the first place.")
    print()
    print("  This is a mechanism diagnostic on 3 seeds. It says which")
    print("  experiment to run next; it is not itself a result about chromatin.")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
