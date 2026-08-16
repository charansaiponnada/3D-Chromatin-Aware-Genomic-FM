#!/usr/bin/env python3
"""P1 -- swap phi at inference. The Phase 4 gate.

architecture_spec.md 4.1.3, as amended 2026-08-16 (decision 5). Takes a trained
real-structure model and feeds it each control at EVALUATION ONLY, measuring two
quantities that the original rule conflated:

  BENEFIT   Delta_S = val_bits(S) - val_bits(real).
            Does wrong structure make the model worse?

  RELIANCE  D_S = mean KL( p_real || p_S ) over masked positions, and
            F_S = fraction of masked positions where the argmax flips.
            Does the model's behaviour depend on phi at all?

Why both. Masked-token prediction over a 32 kb window is dominated by local
k-mer statistics, so a flat Delta_S is consistent with "the mechanism is inert"
AND with "the mechanism is used but MLM cannot express what it contributes".
Those have opposite consequences -- the first says stop, the second says proceed
to Phase 5 -- and a loss delta alone cannot separate them. Divergence can:
a model that ignores phi CANNOT change its predictions when phi is replaced.

    ./3d-gen/bin/python scripts/phase4_p1_swap.py                # all seeds found
    ./3d-gen/bin/python scripts/phase4_p1_swap.py structural_seed0

THE NULL FLOOR -- corrected 2026-08-16 after a dry run on an untrained model.

Two quantities are measured. Only one of them is a threshold.

  floor_kernel   same phi, same masking, evaluated twice. This IS the null
                 floor. A model that ignores phi cannot change its predictions
                 when phi is replaced -- not "changes them a little", but
                 produces bitwise identical logits. Measured on a 20-step model
                 whose W_dstruct is still at its zero init: floor_kernel KL was
                 exactly 0.0 and every control returned KL ~1e-23, i.e. zero to
                 machine precision. Inertness is therefore detectable at
                 floating-point resolution, and this is the correct bar.

  floor_masking  same phi, DIFFERENT masking seed, compared at positions masked
                 under both. Reported as CONTEXT ONLY -- the scale of how far
                 predictions move under a large perturbation of the PRIMARY
                 input. It is deliberately NOT the pass threshold.

Why floor_masking was wrong as a threshold, and this is worth stating because
the spec originally said otherwise: re-masking changes 15% of the DNA the model
is reading, while a phi swap changes an auxiliary conditioning channel.
Requiring D_S1 > floor_masking would demand that lying about the folding move
predictions further than re-masking a seventh of the sequence does. That is not
a test of reliance, it is a test of whether phi dominates the sequence, which
nobody claims and which would be alarming if true. On the dry run floor_masking
was KL 19.56 -- larger than any plausible structural effect.

  RELIANCE PASSES iff  D_S1 >> floor_kernel  and  F_S1 is materially non-zero.

D_S0 -- the divergence from removing structure altogether -- is reported as the
natural upper reference: it is the largest effect phi can have on this model, so
D_S1 / D_S0 says how much of the available structural signal the shuffle
actually disrupts.

Inference only. Runs on one GPU, never writes a checkpoint, never trains.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from chromfm.model import BiMambaLM, ModelConfig, use_scan   # noqa: E402
from phase1_dataset import WindowDataset              # noqa: E402
from train import collate, mask_tokens, LN2           # noqa: E402

NOVEL = REPO / "results" / "novel_model"
CONTROLS = ["real", "S0", "S1", "S2", "S3", "S4"]
EVAL_SEED = 1234          # matches train.py's default; the same masking the
                          # training runs were evaluated under
ALT_SEED = 4321           # for the masking floor only


def load_model(run_dir: Path, device) -> tuple[BiMambaLM, dict]:
    import yaml
    cfg_doc = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    m = cfg_doc["model"]
    cfg = ModelConfig(
        d_model=m["d_model"], n_layer=m["n_layer"], d_state=m["d_state"],
        d_conv=m["d_conv"], expand=m["expand"], vocab_size=m["vocab_size"],
        structural=m["structural"], d_struct=m["d_struct"],
        d_struct_raw=m["d_struct_raw"], d_struct_hidden=m["d_struct_hidden"],
        use_permeability=m["use_permeability"],
        dt_min=m["dt_min"], dt_max=m["dt_max"], dt_floor=m["dt_floor"],
    )
    if not cfg.structural:
        raise ValueError(f"{run_dir.name} is not a structural run; P1 swaps phi "
                         f"on a model that was trained with phi")
    ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
    model = BiMambaLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, {"step": int(ck["step"]), "status": cfg_doc.get("status"),
                   "n_parameters": m["n_parameters"]}


@torch.no_grad()
def run_pass(model, ds, device, symmetry, mask_seed: int, batch_size: int = 2):
    """One full pass over the validation split.

    Returns (bits/nt, accuracy, log-probs at masked positions, a flat index of
    which positions those were). The index lets two passes with DIFFERENT
    masking be compared on their intersection.
    """
    gen = torch.Generator(device=device).manual_seed(mask_seed)
    tot_loss = tot_correct = tot_count = 0.0
    logps, keys = [], []
    order = list(range(len(ds)))
    for b0 in range(0, len(order), batch_size):
        items = [ds[i] for i in order[b0:b0 + batch_size]]
        batch = collate(items)
        tokens = batch["tokens"].to(device)
        inputs, labels = mask_tokens(tokens, 0.15, gen, 0.8, 0.1)
        kw = {}
        if "phi" in batch:
            kw = {"phi": batch["phi"].to(device),
                  "phi_valid": batch["phi_valid"].to(device),
                  "symmetry": symmetry}
        logits = model(inputs, **kw)
        sel = labels != -100
        if not sel.any():
            continue
        lg, lb = logits[sel], labels[sel]
        tot_loss += float(F.cross_entropy(lg, lb, reduction="sum"))
        tot_correct += float((lg.argmax(-1) == lb).sum())
        tot_count += float(sel.sum())
        logps.append(F.log_softmax(lg.float(), dim=-1).cpu())
        # global position id = window index * window length + offset
        bi, pos = sel.nonzero(as_tuple=True)
        keys.append(((bi.cpu() + b0) * tokens.shape[1] + pos.cpu()))
    n = max(tot_count, 1.0)
    return {
        "val_bits_per_nucleotide": (tot_loss / n) / LN2,
        "val_accuracy": tot_correct / n,
        "n_masked": int(tot_count),
        "logp": torch.cat(logps) if logps else torch.empty(0),
        "key": torch.cat(keys) if keys else torch.empty(0, dtype=torch.long),
    }


def divergence(ref: dict, other: dict) -> dict:
    """KL(p_ref || p_other) and argmax-flip rate, on shared masked positions."""
    if ref["key"].numel() == 0 or other["key"].numel() == 0:
        return {"kl_mean": float("nan"), "argmax_flip_frac": float("nan"),
                "n_shared": 0}
    if torch.equal(ref["key"], other["key"]):
        ra, oa = ref["logp"], other["logp"]
        n_shared = ra.shape[0]
    else:
        # different masking -> compare only where both masked
        rk, ok = ref["key"].numpy(), other["key"].numpy()
        shared = np.intersect1d(rk, ok)
        ra = ref["logp"][torch.from_numpy(np.searchsorted(rk, shared))]
        oa = other["logp"][torch.from_numpy(np.searchsorted(ok, shared))]
        n_shared = len(shared)
    p = ra.exp()
    kl = (p * (ra - oa)).sum(-1)
    return {
        "kl_mean": float(kl.mean()),
        "kl_median": float(kl.median()),
        "argmax_flip_frac": float((ra.argmax(-1) != oa.argmax(-1)).float().mean()),
        "n_shared": int(n_shared),
    }


def main() -> int:
    t0 = time.time()
    if not torch.cuda.is_available():
        print("No CUDA device.")
        return 2
    device = torch.device("cuda", 0)
    use_scan("triton")          # same scan both arms trained on

    names = sys.argv[1:] or sorted(
        d.name for d in NOVEL.glob("structural_seed*") if d.is_dir())
    runs = []
    for nm in names:
        d = NOVEL / nm
        cfg = d / "run_config.yaml"
        if not cfg.exists() or not (d / "checkpoint.pt").exists():
            print(f"skip {nm}: no checkpoint yet")
            continue
        if "status: COMPLETED" not in cfg.read_text():
            print(f"skip {nm}: not COMPLETED -- P1 reads finished runs only")
            continue
        runs.append(d)
    if not runs:
        print("No COMPLETED structural runs. Nothing to swap.")
        return 1

    print(f"=== P1 swap at inference ===")
    print(f"runs: {', '.join(r.name for r in runs)}")
    print(f"controls: {', '.join(CONTROLS)}\n")

    datasets = {}
    for c in CONTROLS:
        ctl = "none" if c in ("real", "S0") else c
        datasets[c] = WindowDataset("val", structural=True, rc_augment=False,
                                    seed=0, phi_control=ctl)
    sym0 = torch.from_numpy(
        np.asarray(datasets["real"].symmetry, dtype=np.float32)).to(device)

    all_out = {}
    for run_dir in runs:
        model, meta = load_model(run_dir, device)
        print(f"--- {run_dir.name}  (step {meta['step']}, "
              f"{meta['n_parameters']:,} params) ---")

        ref = run_pass(model, datasets["real"], device, sym0, EVAL_SEED)
        print(f"  real            val {ref['val_bits_per_nucleotide']:.4f} bits  "
              f"acc {ref['val_accuracy']:.4f}  masked {ref['n_masked']:,}")

        # ---- floors, measured not assumed ----
        again = run_pass(model, datasets["real"], device, sym0, EVAL_SEED)
        f_kernel = divergence(ref, again)
        alt = run_pass(model, datasets["real"], device, sym0, ALT_SEED)
        f_mask = divergence(ref, alt)
        print(f"  floor_kernel    KL {f_kernel['kl_mean']:.3e}  "
              f"flip {f_kernel['argmax_flip_frac']:.6f}")
        print(f"  floor_masking   KL {f_mask['kl_mean']:.3e}  "
              f"flip {f_mask['argmax_flip_frac']:.6f}  "
              f"(n={f_mask['n_shared']:,} shared positions)")

        res = {"run": run_dir.name, **meta,
               "real": {k: ref[k] for k in
                        ("val_bits_per_nucleotide", "val_accuracy", "n_masked")},
               "floor_kernel": f_kernel, "floor_masking": f_mask,
               "controls": {}}

        for c in CONTROLS:
            if c == "real":
                continue
            if c == "S0":
                # S0 is s_t := 0, i.e. phi present but structurally silent.
                # Implemented by zeroing phi_valid's effect: feed zeros.
                ds_c = datasets["real"]
                zero = True
            else:
                ds_c, zero = datasets[c], False

            if zero:
                class _Zeroed:
                    def __init__(self, base): self.b = base
                    def __len__(self): return len(self.b)
                    def __getitem__(self, i):
                        it = dict(self.b[i])
                        it["phi"] = np.zeros_like(it["phi"])
                        return it
                    window = ds_c.window
                    symmetry = ds_c.symmetry
                ds_use = _Zeroed(ds_c)
            else:
                ds_use = ds_c

            out = run_pass(model, ds_use, device, sym0, EVAL_SEED)
            dv = divergence(ref, out)
            delta = (out["val_bits_per_nucleotide"]
                     - ref["val_bits_per_nucleotide"])
            # The bar is the KERNEL floor, not the masking floor. An inert
            # mechanism gives bitwise-identical logits, so 1e-12 is already a
            # generous margin above numerical noise.
            live = dv["kl_mean"] > max(f_kernel["kl_mean"], 1e-12)
            res["controls"][c] = {
                "val_bits_per_nucleotide": out["val_bits_per_nucleotide"],
                "val_accuracy": out["val_accuracy"],
                "delta_bits": delta,
                **{k: v for k, v in dv.items() if k != "n_shared"},
                "kl_above_kernel_floor": bool(live),
            }
            print(f"  {c:<15} val {out['val_bits_per_nucleotide']:.4f}  "
                  f"delta {delta:+.4f}  KL {dv['kl_mean']:.3e}  "
                  f"flip {dv['argmax_flip_frac']:.6f}  "
                  f"{'LIVE' if live else 'inert'}")

        all_out[run_dir.name] = res
        print()
        del model
        torch.cuda.empty_cache()

    # ---------------- aggregate, and read the gate ----------------
    print("=== aggregate over completed seeds ===")
    n = len(all_out)
    print(f"seeds: {n}  "
          f"{'(GATE NEEDS >=3 -- this is a preview, not the verdict)' if n < 3 else ''}")

    def col(f):
        return np.array([f(v) for v in all_out.values()], dtype=float)

    kl_s0 = float(col(lambda v: v["controls"]["S0"]["kl_mean"]).mean())
    summary = {"n_seeds": n, "gate_readable": n >= 3,
               "kernel_floor_mean": float(col(
                   lambda v: v["floor_kernel"]["kl_mean"]).mean()),
               "masking_floor_mean_CONTEXT_ONLY": float(col(
                   lambda v: v["floor_masking"]["kl_mean"]).mean()),
               "per_control": {}}
    for c in CONTROLS[1:]:
        d = col(lambda v: v["controls"][c]["delta_bits"])
        k = col(lambda v: v["controls"][c]["kl_mean"])
        summary["per_control"][c] = {
            "delta_bits_mean": float(d.mean()),
            "delta_bits_sd": float(d.std(ddof=1)) if n > 1 else None,
            "kl_mean": float(k.mean()),
            "flip_frac_mean": float(col(
                lambda v: v["controls"][c]["argmax_flip_frac"]).mean()),
            # what fraction of the total available structural effect this
            # control disrupts; S0 (structure removed entirely) is the ceiling
            "kl_relative_to_S0": (float(k.mean() / kl_s0)
                                  if kl_s0 > 0 else None),
        }
        s = summary["per_control"][c]
        sd = f"{s['delta_bits_sd']:.4f}" if s["delta_bits_sd"] is not None else "n/a"
        rel = (f"{s['kl_relative_to_S0']:.2f}x S0"
               if s["kl_relative_to_S0"] is not None else "n/a")
        print(f"  {c:<4} delta {s['delta_bits_mean']:+.4f} (sd {sd})   "
              f"KL {s['kl_mean']:.3e}   flip {s['flip_frac_mean']:.4f}   {rel}")
    print(f"\n  kernel floor (the bar):      "
          f"{summary['kernel_floor_mean']:.3e}")
    print(f"  masking floor (context only): "
          f"{summary['masking_floor_mean_CONTEXT_ONLY']:.3e}  "
          f"-- NOT a threshold; see this script's header")

    out_path = NOVEL / "p1_swap_results.json"
    out_path.write_text(json.dumps(
        {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "eval_seed": EVAL_SEED, "alt_masking_seed": ALT_SEED,
         "note": "Delta = benefit, KL = reliance. architecture_spec.md 4.1.3 "
                 "amendment 2026-08-16. The gate is read on S1.",
         "per_run": all_out, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(REPO)}  ({time.time()-t0:.0f}s)")
    if n < 3:
        print("\nNOT THE VERDICT. The gate requires >=3 completed seeds "
              "(architecture_spec.md 4.1.3). This is a harness check and an "
              "early read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
