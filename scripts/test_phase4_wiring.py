#!/usr/bin/env python3
"""Verify the Phase 4 structural wiring before spending GPU hours on it.

CLAUDE.md Phase 4 asks specifically for a test that the mechanism is *using* the
Hi-C signal rather than silently ignoring it. That is what most of this file is:
every check below fails loudly if phi is dropped, mis-shaped, mis-signed, or
detached from the gradient somewhere between the .npz on disk and Delta inside
the scan.

    ./3d-gen/bin/python scripts/test_phase4_wiring.py

Runs on CPU except where a scan is needed. Takes well under a minute.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from chromfm.model import BiMambaLM, ModelConfig      # noqa: E402
from phase1_dataset import (WindowDataset, apply_phi_control,   # noqa: E402
                            PHI_CONTROL_SEED, S2_SHIFT_BP, RES)
from train import collate, batch_struct, DeltaCapture, empirical_tau  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def small_cfg(structural: bool) -> ModelConfig:
    return ModelConfig(d_model=32, n_layer=2, d_state=8, expand=2,
                       structural=structural)


def main() -> int:
    torch.manual_seed(0)

    # ---------------------------------------------------------------- controls
    print("\nphi controls (architecture_spec.md 4.1.3)")
    ds = WindowDataset("train", structural=True, rc_augment=False, seed=0)
    phi0, use0 = ds.phi, ds.usable

    p1, u1 = apply_phi_control(phi0, use0, "S1")
    check("S1 preserves the marginal exactly",
          np.allclose(np.sort(phi0, axis=0), np.sort(p1, axis=0)),
          "sorted columns identical")
    check("S1 destroys bin alignment",
          not np.allclose(phi0, p1),
          f"{float((phi0 != p1).any(axis=1).mean()):.3f} of bins moved")
    check("S1 carries `usable` through the same permutation",
          int(u1.sum()) == int(use0.sum()) and not np.array_equal(u1, use0),
          f"{int(u1.sum())} usable bins, same count, different positions")
    check("S1 is deterministic across constructions",
          np.array_equal(apply_phi_control(phi0, use0, "S1")[0], p1),
          f"fixed seed {PHI_CONTROL_SEED}")

    p2, u2 = apply_phi_control(phi0, use0, "S2")
    shift = S2_SHIFT_BP // RES
    check("S2 preserves the marginal exactly",
          np.allclose(np.sort(phi0, axis=0), np.sort(p2, axis=0)))
    # Tolerance is 1e-3, not 1e-6: np.roll replaces one real bin-to-bin
    # transition with the seam, which moves the successive-difference sd by
    # ~1.2e-4 relative on 27,679 bins. That residual is the control working as
    # designed, not drift -- S1 moves the same quantity by a factor of 4.8.
    check("S2 preserves local autocorrelation",
          np.allclose(np.diff(phi0, axis=0).std(), np.diff(p2, axis=0).std(),
                      rtol=1e-3),
          "successive-difference sd unchanged, unlike S1")
    check("S1 does NOT preserve local autocorrelation",
          not np.isclose(np.diff(phi0, axis=0).std(),
                         np.diff(p1, axis=0).std(), rtol=1e-2),
          f"sd {np.diff(phi0, axis=0).std():.4f} -> {np.diff(p1, axis=0).std():.4f}")
    check("S2 shifts by the requested distance",
          np.allclose(np.roll(p2, -shift, axis=0), phi0),
          f"{S2_SHIFT_BP:,} bp = {shift} bins")

    # S3 and S4 are precomputed files, not transforms of phi. Built 2026-08-16
    # by phase4_build_s3.py and phase4_build_s4.py.
    for name, why in (("S3", "distance-matched rewire"),
                      ("S4", "sequence-matched covariates")):
        p, u = apply_phi_control(phi0, use0, name)
        check(f"{name} loads and matches phi's shape ({why})",
              p.shape == phi0.shape, str(p.shape))
        # Coverage must match exactly, or the control run sees a different
        # number of positions than the real run and coverage confounds the
        # comparison.
        check(f"{name} is defined at exactly the bins real phi is defined at",
              int(u.sum()) == int(use0.sum()),
              f"{int(u.sum()):,} vs {int(use0.sum()):,}")
        check(f"{name} differs from real phi",
              not np.allclose(np.nan_to_num(p), np.nan_to_num(phi0)))

    # S3's defining property: it must preserve the distance-decay curve while
    # destroying which locus pair holds which contact. Read from the build
    # report rather than recomputed, so the assertion tracks what was shipped.
    rep3 = json.loads((REPO / "data/processed/s3_validation_report.json")
                      .read_text(encoding="utf-8"))
    check("S3 preserves P(s) to floating-point exactness",
          rep3["ps_max_abs_deviation"] < 1e-5,
          f"max deviation {rep3['ps_max_abs_deviation']:.2e} over all diagonals")
    worst3 = max(abs(v) for v in rep3["s3_vs_real_feature_correlation"].values())
    check("S3 destroys locus-specific structure (|r| < 0.3 on every feature)",
          worst3 < 0.3, f"worst |r| = {worst3:.4f}")

    # S4's reason for existing: compartment_pc1 is largely 1D-explainable, the
    # rest of phi is not. If that stopped being true the control's framing in
    # architecture_spec.md 4.1.3 would need revisiting.
    rep4 = json.loads((REPO / "data/processed/s4_validation_report.json")
                      .read_text(encoding="utf-8"))
    check("S4 uses no Hi-C at all", rep4["hi_c_used"] is False)
    check("S4 reproduces phi's reverse-complement symmetry",
          rep4["feature_symmetry"] == [int(v) for v in ds.symmetry],
          str(rep4["feature_symmetry"]))
    c14 = rep4["phi_vs_s4_best_correlation"]
    check("S4 captures compartment_pc1, the 1D-explainable coordinate",
          abs(c14["compartment_pc1"]["best_r"]) > 0.5,
          f"r = {c14['compartment_pc1']['best_r']} with "
          f"{c14['compartment_pc1']['with']}")
    others = [abs(v["best_r"]) for k, v in c14.items() if k != "compartment_pc1"]
    check("the other seven phi coordinates are NOT 1D-explainable",
          max(others) < 0.35, f"worst |r| = {max(others):.4f}")

    try:
        apply_phi_control(phi0, use0, "S9")
        check("an unknown control is rejected", False)
    except ValueError:
        check("an unknown control is rejected", True)

    try:
        WindowDataset("train", structural=False, phi_control="S1")
        check("phi_control with structural=False is rejected", False)
    except ValueError:
        check("phi_control with structural=False is rejected", True,
              "would otherwise be a silent no-op mislabelled as a control")

    # ------------------------------------------------------------------ shapes
    print("\ndataset -> collate -> model")
    ds_s = WindowDataset("val", structural=True, rc_augment=False, seed=0)
    batch = collate([ds_s[0], ds_s[1]])
    check("collate carries phi and phi_valid",
          "phi" in batch and "phi_valid" in batch)
    check("phi is (batch, window, d_struct_raw) float32",
          tuple(batch["phi"].shape) == (2, ds_s.window, 8)
          and batch["phi"].dtype == torch.float32,
          str(tuple(batch["phi"].shape)))
    check("phi_valid is (batch, window) bool",
          tuple(batch["phi_valid"].shape) == (2, ds_s.window)
          and batch["phi_valid"].dtype == torch.bool)
    check("invalid positions carry exactly zero phi",
          float(batch["phi"][~batch["phi_valid"]].abs().max() if
                (~batch["phi_valid"]).any() else 0.0) == 0.0)

    # rc augmentation returns reversed views; collate must survive them
    ds_rc = WindowDataset("train", structural=True, rc_augment=True, seed=7)
    got_rc = False
    for i in range(40):
        item = ds_rc[i]
        if item["rc"]:
            got_rc = True
            collate([item])
            break
    check("collate accepts reverse-complemented (negative-stride) phi", got_rc,
          "found an rc window and stacked it without error")

    # --------------------------------------------------------------- parameters
    print("\nparameter accounting")
    base = BiMambaLM(ModelConfig(structural=False))
    struct = BiMambaLM(ModelConfig(structural=True))
    nb, nstr = base.n_params(), struct.n_params()
    over = (nstr - nb) / nb
    check("structural arm is within the 5% parameter budget", over <= 0.05,
          f"{nb:,} -> {nstr:,} = +{over*100:.2f}%")

    # ------------------------------------------------------------ signal is used
    print("\nis the mechanism actually using phi?")
    cfg = small_cfg(True)
    m = BiMambaLM(cfg)
    tok = batch["tokens"][:, :512]
    phi = batch["phi"][:, :512]
    pv = batch["phi_valid"][:, :512]
    sym = torch.from_numpy(ds_s.symmetry.astype(np.float32))

    with torch.no_grad():
        y0 = m(tok, phi=phi, phi_valid=pv, symmetry=sym)
    check("W_dstruct is zero-initialised, so the structural arm starts as the "
          "baseline", all(float(getattr(l, d).W_dstruct.weight.abs().max()) == 0.0
                          for l in m.layers for d in ("fwd", "rev")))

    # With W_dstruct at zero the model MUST ignore phi. That is the init
    # equivalence guarantee -- and it means "different phi -> different output"
    # is only a meaningful test once the weights are non-zero.
    with torch.no_grad():
        y_shuf = m(tok, phi=torch.randn_like(phi), phi_valid=pv, symmetry=sym)
    check("at init, phi genuinely has no effect (init equivalence holds)",
          torch.allclose(y0, y_shuf, atol=0, rtol=0))

    with torch.no_grad():
        for l in m.layers:
            for d in ("fwd", "rev"):
                getattr(l, d).W_dstruct.weight.normal_(0, 0.5)
        y1 = m(tok, phi=phi, phi_valid=pv, symmetry=sym)
        y2 = m(tok, phi=torch.randn_like(phi), phi_valid=pv, symmetry=sym)
    check("once W_dstruct is non-zero, shuffling phi changes the output",
          not torch.allclose(y1, y2, atol=1e-6),
          f"max |dy| = {float((y1-y2).abs().max()):.4f}")

    # ------------------------------------------------------- tau reads the right Delta
    print("\ntau measurement on the structural arm")
    sk = {"phi": phi, "phi_valid": pv, "symmetry": sym}
    with DeltaCapture(m) as cap:
        with torch.no_grad():
            m(tok, **sk)
        k = "layer0.fwd"
        check("DeltaCapture hooks W_dstruct on the structural arm",
              k in cap.struct, "structural contribution captured")
        check("dt_pre() differs from the raw dt_proj output",
              not torch.allclose(cap.dt_pre(k), cap.store[k]),
              "tau would otherwise be the BASELINE's, silently")

    t_real = empirical_tau(m, tok, seed=1, struct_kwargs=sk)
    t_shuf = empirical_tau(m, tok, seed=1, struct_kwargs={
        **sk, "phi": torch.randn_like(phi)})
    check("tau responds to the structural input",
          t_real["summary"]["tau_median"] != t_shuf["summary"]["tau_median"],
          f"median {t_real['summary']['tau_median']:.1f} vs "
          f"{t_shuf['summary']['tau_median']:.1f}")

    mb = BiMambaLM(small_cfg(False))
    with DeltaCapture(mb) as cap:
        with torch.no_grad():
            mb(tok)
        check("baseline path is unchanged: no W_dstruct, dt_pre() == dt_proj out",
              not cap.struct and torch.equal(cap.dt_pre("layer0.fwd"),
                                             cap.store["layer0.fwd"]))

    # ------------------------------------------------------------------ gradient
    print("\ngradient reaches the structural parameters")
    m.zero_grad()
    out = m(tok, **sk)
    out.square().mean().backward()
    enc_g = m.struct_encoder.net[0].weight.grad
    wds_g = m.layers[0].fwd.W_dstruct.weight.grad
    check("gradient reaches the structural encoder",
          enc_g is not None and float(enc_g.abs().max()) > 0,
          f"max |g| = {float(enc_g.abs().max()):.3e}")
    check("gradient reaches W_dstruct",
          wds_g is not None and float(wds_g.abs().max()) > 0,
          f"max |g| = {float(wds_g.abs().max()):.3e}")

    # ------------------------------------------------------------- batch_struct
    print("\nbatch_struct plumbing")
    check("batch_struct returns {} for a baseline batch",
          batch_struct({"tokens": tok}, torch.device("cpu"), None) == {})
    bs = batch_struct(batch, torch.device("cpu"), sym)
    check("batch_struct passes phi, phi_valid and symmetry",
          set(bs) == {"phi", "phi_valid", "symmetry"})

    print(f"\n{len(PASS)}/{len(PASS)+len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
