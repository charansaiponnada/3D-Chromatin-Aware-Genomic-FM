"""Unit tests for the BiMamba backbone and its structural pathway.

Runs on CPU at small L. The point is not performance -- it is to prove the
structural mechanism is wired to something that changes the output and receives
gradient, before any GPU time is spent. architecture_spec.md 4.1.4 lists three
failure modes (F1, F4, F7) that all present as "trains to baseline and raises no
error", so the wiring has to be checked directly rather than inferred from loss.

Run:  python scripts/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chromfm.model import BiMambaLM, ModelConfig, build  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))


def small(**kw) -> ModelConfig:
    return ModelConfig(d_model=32, n_layer=2, d_state=4, vocab_size=16, **kw)


def test_param_match():
    """Full-size counts must equal scripts/param_accounting.py."""
    base = BiMambaLM(ModelConfig(structural=False))
    struct = BiMambaLM(ModelConfig(structural=True))
    nb, ns = base.n_params(), struct.n_params()
    # embedding is tied to lm_head, so it is counted once
    delta_pct = 100.0 * (ns - nb) / nb
    check("baseline parameter count == 7,725,312", nb == 7_725_312, f"{nb:,}")
    check("structural parameter count == 7,758,354", ns == 7_758_354, f"{ns:,}")
    check("delta within 5% budget", abs(delta_pct) <= 5.0, f"+{delta_pct:.4f}%")


def test_forward_shapes():
    m = BiMambaLM(small())
    tok = torch.randint(2, 6, (2, 64))
    out = m(tok)
    check("baseline forward shape", tuple(out.shape) == (2, 64, 16), str(tuple(out.shape)))

    ms = BiMambaLM(small(structural=True))
    phi = torch.randn(2, 64, 8)
    valid = torch.ones(2, 64, dtype=torch.bool)
    sym = torch.tensor([1, 1, 1, -1, 1, -1, 1, 1])
    outs = ms(tok, phi, valid, sym)
    check("structural forward shape", tuple(outs.shape) == (2, 64, 16), str(tuple(outs.shape)))


def test_init_equivalence():
    """At init the structural model must equal the baseline numerically.

    W_dstruct = 0 and b_gate = -8, so any later divergence is attributable to
    learned structural signal rather than an initialisation shift.
    """
    torch.manual_seed(0)
    ms = BiMambaLM(small(structural=True))
    tok = torch.randint(2, 6, (1, 48))
    phi = torch.randn(1, 48, 8) * 3.0
    valid = torch.ones(1, 48, dtype=torch.bool)
    with torch.no_grad():
        a = ms(tok, phi, valid)
        b = ms(tok, torch.zeros_like(phi), valid)
    gap = (a - b).abs().max().item()
    # p = softplus(-8) ~ 3.4e-4 is applied regardless of s, so this is not exactly
    # zero -- it must simply be tiny relative to the activations
    scale = a.abs().max().item()
    check("structural == baseline at init (relative)", gap / scale < 1e-3,
          f"max gap {gap:.3e} vs activation scale {scale:.3e}")


def test_structure_changes_output():
    """After perturbing W_dstruct, phi must actually move the output."""
    torch.manual_seed(0)
    ms = BiMambaLM(small(structural=True))
    with torch.no_grad():
        for layer in ms.layers:
            for d in (layer.fwd, layer.rev):
                d.W_dstruct.weight.normal_(0, 0.5)
                d.w_gate.weight.normal_(0, 0.5)
                d.w_gate.bias.fill_(0.0)
    tok = torch.randint(2, 6, (1, 48))
    valid = torch.ones(1, 48, dtype=torch.bool)
    with torch.no_grad():
        a = ms(tok, torch.randn(1, 48, 8), valid)
        b = ms(tok, torch.randn(1, 48, 8), valid)
    diff = (a - b).abs().max().item()
    check("different phi -> different output", diff > 1e-4, f"max diff {diff:.3e}")


def test_gradient_reaches_structural_pathway():
    """Zero-init must not block learning: dL/dW = (dL/dDelta) s^T is non-zero."""
    torch.manual_seed(0)
    ms = BiMambaLM(small(structural=True))
    tok = torch.randint(2, 6, (1, 32))
    phi = torch.randn(1, 32, 8)
    valid = torch.ones(1, 32, dtype=torch.bool)
    out = ms(tok, phi, valid)
    out.square().mean().backward()

    gw = ms.layers[0].fwd.W_dstruct.weight.grad
    gg = ms.layers[0].fwd.w_gate.weight.grad
    ge = ms.struct_encoder.net[0].weight.grad
    check("grad reaches W_dstruct at zero init", gw is not None and gw.abs().sum() > 0,
          f"|grad| = {gw.abs().sum():.3e}")
    check("grad reaches permeability gate", gg is not None and gg.abs().sum() > 0,
          f"|grad| = {gg.abs().sum():.3e}")

    # The encoder sits BEHIND W_dstruct, and dL/ds = W_dstruct^T . dL/dDelta.
    # With W_dstruct zero-initialised that product is exactly zero, so the
    # encoder is gradient-isolated on step 0 by construction. It unblocks once
    # W_dstruct moves. This is asserted rather than treated as a failure so a
    # regression that leaves it permanently blocked would be caught.
    check("structural encoder is gradient-isolated at init (by design)",
          ge is not None and ge.abs().sum() == 0, f"|grad| = {ge.abs().sum():.3e}")

    torch.optim.SGD(ms.parameters(), lr=1e-2).step()
    ms.zero_grad()
    ms(tok, phi, valid).square().mean().backward()
    ge2 = ms.struct_encoder.net[0].weight.grad
    check("structural encoder unblocks after one step", ge2.abs().sum() > 0,
          f"|grad| = {ge2.abs().sum():.3e} -- but see the ENCODER BOTTLENECK note")


def test_invalid_masking():
    """phi_valid = False must zero the structural signal, never leak a NaN."""
    torch.manual_seed(0)
    ms = BiMambaLM(small(structural=True))
    with torch.no_grad():
        for layer in ms.layers:
            for d in (layer.fwd, layer.rev):
                d.W_dstruct.weight.normal_(0, 0.5)
    tok = torch.randint(2, 6, (1, 32))
    phi = torch.randn(1, 32, 8)
    valid = torch.ones(1, 32, dtype=torch.bool)
    valid[:, 16:] = False
    phi_poisoned = phi.clone()
    phi_poisoned[:, 16:] = float("nan")
    with torch.no_grad():
        a = ms(tok, phi * valid.unsqueeze(-1), valid)
    ok_finite = torch.isfinite(a).all().item()
    check("masked positions produce finite output", bool(ok_finite))
    # and the dataset layer is what guarantees phi is never NaN in the first place
    check("NaN phi would propagate (dataset must prevent it)",
          not torch.isfinite(ms(tok, phi_poisoned, valid)).all().item(),
          "confirms the masking responsibility sits in phase1_dataset.py")


def test_rc_symmetry_applied():
    """The reverse pass must receive sign-flipped antisymmetric coordinates."""
    torch.manual_seed(0)
    ms = BiMambaLM(small(structural=True))
    with torch.no_grad():
        for layer in ms.layers:
            for d in (layer.fwd, layer.rev):
                d.W_dstruct.weight.normal_(0, 0.5)
    tok = torch.randint(2, 6, (1, 32))
    phi = torch.randn(1, 32, 8)
    valid = torch.ones(1, 32, dtype=torch.bool)
    sym = torch.tensor([1, 1, 1, -1, 1, -1, 1, 1])
    with torch.no_grad():
        with_sym = ms(tok, phi, valid, sym)
        without = ms(tok, phi, valid, None)
    diff = (with_sym - without).abs().max().item()
    check("symmetry vector changes the reverse pass", diff > 1e-5, f"max diff {diff:.3e}")


def test_tau_stats():
    m = BiMambaLM(ModelConfig(structural=False))
    st = m.tau_stats()
    tau_max = max(v["tau_max"] for v in st.values())
    med = sorted(v["tau_median"] for v in st.values())[len(st) // 2]
    check("tau_stats returns per-layer entries", len(st) == 32, f"{len(st)} entries")
    check("tau_max near the F4 measurement (~994 tokens)",
          500 < tau_max < 1500, f"{tau_max:.0f} tokens")
    print(f"       median tau across layers: {med:.1f} tokens "
          f"(a 5 kb bin is 5,000 -- see F4)")


def main():
    print("BiMamba model tests (CPU, small L)\n")
    for fn in (test_param_match, test_forward_shapes, test_init_equivalence,
               test_structure_changes_output, test_gradient_reaches_structural_pathway,
               test_invalid_masking, test_rc_symmetry_applied, test_tau_stats):
        print(f"{fn.__name__}:")
        fn()
        print()
    n_pass = sum(1 for _, ok in results if ok)
    print(f"{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
