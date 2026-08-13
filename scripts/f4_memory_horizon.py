"""F4 check -- is the SSM's memory horizon comparable to a Hi-C bin?

architecture_spec.md 4.1.4 failure mode F4: if the state's effective memory
horizon tau is much shorter than one Hi-C bin, then s_t is locally constant
across everything the state can see, and mechanism (a) degenerates into F1
(a constant absorbed into b_dt). The data card flags this as a check that
must NOT be deferred to Phase 4 debugging, because the fix is a Phase 1
resolution decision.

tau for channel i, state n, in tokens:
    A_bar = exp(Delta_i * A_in),  A_in < 0
    tau   = -1 / (Delta_i * A_in)      (e-folding length)

This computes tau from Mamba's standard initialisation, instantiated rather
than asserted, and compares it against the 5 kb bin used by the pilot.

Run:  python scripts/f4_memory_horizon.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

# --- config must mirror scripts/param_accounting.py ----------------------
D_MODEL, N_LAYER, D_STATE, EXPAND = 256, 16, 16, 2
D_INNER = D_MODEL * EXPAND
DT_RANK = math.ceil(D_MODEL / 16)

# Delta init. These MUST mirror ModelConfig.dt_min/dt_max/dt_floor in
# src/chromfm/model.py -- this script exists to predict what the model does, so
# a drift between the two makes it predict a model that is not being trained.
# Mamba's reference values were (1e-3, 1e-1, 1e-4); they cap tau at 1/dt_min =
# 1,000 tokens, which failed the F4 gate in Phase 3 (results/baselines/
# phase3_report.txt). Changed 2026-08-12.
DT_MIN, DT_MAX, DT_INIT_FLOOR = 1e-6, 1e-1, 1e-7

BIN_BP = 5_000            # pilot Hi-C resolution -> tokens per bin at 1 bp/token
ALT_BINS = (1_000, 2_000, 5_000, 10_000)   # levels present in the same mcool


def mamba_dt_init(d_inner: int, seed: int = 0) -> torch.Tensor:
    """Reproduce Mamba's dt_proj bias init, then recover Delta = softplus(bias)."""
    g = torch.Generator().manual_seed(seed)
    dt = torch.exp(
        torch.rand(d_inner, generator=g) * (math.log(DT_MAX) - math.log(DT_MIN))
        + math.log(DT_MIN)
    ).clamp(min=DT_INIT_FLOOR)
    # inverse softplus, as in the reference implementation
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    return torch.nn.functional.softplus(inv_dt)      # == dt, by construction


def main() -> None:
    delta = mamba_dt_init(D_INNER).numpy()           # (d_inner,)
    # A = -exp(A_log), A_log = log(1..d_state)  ->  A = -(1..d_state)
    A = -np.arange(1, D_STATE + 1, dtype=np.float64)  # (d_state,)

    tau = 1.0 / (delta[:, None] * np.abs(A)[None, :])  # tokens

    print("=== per-layer memory horizon at initialisation ===")
    print(f"config: d_model={D_MODEL} d_inner={D_INNER} d_state={D_STATE} "
          f"n_layer={N_LAYER}")
    print(f"Delta at init: min {delta.min():.5f}  median {np.median(delta):.5f}  "
          f"max {delta.max():.5f}")
    print(f"|A|: {abs(A[0]):.0f} .. {abs(A[-1]):.0f}")
    print()
    qs = [0, 50, 90, 99, 100]
    for q in qs:
        print(f"  tau p{q:<3} = {np.percentile(tau, q):10.1f} tokens "
              f"= {np.percentile(tau, q)/1000:7.3f} kb")
    tau_max = tau.max()
    print()
    print(f"  slowest single channel: tau_max = {tau_max:.0f} tokens "
          f"({tau_max/1000:.2f} kb)")

    print()
    print("=== comparison against Hi-C bin sizes ===")
    print(f"{'bin':>8} {'tokens/bin':>12} {'tau_max/bin':>13}  verdict")
    for b in ALT_BINS:
        ratio = tau_max / b
        if ratio >= 1.0:
            v = "OK - slowest channel spans a bin"
        elif ratio >= 0.2:
            v = "marginal"
        else:
            v = "F4 RISK - bin invisible to every channel"
        print(f"{b:>8} {b:>12,} {ratio:>13.3f}  {v}")

    frac = (tau >= BIN_BP).mean()
    print()
    print(f"  fraction of (channel, state) pairs with tau >= {BIN_BP} tokens: "
          f"{frac:.4f}")

    print()
    print("=== depth composition (heuristic, NOT a bound) ===")
    print("  A stack relays information: L layers each with horizon tau can")
    print("  propagate roughly L*tau before signal decays. This is a heuristic")
    print("  and is not a substitute for measuring the trained model.")
    for b in ALT_BINS:
        eff = N_LAYER * tau_max
        print(f"    {N_LAYER} x tau_max = {eff:,.0f} tokens "
              f"= {eff/b:6.2f} x a {b//1000} kb bin")

    print()
    print("=== read-out ===")
    if tau_max < BIN_BP:
        print(f"  At init, NO channel has a memory horizon reaching one "
              f"{BIN_BP//1000} kb bin")
        print(f"  (tau_max = {tau_max:.0f} tokens vs {BIN_BP:,} tokens per bin).")
        print("  Within a single layer, s_t is therefore constant across")
        print("  everything the state can see -- the precondition for F4.")
        print()
        print("  Note: Delta is LEARNED, so tau can grow during training, and")
        print("  depth composition extends the effective horizon. This does not")
        print("  prove failure. It does mean the mechanism starts in the")
        print("  degenerate regime and must climb out of it.")
    else:
        print("  Slowest channels span a bin at init -- F4 precondition absent.")


if __name__ == "__main__":
    main()
