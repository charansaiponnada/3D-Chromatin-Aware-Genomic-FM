"""Parameter accounting for the structural-bias mechanism (architecture_spec.md 4.1.2).

Instantiates the baseline BiMamba-class backbone and the structurally-biased
variant as real nn.Modules and counts parameter tensors directly. Only the
parameter *set* is modelled here -- the selective-scan kernel is irrelevant to
a parameter count, so this runs on CPU with no mamba-ssm dependency.

Run:  python scripts/param_accounting.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 256
    n_layer: int = 16
    d_state: int = 16       # N
    d_conv: int = 4
    expand: int = 2         # E
    vocab_size: int = 16    # char-level DNA + specials
    d_struct: int = 2       # d_s: insulation + directionality
    d_struct_raw: int = 8   # raw per-bin Hi-C features before the encoder
    d_struct_hidden: int = 16

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def dt_rank(self) -> int:
        return math.ceil(self.d_model / 16)


class MambaInner(nn.Module):
    """Direction-specific parameters of a Mamba block.

    Excludes in_proj/out_proj, which BiMamba shares across directions.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.conv1d = nn.Conv1d(
            c.d_inner, c.d_inner, kernel_size=c.d_conv,
            groups=c.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(c.d_inner, c.dt_rank + 2 * c.d_state, bias=False)
        self.dt_proj = nn.Linear(c.dt_rank, c.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(c.d_inner, c.d_state))
        self.D = nn.Parameter(torch.empty(c.d_inner))


class StructuralBias(nn.Module):
    """Additive pre-softplus bias on Delta, plus a scalar permeability gate.

    W_dstruct has no bias term: dt_proj already carries b_dt, and a second
    additive constant would be exactly redundant with it (see failure mode F1).
    """

    def __init__(self, c: Config):
        super().__init__()
        self.W_dstruct = nn.Linear(c.d_struct, c.d_inner, bias=False)
        self.w_gate = nn.Linear(c.d_struct, 1, bias=True)


class BiMambaLayer(nn.Module):
    def __init__(self, c: Config, structural: bool):
        super().__init__()
        self.in_proj = nn.Linear(c.d_model, 2 * c.d_inner, bias=False)   # shared
        self.out_proj = nn.Linear(c.d_inner, c.d_model, bias=False)      # shared
        self.fwd = MambaInner(c)
        self.rev = MambaInner(c)
        self.norm = nn.RMSNorm(c.d_model)
        if structural:
            # Separate per direction: the directionality index is antisymmetric
            # under reverse-complement, so the two directions must be free to
            # learn opposite signs (see failure mode F7).
            self.struct_fwd = StructuralBias(c)
            self.struct_rev = StructuralBias(c)


class StructuralEncoder(nn.Module):
    """Shared across layers: raw per-bin Hi-C features -> s_t in R^{d_struct}."""

    def __init__(self, c: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c.d_struct_raw, c.d_struct_hidden),
            nn.GELU(),
            nn.Linear(c.d_struct_hidden, c.d_struct),
        )


class Backbone(nn.Module):
    def __init__(self, c: Config, structural: bool):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.d_model)  # tied with LM head
        self.layers = nn.ModuleList(
            BiMambaLayer(c, structural) for _ in range(c.n_layer)
        )
        self.norm_f = nn.RMSNorm(c.d_model)
        if structural:
            self.struct_encoder = StructuralEncoder(c)


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    c = Config()
    baseline = Backbone(c, structural=False)
    structural = Backbone(c, structural=True)

    n_base = count(baseline)
    n_struct = count(structural)
    added = n_struct - n_base
    pct = 100.0 * added / n_base

    print(f"config: d_model={c.d_model} n_layer={c.n_layer} d_state={c.d_state} "
          f"d_inner={c.d_inner} dt_rank={c.dt_rank} d_struct={c.d_struct}")
    print()
    print("--- per-layer breakdown (baseline) ---")
    layer = baseline.layers[0]
    for name in ("in_proj", "out_proj", "fwd", "rev", "norm"):
        print(f"  {name:<10} {count(getattr(layer, name)):>10,}")
    print(f"  {'LAYER':<10} {count(layer):>10,}")
    print()
    print("--- per-layer added by structural bias ---")
    slayer = structural.layers[0]
    for name in ("struct_fwd", "struct_rev"):
        sub = getattr(slayer, name)
        print(f"  {name:<12} {count(sub):>8,}  "
              f"(W_dstruct {count(sub.W_dstruct):,} + w_gate {count(sub.w_gate):,})")
    print(f"  {'per layer':<12} {count(slayer) - count(layer):>8,}")
    print(f"  struct_encoder (shared, once) {count(structural.struct_encoder):,}")
    print()
    print("--- totals ---")
    print(f"  baseline            {n_base:>12,}")
    print(f"  structural-bias     {n_struct:>12,}")
    print(f"  added               {added:>12,}")
    print(f"  delta               {pct:>11.4f}%")
    print()
    verdict = "PASS" if abs(pct) <= 5.0 else "FAIL"
    print(f"  matched-parameter constraint (<=5%): {verdict}")


if __name__ == "__main__":
    main()
