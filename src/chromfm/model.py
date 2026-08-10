"""BiMamba backbone: the Phase 3 baseline and the Phase 4 structural variant.

The two models differ only by the structural pathway described in
docs/architecture_spec.md 4.1. Parameter counts are asserted against
scripts/param_accounting.py so the matched-compute claim stays true.

SCAN BACKEND
------------
Two implementations, selected by device:

  * `selective_scan_ref` below -- a sequential loop over positions. Correct and
    differentiable, and what the CPU unit tests run at small L. NOT viable at
    L=32768 (32768 x 16 layers x 2 directions Python iterations per forward).

  * `chromfm.scan_triton.selective_scan_triton` -- a custom Triton kernel that
    carries the permeability term p, used automatically on CUDA.

The custom kernel exists because mamba_ssm's fused scan cannot express p: it
computes Abar = exp(delta * A), and folding p in would need
delta' = delta + log(g)/A[n], which depends on the state index n. A per-channel
delta cannot produce a decay uniform across n. The Delta bias needs no kernel
work; only p does.

Both arms of the experiment must run the same backend or the matched-compute
comparison is confounded, so the baseline uses this scan too even though it
never supplies p.

    !!  The Triton kernel is UNVERIFIED until scripts/validate_kernel.py is run
    on a GPU. Never train on an unvalidated scan: a subtly wrong gradient
    produces a plausible loss curve and silently invalidates the experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layer: int = 16
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    vocab_size: int = 16
    d_struct: int = 2
    d_struct_raw: int = 8
    d_struct_hidden: int = 16
    structural: bool = False
    use_permeability: bool = True     # the p_t term; see KERNEL NOTE

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def dt_rank(self) -> int:
        return math.ceil(self.d_model / 16)


def selective_scan_ref(u, delta, A, B, C, D, p=None):
    """Reference selective scan. Sequential in time; use only at small L.

    u, delta : (b, d, l)
    A        : (d, n)          negative
    B, C     : (b, n, l)
    D        : (d,)
    p        : (b, l) or None  non-negative extra log-decay, uniform over d and n
    returns  : (b, d, l)
    """
    b, d, l = u.shape
    n = A.shape[1]
    # A is (d, n); it must broadcast against (b, d, l, 1) on the *channel* axis,
    # so it needs an explicit (1, d, 1, n) view. Relying on trailing-axis
    # broadcasting silently aligns n against l and only agrees when l == d_inner.
    dA = torch.exp(delta.unsqueeze(-1) * A.view(1, d, 1, n))   # (b, d, l, n)
    if p is not None:
        dA = dA * torch.exp(-p)[:, None, :, None]
    dBu = delta.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1) * u.unsqueeze(-1)

    h = u.new_zeros((b, d, n))
    ys = []
    for t in range(l):
        h = dA[:, :, t] * h + dBu[:, :, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, t]))
    y = torch.stack(ys, dim=-1)
    return y + u * D.view(1, d, 1)


_FORCE_SCAN: str | None = None      # "ref" or "triton"; set by validate_kernel.py


def use_scan(backend: str | None) -> None:
    """Pin the scan backend. None restores automatic device-based selection."""
    global _FORCE_SCAN
    if backend not in (None, "ref", "triton"):
        raise ValueError(f"unknown scan backend: {backend}")
    _FORCE_SCAN = backend


def _pick_scan(device):
    if _FORCE_SCAN == "ref":
        return selective_scan_ref
    from . import scan_triton
    if _FORCE_SCAN == "triton":
        return scan_triton.selective_scan_triton
    if scan_triton.available(device):
        return scan_triton.selective_scan_triton
    return selective_scan_ref


class MambaDirection(nn.Module):
    """One direction of a BiMamba layer. Owns everything except in/out proj."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.conv1d = nn.Conv1d(c.d_inner, c.d_inner, c.d_conv,
                                groups=c.d_inner, padding=c.d_conv - 1)
        self.x_proj = nn.Linear(c.d_inner, c.dt_rank + 2 * c.d_state, bias=False)
        self.dt_proj = nn.Linear(c.dt_rank, c.d_inner, bias=True)

        A = torch.arange(1, c.d_state + 1, dtype=torch.float32).repeat(c.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(c.d_inner))

        self._init_dt()

        if c.structural:
            # no bias: dt_proj already carries b_dt, a second constant is
            # exactly redundant with it (failure mode F1)
            self.W_dstruct = nn.Linear(c.d_struct, c.d_inner, bias=False)
            nn.init.zeros_(self.W_dstruct.weight)
            if c.use_permeability:
                self.w_gate = nn.Linear(c.d_struct, 1, bias=True)
                nn.init.zeros_(self.w_gate.weight)
                # b_g trades init-equivalence against gradient scale: the
                # gradient through softplus is sigmoid(b_g), so b_g = -8 gives
                # p = 3.4e-4 but attenuates learning by ~3000x. b_g = -4 keeps p
                # negligible (0.018) with 53x more gradient. Measured in
                # scripts/test_model.py; failure mode F2.
                nn.init.constant_(self.w_gate.bias, -4.0)

    def _init_dt(self, dt_min=1e-3, dt_max=1e-1, floor=1e-4):
        c = self.c
        nn.init.uniform_(self.dt_proj.weight, -c.dt_rank ** -0.5, c.dt_rank ** -0.5)
        dt = torch.exp(
            torch.rand(c.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, s=None):
        """x: (b, l, d_inner) already gated upstream. s: (b, l, d_struct) or None."""
        b, l, _ = x.shape
        xc = self.conv1d(x.transpose(1, 2))[..., :l]
        xc = F.silu(xc).transpose(1, 2)                       # (b, l, d_inner)

        proj = self.x_proj(xc)
        dt, B, C = torch.split(
            proj, [self.c.dt_rank, self.c.d_state, self.c.d_state], dim=-1)
        dt_pre = self.dt_proj(dt)                             # (b, l, d_inner)

        p = None
        if self.c.structural and s is not None:
            dt_pre = dt_pre + self.W_dstruct(s)
            if self.c.use_permeability:
                p = F.softplus(self.w_gate(s).squeeze(-1))    # (b, l)

        delta = F.softplus(dt_pre)
        A = -torch.exp(self.A_log)
        scan = _pick_scan(xc.device)
        return scan(
            xc.transpose(1, 2), delta.transpose(1, 2), A,
            B.transpose(1, 2), C.transpose(1, 2), self.D, p,
        ).transpose(1, 2)                                      # (b, l, d_inner)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class BiMambaLayer(nn.Module):
    """in_proj and out_proj are shared across directions, as in Caduceus."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(c.d_model)
        self.in_proj = nn.Linear(c.d_model, 2 * c.d_inner, bias=False)
        self.out_proj = nn.Linear(c.d_inner, c.d_model, bias=False)
        self.fwd = MambaDirection(c)
        self.rev = MambaDirection(c)
        self.c = c

    def forward(self, u, s=None, s_rev=None):
        x, z = self.in_proj(self.norm(u)).chunk(2, dim=-1)
        gate = F.silu(z)
        y_f = self.fwd(x, s)
        y_r = self.rev(x.flip(1), s_rev).flip(1)
        return u + self.out_proj((y_f + y_r) * gate)


class StructuralEncoder(nn.Module):
    """Raw per-bin Hi-C features -> the low-dimensional s_t. Shared across layers."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c.d_struct_raw, c.d_struct_hidden),
            nn.GELU(),
            nn.Linear(c.d_struct_hidden, c.d_struct),
        )

    def forward(self, phi):
        return self.net(phi)


class BiMambaLM(nn.Module):
    """Masked-language-model over nucleotides, optionally structure-conditioned."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(BiMambaLayer(c) for _ in range(c.n_layer))
        self.norm_f = RMSNorm(c.d_model)
        if c.structural:
            self.struct_encoder = StructuralEncoder(c)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight          # tied

    def forward(self, tokens, phi=None, phi_valid=None, symmetry=None):
        """tokens (b,l) int64; phi (b,l,d_struct_raw); phi_valid (b,l) bool."""
        u = self.embed(tokens)
        s = s_rev = None
        if self.c.structural:
            if phi is None:
                raise ValueError("structural model called without phi")
            s = self.struct_encoder(phi)

            # The reverse pass sees the window reversed, so its structural input
            # must be reversed too, with the antisymmetric coordinates of phi
            # negated (failure mode F7).
            #
            # The sign flip belongs on the RAW features, not on the encoded s:
            # the encoder mixes all eight coordinates into d_struct dimensions,
            # so there is no correspondence left to flip afterwards. That means
            # a second encoder pass, which is cheap -- the encoder is 178 params.
            phi_rev = phi.flip(1)
            if symmetry is not None:
                phi_rev = phi_rev * symmetry.view(1, 1, -1).to(phi.dtype)
            s_rev = self.struct_encoder(phi_rev)

            if phi_valid is not None:
                m = phi_valid.unsqueeze(-1).to(s.dtype)
                s = s * m
                s_rev = s_rev * m.flip(1)
        for layer in self.layers:
            u = layer(u, s, s_rev)
        return self.lm_head(self.norm_f(u))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def tau_stats(self) -> dict:
        """Memory horizon in tokens, per layer, from the current dt bias and A.

        This is the measurement architecture_spec.md 4.1.4 F4 makes a Phase 4
        gate: if trained tau never approaches TAD scale, structure-at-TAD-scale
        is not expressible at this model size.
        """
        out = {}
        for i, layer in enumerate(self.layers):
            for name in ("fwd", "rev"):
                d = getattr(layer, name)
                with torch.no_grad():
                    delta = F.softplus(d.dt_proj.bias)          # (d_inner,)
                    A = torch.exp(d.A_log)                      # |A|, (d_inner, n)
                    tau = 1.0 / (delta[:, None] * A)
                out[f"layer{i}.{name}"] = {
                    "tau_median": float(tau.median()),
                    "tau_p99": float(tau.flatten().quantile(0.99)),
                    "tau_max": float(tau.max()),
                }
        return out


def build(structural: bool = False, **kw) -> BiMambaLM:
    return BiMambaLM(ModelConfig(structural=structural, **kw))
