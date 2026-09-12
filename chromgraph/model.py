"""The model.

Three things here are the project's contribution and everything else is
deliberately ordinary:

    ContactBiasedAttention   measured contacts shift the attention logit
    EdgeBias                 learned b_HiC and b_dist
    structure dropout        the whole distal graph vanishes with probability p

Two implementation decisions worth knowing about:

  * The local encoder downsamples with a convolution tower BEFORE the
    state-space layers. Running an SSM at single-base resolution over 5,000
    positions per window, 128 windows per sample, is where all the compute goes
    and almost none of the signal is. Akita, Enformer and C.Origami all do the
    same thing.

  * The SSM is a diagonal (S4D-style) model evaluated by FFT convolution in
    pure PyTorch. It is a real state-space model, it is bidirectional, and it
    needs no CUDA kernel -- so there is no `mamba-ssm` dependency to fight with
    on a cluster, and the model runs on a laptop CPU for testing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromgraph.config import Config

N_BASES = 5  # A C G T N


# --------------------------------------------------------------------------- #
# scatter helpers  (sparse attention without a graph library)
# --------------------------------------------------------------------------- #

def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over every group of rows sharing an index.

    scores (E, H) -> (E, H), normalised within each destination node.
    The max subtraction is the same shift-invariance the write-up relies on:
    it changes nothing mathematically and stops exp() overflowing.
    """
    idx = index.unsqueeze(-1).expand_as(scores)
    peak = torch.full((n, scores.shape[-1]), -torch.inf,
                      device=scores.device, dtype=scores.dtype)
    peak = peak.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    peak = torch.nan_to_num(peak, neginf=0.0)
    exp = (scores - peak[index]).exp()
    denom = torch.zeros_like(peak).index_add_(0, index, exp)
    return exp / (denom[index] + 1e-12)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros((n, *src.shape[1:]), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count = torch.zeros(n, device=src.device, dtype=src.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp(min=1).unsqueeze(-1)


# --------------------------------------------------------------------------- #
# local encoder
# --------------------------------------------------------------------------- #

class ConvTower(nn.Module):
    """Strided residual convolutions: (B, C, L) -> (B, d, L / 2**depth).

    This is the compute fix. At bin_size 5000 and depth 5 the state-space
    layers see about 160 positions instead of 5000, which is roughly a 30x
    reduction in the dominant cost of the whole model. Convolutions are also
    what detect sequence motifs, so almost nothing is lost by doing the
    fine-resolution work here.
    """

    def __init__(self, d_model: int, depth: int, width: int = 5):
        super().__init__()
        self.stem = nn.Conv1d(N_BASES, d_model, kernel_size=15, padding=7)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=width, padding=width // 2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            ))
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
            h = F.max_pool1d(h, kernel_size=2, ceil_mode=True)
        return h


class DiagonalSSM(nn.Module):
    """One direction of a diagonal state-space layer, evaluated by FFT.

    For a diagonal SSM with state decay a in (0, 1), the impulse response is
    K[t] = c * a^t * b, so the whole scan is a causal convolution with an
    exponentially decaying kernel. Computing it in the frequency domain is
    O(L log L), exact, and completely stable -- no cumulative products to
    overflow and no custom kernel to install.
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        # a = exp(-exp(log_rate)) keeps the decay strictly inside (0, 1) for any
        # value the optimiser reaches, so the kernel can never diverge.
        self.log_rate = nn.Parameter(torch.linspace(-4.0, 1.0, d_model).unsqueeze(-1)
                                     .repeat(1, d_state))
        self.b = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.c = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.skip = nn.Parameter(torch.ones(d_model))

    def kernel(self, length: int, device, dtype) -> torch.Tensor:
        a = torch.exp(-torch.exp(self.log_rate))                  # (d, n)
        t = torch.arange(length, device=device, dtype=torch.float32)
        decay = a.unsqueeze(-1).clamp(min=1e-6) ** t              # (d, n, L)
        k = (self.b * self.c).unsqueeze(-1) * decay               # (d, n, L)
        return k.sum(1).to(dtype)                                 # (d, L)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, d, L) -> (B, d, L)"""
        length = u.shape[-1]
        k = self.kernel(length, u.device, torch.float32)
        size = 2 * length
        u32 = u.float()
        y = torch.fft.irfft(torch.fft.rfft(u32, n=size) * torch.fft.rfft(k, n=size),
                            n=size)[..., :length]
        return (y + u32 * self.skip.unsqueeze(-1)).to(u.dtype)


class BiSSMBlock(nn.Module):
    """Bidirectional state-space block with a gate, norm and residual.

    Bidirectional because DNA has no reading direction: it is double stranded
    and a motif is a motif read either way, so a causal scan would discard half
    the context for nothing.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = DiagonalSSM(d_model, d_state)
        self.rev = DiagonalSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d)"""
        h = self.norm(x).transpose(1, 2)                          # (B, d, L)
        f = self.fwd(h)
        r = self.rev(h.flip(-1)).flip(-1)
        y = torch.cat([f, r], dim=1).transpose(1, 2)              # (B, L, 2d)
        y = self.out(y) * torch.sigmoid(self.gate(x))
        return x + self.drop(y)


class LocalEncoder(nn.Module):
    """DNA of one window in, one vector out. Sees no other window."""

    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.tower = ConvTower(m.d_model, m.conv_depth)
        self.blocks = nn.ModuleList(
            BiSSMBlock(m.d_model, m.d_state, m.dropout) for _ in range(m.encoder_layers))
        self.norm = nn.LayerNorm(m.d_model)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (N, L) uint8/long -> (N, d)"""
        x = F.one_hot(codes.long(), N_BASES).permute(0, 2, 1).float()
        h = self.tower(x).transpose(1, 2)                          # (N, L', d)
        for block in self.blocks:
            h = block(h)
        return self.norm(h.mean(dim=1))


# --------------------------------------------------------------------------- #
# the contribution
# --------------------------------------------------------------------------- #

class EdgeBias(nn.Module):
    """Turn (contact strength, separation) into a per-head attention bias.

    Two separate heads, on purpose. Keeping b_dist independent of b_HiC is what
    makes the distance-only control constructible: freeze the contact input to
    a constant and b_dist still works, so any loss of performance is
    attributable to contact strength rather than to distance.
    """

    def __init__(self, n_heads: int, hidden: int = 32):
        super().__init__()
        self.hic = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, n_heads))
        self.dist = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, n_heads))

    def forward(self, strength: torch.Tensor, sep: torch.Tensor) -> torch.Tensor:
        c = strength.unsqueeze(-1)
        d = torch.log2(sep.float() + 1.0).unsqueeze(-1)
        return self.hic(c) + self.dist(d)


class ContactBiasedAttention(nn.Module):
    """Multi-head attention restricted to graph edges, with a contact bias.

        a_ij = softmax_{j in N(i)} ( q_i . k_j / sqrt(d_head) + bias_ij )

    Note what is NOT here: no dense N x N attention matrix is ever formed. The
    computation is linear in the number of edges, which is what makes a 640 kb
    context affordable.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        """x (N, d); src/dst (E,); bias (E, H). Returns (N, d)."""
        n = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(n, self.h, self.dh)
        k = k.view(n, self.h, self.dh)
        v = v.view(n, self.h, self.dh)

        if src.numel() == 0:
            # Every node is isolated -- this happens under structure dropout on
            # an arm with no local edges. Attention over nothing is zero, and
            # the residual in the block carries the signal through.
            return self.proj(torch.zeros_like(x))

        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.dh) + bias   # (E, H)
        attn = self.drop(scatter_softmax(score, dst, n))
        msg = attn.unsqueeze(-1) * v[src]                               # (E, H, dh)
        out = torch.zeros_like(v).index_add_(0, dst, msg)
        return self.proj(out.reshape(n, -1))


class Block(nn.Module):
    """attention -> add & norm -> feed forward -> add & norm."""

    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.attn = ContactBiasedAttention(m.d_model, m.n_heads, m.dropout)
        self.n1 = nn.LayerNorm(m.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(m.d_model, m.d_ff), nn.GELU(),
            nn.Dropout(m.dropout), nn.Linear(m.d_ff, m.d_model))
        self.n2 = nn.LayerNorm(m.d_model)
        self.drop = nn.Dropout(m.dropout)

    def forward(self, x, src, dst, bias):
        x = self.n1(x + self.drop(self.attn(x, src, dst, bias)))
        return self.n2(x + self.drop(self.ffn(x)))


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #

class ContactHead(nn.Module):
    """Predict a held-out edge's strength from the two node representations."""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, zi: torch.Tensor, zj: torch.Tensor, sep: torch.Tensor):
        d = torch.log2(sep.float() + 1.0).unsqueeze(-1)
        # Symmetric features: a contact has no direction, so the head must not
        # be able to tell (i, j) from (j, i).
        feats = torch.cat([zi * zj, (zi - zj).abs(), d], dim=-1)
        return self.net(feats).squeeze(-1)


@dataclass
class ModelOutput:
    z: torch.Tensor
    h: torch.Tensor
    masked_nodes: torch.Tensor
    h_recon: torch.Tensor
    dropped_structure: torch.Tensor


class ChromGraphFM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.encoder = LocalEncoder(cfg)
        self.edge_bias = EdgeBias(m.n_heads)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(m.block_layers))

        # Learned stand-in for a masked window, the same idea as a [MASK] token.
        self.mask_token = nn.Parameter(torch.zeros(m.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.recon = nn.Sequential(
            nn.Linear(m.d_model, m.d_ff), nn.GELU(), nn.Linear(m.d_ff, m.d_model))
        self.contact_head = ContactHead(m.d_model)
        self.project = nn.Sequential(
            nn.Linear(m.d_model, m.d_model), nn.GELU(),
            nn.Linear(m.d_model, m.d_model))

        # Late fusion (arm B4): Hi-C summary statistics are concatenated AFTER
        # the encoder has finished, so structure cannot have shaped the
        # representation -- which is exactly the point of the control.
        self.late_fusion = nn.Linear(m.d_model + 3, m.d_model)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch: dict, mask_frac: float = 0.0,
                use_structure: bool = True) -> ModelOutput:
        seq = batch["seq"]                       # (B, N, L)
        b, n, _ = seq.shape
        total = b * n

        h = self.encoder(seq.reshape(total, -1))                 # (B*N, d)

        masked = torch.zeros(total, dtype=torch.bool, device=h.device)
        if mask_frac > 0 and self.training:
            masked = torch.rand(total, device=h.device) < mask_frac
            h = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(h), h)

        src, dst, strength, sep, dropped = self._edges(batch, b, n, use_structure)
        bias = self.edge_bias(strength, sep) if src.numel() else \
            torch.zeros(0, self.cfg.model.n_heads, device=h.device, dtype=h.dtype)

        z = h
        for block in self.blocks:
            z = block(z, src, dst, bias)

        if batch.get("late_fusion", False):
            z = self.late_fusion(torch.cat([z, batch["node_hic_feat"].reshape(total, 3)], -1))

        return ModelOutput(z=z.view(b, n, -1), h=h.view(b, n, -1),
                           masked_nodes=masked.view(b, n),
                           h_recon=self.recon(z).view(b, n, -1),
                           dropped_structure=dropped)

    def _edges(self, batch: dict, b: int, n: int, use_structure: bool):
        """Flatten the batch's edges into one graph, applying structure dropout.

        Structure dropout removes the DISTAL edges only. Local edges are the
        sequential backbone; dropping them too would leave isolated nodes and
        test something other than "can it work without Hi-C".
        """
        idx = batch["edge_index"]            # (B, 2, E) padded
        strength = batch["edge_strength"]    # (B, E)
        sep = batch["edge_sep"]              # (B, E)
        valid = batch["edge_mask"]           # (B, E) bool
        is_local = batch["edge_is_local"]    # (B, E) bool

        p = self.cfg.model.structure_dropout
        if self.training and use_structure and p > 0:
            dropped = torch.rand(b, device=idx.device) < p
        else:
            dropped = torch.zeros(b, dtype=torch.bool, device=idx.device)
        if not use_structure:
            dropped = torch.ones(b, dtype=torch.bool, device=idx.device)

        keep = valid & ~(dropped.unsqueeze(-1) & ~is_local)

        offset = (torch.arange(b, device=idx.device) * n).view(b, 1)
        i = (idx[:, 0] + offset)[keep]
        j = (idx[:, 1] + offset)[keep]
        s = strength[keep]
        d = sep[keep]

        # Both directions: an edge means the two windows may exchange
        # information, not that one talks to the other.
        src = torch.cat([i, j])
        dst = torch.cat([j, i])
        return src, dst, torch.cat([s, s]), torch.cat([d, d]), dropped


def build_model(cfg: Config) -> ChromGraphFM:
    model = ChromGraphFM(cfg)
    print(f"  model: {model.n_parameters() / 1e6:.2f}M parameters")
    return model
