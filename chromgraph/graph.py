"""Phase 2 --- turn a contact band into sparse graphs, and corrupt them for the controls.

One function builds the graph; one function corrupts it. Every experimental arm
differs only in which corruption is applied, which is what makes the comparison
a comparison: same code path, same shapes, same budget.

Two invariants are enforced here rather than hoped for:

  * target edges never appear in the conditioning graph, or the contact head
    could read an edge from its own input and echo it back;
  * corruptions draw from `control_seed`, never the training seed, so every
    training seed sees the identical corrupted graph and seed variance can
    never be confused with corruption variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chromgraph.config import Config

ARMS = ("b0_dna_only", "b1_random_graph", "b2_distance_only", "b3_shuffled_hic",
        "b4_late_fusion", "b5_contrastive_only", "full")

# Arms whose encoder receives no distal structure at all. B4 fuses Hi-C after
# the encoder and B5 uses it only as an alignment target, so neither biases
# attention -- they differ from B0 in the head, not in the graph.
NO_DISTAL_ARMS = frozenset({"b0_dna_only", "b4_late_fusion", "b5_contrastive_only"})


@dataclass
class Sample:
    """One training example: a contiguous run of windows plus its graph."""

    cell_line: str
    chrom: str
    start_bin: int
    n_nodes: int

    seq: np.ndarray            # (n_nodes, bin_size) uint8
    node_ok: np.ndarray        # (n_nodes,) bool -- usable bins

    edge_index: np.ndarray     # (2, E) int64, local coordinates, i < j
    edge_strength: np.ndarray  # (E,) float32 in (0, 1)
    edge_sep: np.ndarray       # (E,) int64, |i - j| in bins
    edge_is_local: np.ndarray  # (E,) bool

    tgt_index: np.ndarray      # (2, T) int64 -- held out, never conditioned on
    tgt_strength: np.ndarray   # (T,) float32

    # Per-node Hi-C summary (sum, max, count of distal strengths), computed
    # from the UNCORRUPTED graph. Used only by the late-fusion arm, where Hi-C
    # must reach the model without ever touching the encoder.
    node_hic_feat: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))

    def assert_disjoint(self) -> None:
        """The invariant the whole contact objective rests on."""
        if self.tgt_index.size == 0 or self.edge_index.size == 0:
            return
        cond = {(int(a), int(b)) for a, b in self.edge_index.T}
        tgt = {(int(a), int(b)) for a, b in self.tgt_index.T}
        overlap = cond & tgt
        if overlap:
            raise AssertionError(
                f"{len(overlap)} target edges are also conditioning edges, e.g. "
                f"{sorted(overlap)[:3]}. The contact head could copy its input.")


# --------------------------------------------------------------------------- #
# graph construction
# --------------------------------------------------------------------------- #

def build_sample(cfg: Config, chrom_data: dict, cell_line: str, chrom: str,
                 start_bin: int, arm: str = "full",
                 rng: np.random.Generator | None = None) -> Sample:
    """Extract one window-run and build its conditioning + target edge sets."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    n = cfg.data.nodes_per_sample
    stop = start_bin + n
    seq = np.asarray(chrom_data["seq"][start_bin:stop])
    node_ok = chrom_data["usable"][start_bin:stop].copy()

    # Pixels whose BOTH ends fall inside this window, re-indexed to local coords.
    row, col = chrom_data["row"], chrom_data["col"]
    inside = (row >= start_bin) & (col < stop)
    li = (row[inside] - start_bin).astype(np.int64)
    lj = (col[inside] - start_bin).astype(np.int64)
    ls = chrom_data["strength"][inside].astype(np.float32)

    # Drop any edge touching an unusable bin -- an ICE-filtered or all-N window
    # carries no information and would just add noise to the attention.
    ok = node_ok[li] & node_ok[lj]
    li, lj, ls = li[ok], lj[ok], ls[ok]

    control_rng = np.random.default_rng(cfg.control_seed + start_bin)

    local_i, local_j, local_s = _local_edges(cfg, n, li, lj, ls)
    distal_i, distal_j, distal_s = _topk_distal(cfg, n, li, lj, ls)

    # Hold out targets from the DISTAL edges only. Local edges are structural
    # scaffolding rather than measurements, and holding them out would just
    # punch holes in the chain.
    node_feat = _node_hic_features(n, distal_i, distal_j, distal_s)

    tgt_i, tgt_j, tgt_s, distal_i, distal_j, distal_s = _split_targets(
        cfg, distal_i, distal_j, distal_s, control_rng)

    # B1 invents edges, so it must be told which pairs are spoken for. Without
    # this it can place a random edge exactly on a held-out target and hand the
    # contact head the answer it is supposed to predict.
    forbidden = {(int(a), int(b)) for a, b in zip(tgt_i, tgt_j)}
    distal_i, distal_j, distal_s = apply_control(
        arm, cfg, n, distal_i, distal_j, distal_s, control_rng, forbidden)

    edge_i = np.concatenate([local_i, distal_i])
    edge_j = np.concatenate([local_j, distal_j])
    edge_s = np.concatenate([local_s, distal_s])
    is_local = np.concatenate([np.ones(local_i.size, bool),
                               np.zeros(distal_i.size, bool)])

    sample = Sample(
        cell_line=cell_line, chrom=chrom, start_bin=start_bin, n_nodes=n,
        seq=seq, node_ok=node_ok,
        edge_index=np.stack([edge_i, edge_j]),
        edge_strength=edge_s.astype(np.float32),
        edge_sep=np.abs(edge_j - edge_i),
        edge_is_local=is_local,
        tgt_index=np.stack([tgt_i, tgt_j]) if tgt_i.size else np.zeros((2, 0), np.int64),
        tgt_strength=tgt_s.astype(np.float32),
        node_hic_feat=node_feat,
    )
    sample.assert_disjoint()
    return sample


def _node_hic_features(n: int, di, dj, ds) -> np.ndarray:
    """(n, 3): summed strength, max strength, edge count, per node."""
    feat = np.zeros((n, 3), np.float32)
    if di.size:
        for arr in (di, dj):
            np.add.at(feat[:, 0], arr, ds)
            np.maximum.at(feat[:, 1], arr, ds)
            np.add.at(feat[:, 2], arr, 1.0)
    return feat


def _local_edges(cfg: Config, n: int, li, lj, ls):
    """Sequential neighbours within +/- local_radius, always present.

    Their strength comes from the measured map where a pixel exists and falls
    back to a neutral 0.5 where it does not, so the chain is never broken by a
    missing measurement.
    """
    have = {}
    m = (lj - li) <= cfg.data.local_radius
    for a, b, s in zip(li[m], lj[m], ls[m]):
        have[(int(a), int(b))] = float(s)

    ii, jj, ss = [], [], []
    for i in range(n):
        for d in range(1, cfg.data.local_radius + 1):
            j = i + d
            if j >= n:
                break
            ii.append(i)
            jj.append(j)
            ss.append(have.get((i, j), 0.5))
    return (np.asarray(ii, np.int64), np.asarray(jj, np.int64),
            np.asarray(ss, np.float32))


def _topk_distal(cfg: Config, n: int, li, lj, ls):
    """Keep each node's strongest partners at least min_separation away.

    Selection is on DETRENDED strength. On raw counts this would return the
    nearest bins every time and the model would learn distance, not structure.
    """
    far = (lj - li) >= cfg.data.min_separation
    fi, fj, fs = li[far], lj[far], ls[far]
    if fi.size == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32)

    # Both endpoints get a vote, so an edge survives if either node ranks it.
    keep = np.zeros(fi.size, bool)
    both = np.concatenate([fi, fj])
    other_order = np.concatenate([np.arange(fi.size), np.arange(fi.size)])
    strengths = np.concatenate([fs, fs])

    order = np.lexsort((-strengths, both))
    both, other_order = both[order], other_order[order]
    starts = np.searchsorted(both, np.arange(n), side="left")
    stops = np.searchsorted(both, np.arange(n), side="right")
    for node in range(n):
        chosen = other_order[starts[node]: starts[node] + min(
            cfg.data.top_k_edges, stops[node] - starts[node])]
        keep[chosen] = True

    ki, kj, ks = fi[keep], fj[keep], fs[keep]
    # Deduplicate: an edge picked by both endpoints appears once.
    key = ki * n + kj
    _, uniq = np.unique(key, return_index=True)
    return ki[uniq], kj[uniq], ks[uniq]


def _split_targets(cfg: Config, di, dj, ds, rng: np.random.Generator):
    """Withhold a fraction of distal edges as contact-prediction targets."""
    if di.size == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32), di, dj, ds
    mask = rng.random(di.size) < cfg.data.held_out_edge_frac
    return di[mask], dj[mask], ds[mask], di[~mask], dj[~mask], ds[~mask]


# --------------------------------------------------------------------------- #
# the controls
# --------------------------------------------------------------------------- #

def apply_control(arm: str, cfg: Config, n: int, di, dj, ds,
                  rng: np.random.Generator,
                  forbidden: set[tuple[int, int]] | None = None):
    """Corrupt the distal edges according to the experimental arm.

    Every arm keeps the same NUMBER of distal edges wherever possible. A
    control that also changed the edge count would confound "wrong structure"
    with "less structure", and the comparison would mean nothing.
    """
    if arm in NO_DISTAL_ARMS:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32)

    if arm == "b1_random_graph":
        # Same count, endpoints drawn uniformly subject to the separation rule
        # and to never colliding with a held-out target.
        return _random_edges(cfg, n, di.size, rng, ds, forbidden or set())

    if arm == "b2_distance_only":
        # Real topology kept, contact strengths flattened to a constant. The
        # model can still see how far apart two windows are -- it just cannot
        # see how strongly they touch. Isolates distance from structure.
        return di, dj, np.full(ds.size, 0.5, np.float32)

    if arm == "b3_shuffled_hic":
        # Real topology, real strength VALUES, permuted between edges. Same
        # marginal distribution, destroyed pairing. This is the decisive arm.
        return di, dj, ds[rng.permutation(ds.size)]

    if arm == "full":
        return di, dj, ds

    raise ValueError(f"no control defined for arm {arm!r}")


def _random_edges(cfg: Config, n: int, count: int, rng: np.random.Generator,
                  donor: np.ndarray, forbidden: set[tuple[int, int]]):
    """Uniformly random edges obeying min_separation, with borrowed strengths."""
    if count == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32)
    seen: set[tuple[int, int]] = set()
    ii, jj = [], []
    guard = 0
    while len(ii) < count and guard < count * 50:
        guard += 1
        a, b = rng.integers(0, n, 2)
        lo, hi = (int(a), int(b)) if a < b else (int(b), int(a))
        if hi - lo < cfg.data.min_separation or (lo, hi) in seen:
            continue
        if (lo, hi) in forbidden:
            continue
        seen.add((lo, hi))
        ii.append(lo)
        jj.append(hi)
    k = len(ii)
    strengths = donor[rng.integers(0, donor.size, k)] if donor.size else np.full(k, 0.5, np.float32)
    return (np.asarray(ii, np.int64), np.asarray(jj, np.int64),
            strengths.astype(np.float32))


# --------------------------------------------------------------------------- #
# distance-matched negatives for the contrastive objective
# --------------------------------------------------------------------------- #

def distance_matched_negatives(n: int, pos_i: int, pos_j: int, count: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Other pairs spanning exactly the same number of bins as the positive.

    Matching on the PAIR's separation, not on a shared anchor. Sampled freely,
    negatives would sit further apart than positives on average and the model
    could minimise the loss by measuring distance -- which it can already read
    off the input.
    """
    sep = abs(pos_j - pos_i)
    starts = np.arange(0, n - sep, dtype=np.int64)
    starts = starts[starts != min(pos_i, pos_j)]
    if starts.size == 0:
        return np.zeros((2, 0), np.int64)
    if starts.size > count:
        starts = rng.choice(starts, size=count, replace=False)
    return np.stack([starts, starts + sep])


def sample_starts(n_bins: int, nodes_per_sample: int, usable: np.ndarray,
                  stride: int | None = None, min_usable: float = 0.6) -> list[int]:
    """Window start positions whose span is mostly usable bins."""
    stride = stride or max(1, nodes_per_sample // 2)
    out = []
    for start in range(0, n_bins - nodes_per_sample + 1, stride):
        if usable[start: start + nodes_per_sample].mean() >= min_usable:
            out.append(start)
    return out
