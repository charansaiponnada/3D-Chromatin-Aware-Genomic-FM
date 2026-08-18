"""Phase 1, step 4 -- build the training dataset layer.

Turns the pilot chromosome into what Phase 3 and Phase 4 actually consume:
tokenised sequence, a window index split into train/val/test, and structural
features that interpolate to token resolution on the fly.

Three decisions are implemented here that the docs flagged as required:

* **phi is interpolated, never stepped.** architecture_spec.md 4.1.4 F4 measured
  the slowest memory horizon at 994 tokens against a 5,000-token bin. A step
  function would hold s_t constant across everything a layer's state can see,
  which is the precondition for the mechanism collapsing into F1. Linear
  interpolation between bin centres makes s_t vary at every position.
* **N gets its own token and an explicit budget.** 12.00% of chr9 is unresolved.
  The policy must be identical for the baseline and structural arms or the
  comparison is confounded, so it lives here rather than in either training script.
* **Invalid bins are masked, never NaN.** 19.31% of bins carry no balancing
  weight. A NaN reaching W_dstruct . s_t would poison the whole recurrence.

phi is NOT materialised per token: that would be 138M x 8 floats. It stays at bin
resolution and WindowDataset interpolates inside __getitem__.

Run:  python scripts/phase1_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"

CHROM = "chr9"
RES = 5_000

# Window length. The F4 analysis puts the 16-layer relayed horizon near 16,000
# tokens, so 32,768 already exceeds what the stack can span -- going longer buys
# context the architecture cannot use while costing memory linearly.
# 2026-08-17: widened 32,768 -> 65,536. keep(phi) measured 0.0490 at 32,768 vs
# 0.1099 at 65,536 (p5_window_scan.json), i.e. 2.24x the within-window
# structural signal the delta-bias mechanism can act on, at a measured 2.08x
# step cost (phase5_memcheck.py, 7.5 of 44.39 GiB peak). The 32,768 index is
# preserved as dataset_index_w32768.npz -- every Phase 3/4/P5 number was
# measured against that file and must not be re-derived against this one.
WINDOW = 65_536
STRIDE_TRAIN = 32_768          # 50% overlap for training coverage
STRIDE_EVAL = 65_536           # no overlap in val/test

MAX_N_FRAC = 0.10              # drop a window if >10% of its sequence is N
MIN_STRUCT_FRAC = 0.50         # drop a window if <50% of positions have usable phi

# Held-out regions, chosen on the two usable arms of chr9. The centromere and
# 9q12 heterochromatin (43.27-60.52 Mb) are excluded automatically by the phi
# validity mask, so they need no special case here.
TEST_REGION = (130_000_000, 138_394_717)   # distal 9q; contains the validated NOTCH1 loop
VAL_REGION = (120_000_000, 129_000_000)
SPLIT_BUFFER = 1_000_000       # dead zone around held-out regions, >= one window

# vocabulary -- ids 0-6 used, 7-15 reserved (param_accounting.py assumes 16)
PAD, MASK, A, C, G, T, N_TOK = 0, 1, 2, 3, 4, 5, 6
VOCAB = {"A": A, "C": C, "G": G, "T": T, "N": N_TOK}
COMPLEMENT = {A: T, T: A, C: G, G: C, N_TOK: N_TOK, PAD: PAD, MASK: MASK}


def tokenise() -> np.ndarray:
    """chr9 FASTA -> int8 token array. Anything not ACGT becomes the N token."""
    path = INTERIM / f"{CHROM}.fa"
    with path.open() as fh:
        fh.readline()
        seq = fh.read().replace("\n", "").upper()
    arr = np.frombuffer(seq.encode(), dtype="S1")
    tokens = np.full(len(arr), N_TOK, dtype=np.int8)
    for base, tok in VOCAB.items():
        tokens[arr == base.encode()] = tok
    counts = {b: int((tokens == t).sum()) for b, t in VOCAB.items()}
    total = len(tokens)
    print(f"  tokenised {total:,} positions")
    for b, c in counts.items():
        print(f"    {b}  {c:>12,}  {100*c/total:6.2f}%")
    return tokens


def load_phi() -> dict:
    z = np.load(PROCESSED / f"phi_{CHROM}_{RES}bp.npz", allow_pickle=True)
    phi = z["phi"].astype(np.float32)          # standardised, NaN on unusable bins
    usable = z["usable"].astype(bool)
    phi = np.nan_to_num(phi, nan=0.0)          # 0 == the standardised mean
    print(f"  phi {phi.shape}, usable bins {usable.sum():,}/{len(usable):,} "
          f"({100*usable.mean():.2f}%)")
    return {
        "phi": phi,
        "usable": usable,
        "names": [str(x) for x in z["feature_names"]],
        "symmetry": z["feature_symmetry"].astype(np.int8),
    }


def interpolate_phi(phi: np.ndarray, usable: np.ndarray,
                    positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation between bin centres. Returns (phi_t, valid_t).

    Bin b covers [b*RES, (b+1)*RES); its centre is b*RES + RES/2. A position is
    structurally valid only when both flanking bins are usable -- interpolating
    across a boundary into the centromere would invent signal.
    """
    n_bins = phi.shape[0]
    u = (positions.astype(np.float64) - RES / 2.0) / RES
    i0 = np.floor(u).astype(np.int64)
    frac = (u - i0).astype(np.float32)
    i1 = i0 + 1
    np.clip(i0, 0, n_bins - 1, out=i0)
    np.clip(i1, 0, n_bins - 1, out=i1)
    out = (1.0 - frac)[:, None] * phi[i0] + frac[:, None] * phi[i1]
    valid = usable[i0] & usable[i1]
    out[~valid] = 0.0
    return out.astype(np.float32), valid


def assign_split(start: int, end: int) -> str | None:
    """Which split a window belongs to, or None if it lands in a buffer.

    Held-out windows must sit *entirely* inside their region. A window that only
    straddles the edge is discarded rather than assigned, so every split is an
    exact genomic interval and no evaluation window carries sequence from the
    buffer zone.
    """
    def contains(region):
        return start >= region[0] and end <= region[1]

    def overlaps(region, pad=0):
        return start < region[1] + pad and end > region[0] - pad

    if contains(TEST_REGION):
        return "test"
    if contains(VAL_REGION):
        return "val"
    if overlaps(TEST_REGION, SPLIT_BUFFER) or overlaps(VAL_REGION, SPLIT_BUFFER):
        return None                     # buffer: discard to stop overlap leakage
    return "train"


def build_index(tokens: np.ndarray, phi: dict) -> dict:
    chrom_len = len(tokens)
    is_n = tokens == N_TOK
    n_cumsum = np.concatenate([[0], np.cumsum(is_n, dtype=np.int64)])

    usable = phi["usable"]
    bins_per_window = WINDOW // RES

    kept: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    dropped = {"buffer": 0, "n_frac": 0, "struct_frac": 0, "short": 0}

    def maybe_keep(split: str | None, start: int) -> None:
        if split is None:
            dropped["buffer"] += 1
            return
        n_frac = (n_cumsum[start + WINDOW] - n_cumsum[start]) / WINDOW
        if n_frac > MAX_N_FRAC:
            dropped["n_frac"] += 1
            return
        b0 = start // RES
        b1 = min(b0 + bins_per_window + 1, len(usable))
        if usable[b0:b1].mean() < MIN_STRUCT_FRAC:
            dropped["struct_frac"] += 1
            return
        kept[split].append(start)

    # train: full-chromosome scan at STRIDE_TRAIN (50% overlap).
    for start in range(0, chrom_len, STRIDE_TRAIN):
        end = start + WINDOW
        if end > chrom_len:
            dropped["short"] += 1
            continue
        split = assign_split(start, end)
        if split == "train":
            maybe_keep(split, start)

    # val/test: enumerated independently, anchored at each region boundary and
    # stepping by STRIDE_EVAL == WINDOW, so consecutive held-out windows are
    # non-overlapping and the region is covered without depending on the train
    # grid's alignment to the chromosome origin.
    for split_name, region in (("val", VAL_REGION), ("test", TEST_REGION)):
        for start in range(region[0], region[1], STRIDE_EVAL):
            end = start + WINDOW
            if end > chrom_len:
                dropped["short"] += 1
                continue
            split = assign_split(start, end)
            if split != split_name:
                continue                # window must sit entirely in its region
            maybe_keep(split, start)

    print(f"  window {WINDOW:,} bp, train stride {STRIDE_TRAIN:,}, "
          f"eval stride {STRIDE_EVAL:,}")
    for k, v in kept.items():
        span = len(v) * WINDOW / 1e6
        print(f"    {k:<6} {len(v):>6,} windows  (~{span:,.0f} Mb of sequence)")
    print(f"  dropped: {dropped}")
    return {k: np.array(v, dtype=np.int64) for k, v in kept.items()}, dropped


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("=== tokenise ===")
    tokens = tokenise()

    print("=== phi ===")
    phi = load_phi()

    print("=== window index ===")
    index, dropped = build_index(tokens, phi)

    print("=== interpolation check ===")
    # a window inside the validated NOTCH1 region -- structure must vary within it
    probe_start = 136_450_000
    pos = np.arange(probe_start, probe_start + WINDOW)
    pi, pv = interpolate_phi(phi["phi"], phi["usable"], pos)
    step_bins = pos // RES
    stepped = phi["phi"][np.clip(step_bins, 0, phi["phi"].shape[0] - 1)]
    print(f"  probe window at {CHROM}:{probe_start:,}  valid {100*pv.mean():.1f}%")
    print(f"  distinct values per feature within one window:")
    for i, nm in enumerate(phi["names"][:3]):
        print(f"    {nm:<20} stepped {len(np.unique(stepped[:, i])):>6,}   "
              f"interpolated {len(np.unique(pi[:, i])):>8,}")
    print("  (stepped = one value per 5 kb bin; interpolated varies per token,")
    print("   which is the F4 mitigation architecture_spec.md 4.1.4 requires)")

    tok_path = PROCESSED / f"tokens_{CHROM}.npy"
    np.save(tok_path, tokens)

    idx_path = PROCESSED / "dataset_index.npz"
    np.savez_compressed(
        idx_path,
        train=index["train"], val=index["val"], test=index["test"],
        phi=phi["phi"], usable=phi["usable"],
        feature_names=np.array(phi["names"]),
        feature_symmetry=phi["symmetry"],
        window=np.int64(WINDOW), resolution=np.int64(RES),
        stride_train=np.int64(STRIDE_TRAIN), stride_eval=np.int64(STRIDE_EVAL),
        vocab_size=np.int64(16), pad=np.int64(PAD), mask=np.int64(MASK),
        test_region=np.array(TEST_REGION), val_region=np.array(VAL_REGION),
        split_buffer=np.int64(SPLIT_BUFFER),
    )

    meta = {
        "chrom": CHROM, "window": WINDOW,
        "stride_train": STRIDE_TRAIN, "stride_eval": STRIDE_EVAL,
        "max_n_frac": MAX_N_FRAC, "min_struct_frac": MIN_STRUCT_FRAC,
        "test_region": list(TEST_REGION), "val_region": list(VAL_REGION),
        "split_buffer": SPLIT_BUFFER,
        "n_windows": {k: int(len(v)) for k, v in index.items()},
        "dropped": dropped,
        "feature_names": phi["names"],
        "feature_symmetry": [int(x) for x in phi["symmetry"]],
        "vocab": {"PAD": PAD, "MASK": MASK, "A": A, "C": C, "G": G, "T": T, "N": N_TOK},
        "phi_delivery": "interpolated to token resolution at load time, not stepped",
    }
    (PROCESSED / "dataset_meta.json").write_text(json.dumps(meta, indent=2),
                                                 encoding="utf-8")

    print()
    print(f"  {tok_path.relative_to(REPO)}  {tok_path.stat().st_size/1e6:.1f} MB")
    print(f"  {idx_path.relative_to(REPO)}  {idx_path.stat().st_size/1e6:.1f} MB")
    print(f"  {(PROCESSED / 'dataset_meta.json').relative_to(REPO)}")
    print("\nready for Phase 3. Nothing has been trained.")


# Shuffled-structure controls, architecture_spec.md 4.1.3. All operate on phi
# BEFORE the structural encoder and leave the sequence completely untouched, so
# any difference they produce is attributable to structure and nothing else.
PHI_CONTROLS = ("none", "S1", "S2", "S3", "S4")

# S2 shifts by at least this many bases. The spec asks for "much greater than
# the max memory horizon"; the largest trained tau measured in Phase 3 v2 is
# ~6e7 tokens, but tau above the 32,768-token window is not behaviourally
# distinguishable (architecture_spec.md 4.1.4 scope limit), so the window is the
# quantity that matters and 10 Mb is ~300x it.
S2_SHIFT_BP = 10_000_000

# Controls are drawn from a FIXED seed, deliberately not the training seed. Each
# training seed must see the SAME shuffled structure, otherwise seed variance and
# shuffle variance are confounded and the 2*sigma_real gate cannot be read.
PHI_CONTROL_SEED = 20260815


def apply_phi_control(phi: np.ndarray, usable: np.ndarray, control: str
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (phi, usable) transformed by one of the S-controls.

    S1 GLOBAL-PERM      permute phi rows uniformly across usable bins. Destroys
                        sequence<->structure correspondence AND local
                        autocorrelation; preserves the marginal distribution
                        exactly. The primary reliance probe.
    S2 CIRCULAR-SHIFT   roll phi by 10 Mb. Destroys alignment only; preserves
                        marginal and local autocorrelation. Controls for "the
                        model just likes any smooth auxiliary channel".
    S3 DISTANCE-MATCHED phi recomputed from contacts resampled under the
                        empirical P(s). CANNOT be produced by permuting phi --
                        it requires the contact matrix -- so it is read from a
                        file the Phase 1 feature pipeline must write.
    S4 SEQUENCE-MATCHED phi replaced by eight ALIGNED covariates computed from
                        sequence and GENCODE alone, no Hi-C. Also read from a
                        file (scripts/phase4_build_s4.py). Rules out "structure
                        is just GC content and gene density"; S2 cannot, because
                        that objection needs alignment PRESERVED.

    `usable` is carried through the same transform as `phi`: a permuted or
    shifted feature vector that kept the original validity mask would let
    invented structure land on bins that were masked out for good reason.
    """
    if control not in PHI_CONTROLS:
        raise ValueError(f"unknown phi control {control!r}; expected {PHI_CONTROLS}")
    if control == "none":
        return phi, usable
    if control == "S1":
        rng = np.random.default_rng(PHI_CONTROL_SEED)
        perm = rng.permutation(phi.shape[0])
        return phi[perm], usable[perm]
    if control == "S2":
        shift = S2_SHIFT_BP // RES
        if shift >= phi.shape[0]:
            raise ValueError(
                f"S2 shift of {S2_SHIFT_BP:,} bp is {shift} bins, but phi has "
                f"only {phi.shape[0]}; a shift >= the chromosome wraps to a "
                f"near-identity and is not a control")
        return np.roll(phi, shift, axis=0), np.roll(usable, shift, axis=0)
    # S3 and S4 are both precomputed feature files rather than transforms of
    # phi, for the same underlying reason: neither can be produced by moving the
    # existing numbers around. S3 needs the contact matrix; S4 needs the
    # sequence and the annotation.
    builder = {
        "S3": ("scripts/phase1_features.py --rewire-distance-matched",
               "the distance-matched rewire control: it resamples contacts "
               "under the empirical P(s) and RECOMPUTES phi"),
        "S4": ("scripts/phase4_build_s4.py",
               "the sequence-matched control: eight aligned covariates from "
               "sequence and GENCODE, with no Hi-C input at all"),
    }[control]
    path = PROCESSED / f"phi_{CHROM}_{RES}bp_{control}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. {control} is {builder[1]}, so it cannot be made "
            f"by permuting the phi array. Build it with: {builder[0]}")
    z = np.load(path, allow_pickle=True)
    ctl_phi, ctl_usable = z["phi"], z["usable"].astype(bool)
    if ctl_phi.shape != phi.shape:
        raise ValueError(
            f"{control} phi has shape {ctl_phi.shape}, real phi has "
            f"{phi.shape}; a control of a different shape is not a control")
    # The control must be defined where the real features are, or the two runs
    # see different positions and the comparison is confounded by coverage.
    if int((ctl_usable & usable).sum()) < int(usable.sum()):
        lost = int(usable.sum()) - int((ctl_usable & usable).sum())
        print(f"[phi_control] WARNING: {control} is undefined at {lost:,} bins "
              f"where real phi is defined; restricting to the intersection")
    return ctl_phi, (ctl_usable & usable)


class WindowDataset:
    """Windows of tokenised sequence with structural features attached.

    phi is interpolated per position here rather than precomputed, which keeps
    the on-disk footprint at 1.3 MB instead of 4.4 GB.

    Returns dicts of numpy arrays; wrap in torch.from_numpy at the collate step
    so this module stays importable without torch.
    """

    def __init__(self, split: str, structural: bool = True,
                 rc_augment: bool = False, seed: int = 0,
                 phi_control: str = "none"):
        z = np.load(PROCESSED / "dataset_index.npz", allow_pickle=True)
        self.starts = z[split]
        self.window = int(z["window"])
        self.phi = z["phi"]
        self.usable = z["usable"].astype(bool)
        self.symmetry = z["feature_symmetry"].astype(np.int8)
        self.tokens = np.load(PROCESSED / f"tokens_{CHROM}.npy", mmap_mode="r")
        self.structural = structural
        self.phi_control = phi_control
        if structural:
            self.phi, self.usable = apply_phi_control(
                self.phi, self.usable, phi_control)
        elif phi_control != "none":
            raise ValueError(
                f"phi_control={phi_control!r} with structural=False: the "
                f"baseline never reads phi, so this would silently be a no-op "
                f"and produce a run that looks like a control but is not one")
        self.rc_augment = rc_augment
        self.rng = np.random.default_rng(seed)
        self._comp = np.arange(16, dtype=np.int8)
        for a, b in COMPLEMENT.items():
            self._comp[a] = b

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict:
        start = int(self.starts[i])
        tok = np.asarray(self.tokens[start:start + self.window]).copy()
        out = {"tokens": tok, "start": start}

        if self.structural:
            pos = np.arange(start, start + self.window)
            s, valid = interpolate_phi(self.phi, self.usable, pos)
            out["phi"] = s
            out["phi_valid"] = valid

        if self.rc_augment and self.rng.random() < 0.5:
            out["tokens"] = self._comp[tok[::-1]]
            if self.structural:
                # reversing the window flips the antisymmetric coordinates
                out["phi"] = out["phi"][::-1] * self.symmetry[None, :]
                out["phi_valid"] = out["phi_valid"][::-1]
            out["rc"] = True
        else:
            out["rc"] = False
        return out


if __name__ == "__main__":
    main()
