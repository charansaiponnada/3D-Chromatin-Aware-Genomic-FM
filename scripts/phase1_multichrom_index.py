"""Phase 1 (multi-chromosome) -- build dataset_index_multichrom.npz.

docs/RESEARCH_PLAN_2026-08-26.md Phase B1: the one genuinely new artefact the
multi-chromosome build needs. CLAUDE.md known weakness 5: splits inside chr9
are within one autocorrelated chromosome, so held-out phi leaks -- and it
biases TOWARD the structural arm. This index instead splits BY CHROMOSOME.

Chromosome roles, decided 2026-08-26 for this build (see the module-level
CHROM_ROLES below for the reasoning on the specific choice of chromosomes):
  test  : chr9   -- unchanged from the single-chromosome pipeline's convention
                    (phase1_acquire.py's CHROM default comment): CHROME holds
                    chr9 out as test, and matching that keeps the model
                    comparable to it. In THIS build chr9 is held out in its
                    ENTIRETY -- there is no more within-chr9 train/val carve
                    -out, because the split unit is now the whole chromosome.
  val   : two chromosomes
  train : the rest

Each train chromosome is scanned at STRIDE_TRAIN (50% overlap, matching the
single-chromosome convention); val and test chromosomes are scanned at
STRIDE_EVAL == WINDOW (non-overlapping), also matching. The N-fraction and
structural-coverage filters (MAX_N_FRAC, MIN_STRUCT_FRAC) are the same
constants phase1_dataset.py uses, imported rather than restated, so a window
that would be kept or dropped in the single-chromosome pipeline is kept or
dropped identically here.

REQUIRES, per chromosome in CHROM_ROLES: `tokens_{chrom}.npy` (or the raw
`data/interim/{chrom}.fa` to build it) and `phi_{chrom}_5000bp.npz`
(phase1_acquire.py --chrom {chrom} then phase1_features.py --chrom {chrom}
must already have been run).

phi standardisation is PER-CHROMOSOME (decided in phase1_features.py's module
docstring, 2026-08-26) -- each chromosome's phi is z-scored against its own
usable bins only. keep(phi) and any phi-derived number computed on the
combined index is therefore relative to each window's own chromosome's
variance, not one shared absolute scale; this must travel with any number
computed from this index.

Run:  python scripts/phase1_multichrom_index.py
      python scripts/phase1_multichrom_index.py --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"
sys.path.insert(0, str(REPO / "scripts"))

import phase1_dataset as PD   # noqa: E402  -- WINDOW, RES, MAX_N_FRAC, etc.

# Chromosome roles for this build. Picked to land near "~1 Gb total, 6-8
# chromosomes" (docs/NEXT_SESSION.md §4 / RESEARCH_PLAN_2026-08-26.md B1):
# chr10-chr15 are, like chr9, mid-sized autosomes (each 100-135 Mb, GRCh38),
# avoiding the very large (chr1-chr8, would blow the ~1 Gb target on their
# own) and the small acrocentric ones (chr13-15, chr21-22 carry large
# repetitive short arms that complicate compartment calling; chr13/14/15 are
# included here anyway as val/train because they are still net-informative
# and the phi validity mask already drops centromeric/acrocentric bins, as it
# does for chr9's own centromere).
CHROM_ROLES = {
    "chr9": "test",
    "chr10": "train", "chr11": "train", "chr12": "train", "chr13": "train",
    "chr14": "val", "chr15": "val",
}

OUT = PROCESSED / "dataset_index_multichrom.npz"
META = PROCESSED / "dataset_meta_multichrom.json"


def per_chromosome_windows(chrom: str, role: str) -> tuple[np.ndarray, dict]:
    """(starts, dropped-counts) for one chromosome, scanned whole -- no
    within-chromosome held-out region, because the split unit here is the
    chromosome itself."""
    PD.CHROM = chrom
    tokens = PD.tokenise()
    phi_d = PD.load_phi()
    chrom_len = len(tokens)
    is_n = tokens == PD.N_TOK
    n_cumsum = np.concatenate([[0], np.cumsum(is_n, dtype=np.int64)])
    usable = phi_d["usable"]
    bins_per_window = PD.WINDOW // PD.RES

    stride = PD.STRIDE_TRAIN if role == "train" else PD.STRIDE_EVAL
    kept: list[int] = []
    dropped = {"n_frac": 0, "struct_frac": 0, "short": 0}
    for start in range(0, chrom_len, stride):
        end = start + PD.WINDOW
        if end > chrom_len:
            dropped["short"] += 1
            continue
        n_frac = (n_cumsum[start + PD.WINDOW] - n_cumsum[start]) / PD.WINDOW
        if n_frac > PD.MAX_N_FRAC:
            dropped["n_frac"] += 1
            continue
        b0 = start // PD.RES
        b1 = min(b0 + bins_per_window + 1, len(usable))
        if usable[b0:b1].mean() < PD.MIN_STRUCT_FRAC:
            dropped["struct_frac"] += 1
            continue
        kept.append(start)

    tok_path = PROCESSED / f"tokens_{chrom}.npy"
    if not tok_path.exists():
        np.save(tok_path, tokens)
        print(f"    wrote {tok_path.relative_to(REPO)}")

    return np.array(kept, dtype=np.int64), dropped, phi_d, tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="verify every chromosome's inputs exist, build "
                         "nothing")
    args = ap.parse_args()

    print("=" * 74)
    print("multi-chromosome dataset index -- roles:")
    for c, r in CHROM_ROLES.items():
        print(f"    {c:<6} {r}")
    print("=" * 74)

    missing = []
    for chrom in CHROM_ROLES:
        fa = INTERIM / f"{chrom}.fa"
        tok = PROCESSED / f"tokens_{chrom}.npy"
        phi = PROCESSED / f"phi_{chrom}_{PD.RES}bp.npz"
        have_seq = fa.exists() or tok.exists()
        have_phi = phi.exists()
        status = "OK" if (have_seq and have_phi) else "MISSING"
        print(f"  {chrom:<6} seq:{'y' if have_seq else 'n'} "
              f"phi:{'y' if have_phi else 'n'}   {status}")
        if not (have_seq and have_phi):
            missing.append(chrom)

    if missing:
        print(f"\nMISSING inputs for: {missing}")
        print("Run, per chromosome:")
        for c in missing:
            print(f"  ./3d-gen/bin/python -u scripts/phase1_acquire.py --chrom {c}")
            print(f"  ./3d-gen/bin/python -u scripts/phase1_features.py --chrom {c}")
        if args.check_only or True:
            print("\nStopping -- this run built nothing. Not all chromosomes "
                  "in CHROM_ROLES have inputs on disk yet.")
        return 1

    if args.check_only:
        print("\nall chromosome inputs present.")
        return 0

    t0 = time.time()
    names_names, names_sym = None, None
    per_chrom_phi, per_chrom_usable = {}, {}
    starts_by_split: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    chromid_by_split: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    chrom_list = list(CHROM_ROLES.keys())
    dropped_all = {}

    for cid, (chrom, role) in enumerate(CHROM_ROLES.items()):
        print(f"\n=== {chrom} ({role}) ===")
        starts, dropped, phi_d, tokens = per_chromosome_windows(chrom, role)
        span_mb = len(starts) * PD.WINDOW / 1e6
        print(f"  {len(starts):,} windows (~{span_mb:,.0f} Mb), dropped {dropped}")
        starts_by_split[role].append(starts)
        chromid_by_split[role].append(np.full(len(starts), cid, dtype=np.int64))
        dropped_all[chrom] = dropped

        per_chrom_phi[f"phi_{chrom}"] = phi_d["phi"]
        per_chrom_usable[f"usable_{chrom}"] = phi_d["usable"]
        if names_names is None:
            names_names = phi_d["names"]
            names_sym = phi_d["symmetry"]
        else:
            assert phi_d["names"] == names_names, (
                f"{chrom} feature_names differ from {chrom_list[0]}")

    save_kwargs = {}
    for split in ("train", "val", "test"):
        save_kwargs[f"{split}_starts"] = (np.concatenate(starts_by_split[split])
                                          if starts_by_split[split] else np.array([], dtype=np.int64))
        save_kwargs[f"{split}_chrom_id"] = (np.concatenate(chromid_by_split[split])
                                            if chromid_by_split[split] else np.array([], dtype=np.int64))
    save_kwargs.update(per_chrom_phi)
    save_kwargs.update(per_chrom_usable)

    np.savez_compressed(
        OUT,
        chrom_names=np.array(chrom_list),
        chrom_roles=np.array([CHROM_ROLES[c] for c in chrom_list]),
        feature_names=np.array(names_names),
        feature_symmetry=names_sym,
        window=np.int64(PD.WINDOW), resolution=np.int64(PD.RES),
        stride_train=np.int64(PD.STRIDE_TRAIN), stride_eval=np.int64(PD.STRIDE_EVAL),
        vocab_size=np.int64(16), pad=np.int64(PD.PAD), mask=np.int64(PD.MASK),
        phi_standardisation=np.array("per-chromosome"),
        **save_kwargs,
    )

    meta = {
        "chrom_roles": CHROM_ROLES,
        "window": PD.WINDOW, "resolution": PD.RES,
        "stride_train": PD.STRIDE_TRAIN, "stride_eval": PD.STRIDE_EVAL,
        "max_n_frac": PD.MAX_N_FRAC, "min_struct_frac": PD.MIN_STRUCT_FRAC,
        "phi_standardisation": "per-chromosome",
        "n_windows": {split: int(len(save_kwargs[f"{split}_starts"]))
                      for split in ("train", "val", "test")},
        "dropped": dropped_all,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("Split is BY CHROMOSOME, not by coordinate. chr9 is held out "
                 "in its ENTIRETY as test (no internal train/val carve-out, "
                 "unlike the single-chromosome dataset_index.npz). "
                 "tokens for each chromosome are in tokens_{chrom}.npy; this "
                 "index does not duplicate them, only phi/usable and the "
                 "(start, chrom_id) window lists."),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print("=" * 74)
    for split in ("train", "val", "test"):
        n = len(save_kwargs[f"{split}_starts"])
        chroms_in_split = [c for c, r in CHROM_ROLES.items() if r == split]
        span_mb = n * PD.WINDOW / 1e6
        print(f"  {split:<6} {n:>6,} windows  (~{span_mb:,.0f} Mb)  {chroms_in_split}")
    print(f"\nwrote {OUT.relative_to(REPO)} ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {META.relative_to(REPO)}")
    print(f"\ndone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
