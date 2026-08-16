#!/usr/bin/env python3
"""Build the S3 DISTANCE-MATCHED REWIRE control (architecture_spec.md 4.1.3).

The spec calls S3 "the control that matters most", and it is:

    Contact probability decays smoothly with genomic distance under the
    fractal-globule model. If a model does as well on distance-matched contacts
    as on real ones, the mechanism has learned a re-parameterised positional
    prior, not a structural one -- and the paper's central claim is false
    regardless of how good the headline numbers look.

WHAT "DISTANCE-MATCHED" MEANS HERE, CONCRETELY
----------------------------------------------
For every genomic separation d, the set of contact values observed at that
separation is kept exactly, and permuted across locus pairs. So:

  PRESERVED   P(s), exactly -- not approximately, not in expectation. The
              multiset of values on each diagonal is identical to the real one,
              so every distance-decay statistic is unchanged by construction.
  PRESERVED   the measurability pattern. NaN stays NaN, zero-coverage bins stay
              zero-coverage, so the control sees the same positions as the real
              run and coverage cannot confound the comparison.
  DESTROYED   locus specificity -- which particular pair of loci holds which
              contact value. That is precisely the 3D information the mechanism
              claims to use.

Permuting within a diagonal is the right operation rather than replacing values
with their diagonal mean: a mean-substituted control would also flatten the
variance of contacts, and would then be testing two things at once.

Both Hi-C inputs are already cached under data/interim/, so this runs offline.
phi is recomputed by calling phase1_features' OWN functions, so the only
difference between real phi and S3 phi is the contact matrix they were computed
from -- not a reimplementation that might drift.

    ./3d-gen/bin/python scripts/phase4_build_s3.py

Writes data/processed/phi_chr9_5000bp_S3.npz and a validation report.
CPU only, no network.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
PROCESSED = REPO / "data" / "processed"
INTERIM = REPO / "data" / "interim"

import phase1_features as pf                                    # noqa: E402
from phase1_dataset import PHI_CONTROL_SEED                     # noqa: E402

CHROM, RES = pf.CHROM, pf.RES


def rewire_band(band: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute contact values within each diagonal of the 5 kb band.

    band[i, d] is the contact between bin i and bin i+d, so column d IS the
    diagonal at separation d*RES. Permuting a column across rows is therefore
    exactly a distance-matched rewire, and NaNs are left in place so the
    measurability pattern survives untouched.
    """
    out = band.copy()
    moved = kept_nan = 0
    for d in range(band.shape[1]):
        col = band[:, d]
        ok = np.isfinite(col)
        k = int(ok.sum())
        kept_nan += int((~ok).sum())
        if k < 2:
            continue
        idx = np.flatnonzero(ok)
        out[idx, d] = col[rng.permutation(idx)]
        moved += k
    print(f"  band: permuted {moved:,} measurable cells across "
          f"{band.shape[1]} diagonals, {kept_nan:,} NaNs left in place")
    return out


def rewire_matrix(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same operation on the square 250 kb matrix used for compartments.

    Done on the upper triangle and mirrored, because a contact matrix is
    symmetric and permuting both triangles independently would produce an
    object that is not a contact matrix at all.
    """
    k = M.shape[0]
    out = np.full_like(M, np.nan)
    np.fill_diagonal(out, np.diag(M))
    moved = 0
    for d in range(1, k):
        i = np.arange(k - d)
        v = M[i, i + d]
        ok = np.isfinite(v)
        if int(ok.sum()) < 2:
            out[i, i + d] = v
            out[i + d, i] = v
            continue
        w = v.copy()
        sel = np.flatnonzero(ok)
        w[sel] = v[rng.permutation(sel)]
        moved += int(ok.sum())
        out[i, i + d] = w
        out[i + d, i] = w
    print(f"  coarse matrix: permuted {moved:,} cells across {k - 1} diagonals, "
          f"symmetry restored by mirroring")
    return out


def diagonal_means(band: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmean(band[:, d]) if np.isfinite(band[:, d]).any()
                         else np.nan for d in range(band.shape[1])])


def main() -> int:
    t0 = time.time()
    print("=== building S3 DISTANCE-MATCHED REWIRE control ===")
    print(f"seed {PHI_CONTROL_SEED} (fixed, not the training seed)\n")
    rng = np.random.default_rng(PHI_CONTROL_SEED)

    print("=== loading real band ===")
    B = pf.load_band()
    band, valid, n, w = B["band"], B["valid"], B["n"], B["w"]
    bin_start = B["bin_start"]

    print("\n=== rewiring ===")
    ps_before = diagonal_means(band)
    band_s3 = rewire_band(band, rng)
    ps_after = diagonal_means(band_s3)

    fin = np.isfinite(ps_before) & np.isfinite(ps_after)
    max_dev = float(np.max(np.abs(ps_before[fin] - ps_after[fin]))) if fin.any() else 0.0
    print(f"  P(s) check: max |mean(d) before - after| = {max_dev:.3e} "
          f"over {int(fin.sum())} diagonals")
    if max_dev > 1e-5:
        print("  ERROR: permuting within a diagonal must leave its mean exactly "
              "unchanged. It did not, so this is not a distance-matched control.")
        return 1

    # How much locus specificity actually went away? If this correlation were
    # high the control would be toothless, so it is measured rather than assumed.
    a, b = band[np.isfinite(band)], band_s3[np.isfinite(band_s3)]
    same = float(np.mean(a == b)) if a.size == b.size else float("nan")
    print(f"  cells whose value did not move: {same:.4f}")

    print("\n=== recomputing phi from the rewired band ===")
    print("(phase1_features' own functions -- the ONLY difference from real phi "
          "is the matrix)\n")
    cum, cnt = pf._rowcum(band_s3)
    raw: dict[str, np.ndarray] = {}
    for bp in pf.INSULATION_WINDOWS_BP:
        wb = bp // RES
        raw[f"insulation_{bp//1000}kb"] = pf.insulation(band_s3, n, wb, cum, cnt)
        v = raw[f"insulation_{bp//1000}kb"]
        print(f"  insulation_{bp//1000}kb finite {np.isfinite(v).mean():.4f}")

    wdi = pf.DI_WINDOW_BP // RES
    A, Bm = pf.upstream_downstream(band_s3, n, wdi)
    raw["directionality_2Mb"] = pf.directionality_index(A, Bm)
    tot = A + Bm
    raw["log_contact_density"] = np.where(tot > 0, np.log1p(tot), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["upstream_mass_frac"] = np.where(tot > 0, A / tot, np.nan)
    wsr = pf.SHORT_RANGE_BP // RES
    short = np.nansum(band_s3[:, 1:wsr + 1], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["short_long_ratio"] = np.where(Bm > 0, short / Bm, np.nan)
    for k in ("directionality_2Mb", "log_contact_density",
              "upstream_mass_frac", "short_long_ratio"):
        print(f"  {k:<22} finite {np.isfinite(raw[k]).mean():.4f}")

    print("\n=== compartments from the rewired coarse matrix ===")
    zc = np.load(INTERIM / f"hic_coarse_{CHROM}_250000bp.npz")
    M_s3 = rewire_matrix(zc["matrix"].astype(np.float64), rng)
    tmp = INTERIM / f"hic_coarse_{CHROM}_250000bp_S3.npz"
    np.savez_compressed(tmp, matrix=M_s3, resolution=zc["resolution"])
    real_interim = pf.INTERIM
    try:
        # point compartments() at the rewired matrix by swapping the filename it
        # resolves, without touching phase1_features itself
        class _Shim:
            def __truediv__(self, name):
                if name == f"hic_coarse_{CHROM}_250000bp.npz":
                    return tmp
                return real_interim / name
        pf.INTERIM = _Shim()
        raw["compartment_pc1"] = pf.compartments(bin_start, n)
    finally:
        pf.INTERIM = real_interim
    print(f"  compartment_pc1 finite "
          f"{np.isfinite(raw['compartment_pc1']).mean():.4f}")

    print("\n=== assembling S3 phi ===")
    phi = np.stack([raw[k] for k in pf.FEATURE_NAMES], axis=1).astype(np.float32)
    usable = valid & np.isfinite(phi).all(axis=1)

    zr = np.load(PROCESSED / f"phi_{CHROM}_{RES}bp.npz", allow_pickle=True)
    real_usable = zr["usable"].astype(bool)
    print(f"  usable under real phi: {int(real_usable.sum()):,}")
    print(f"  usable under S3 phi:   {int(usable.sum()):,}")
    print(f"  intersection:          {int((usable & real_usable).sum()):,}")

    mu = phi[usable].mean(axis=0)
    sd = phi[usable].std(axis=0)
    sd[sd == 0] = 1.0
    phi_z = np.full_like(phi, np.nan)
    phi_z[usable] = (phi[usable] - mu) / sd
    print("  standardised chromosome-wide, exactly as real phi is")

    print("\n--- did the rewire actually remove locus-specific structure? ---")
    print("    (correlation of each S3 feature with its real counterpart)")
    both = usable & real_usable
    corrs = {}
    for i, nm in enumerate(pf.FEATURE_NAMES):
        x, y = zr["phi"][both, i], phi_z[both, i]
        r = (float(np.corrcoef(x, y)[0, 1])
             if x.std() > 0 and y.std() > 0 else float("nan"))
        corrs[nm] = round(r, 4)
        flag = "  <-- still correlated" if abs(r) > 0.3 else ""
        print(f"  {nm:<22} r = {r:+.4f}{flag}")

    out = PROCESSED / f"phi_{CHROM}_{RES}bp_S3.npz"
    np.savez_compressed(
        out, phi=phi_z, phi_raw=phi, usable=usable, valid=valid,
        bin_start=bin_start, bin_end=B["bin_end"],
        feature_names=np.array(pf.FEATURE_NAMES),
        feature_symmetry=np.array(pf.FEATURE_SYMMETRY),
        standardisation_mean=mu, standardisation_std=sd,
        resolution=np.int64(RES),
    )
    print(f"\nwrote {out.relative_to(REPO)} ({out.stat().st_size/1e6:.1f} MB)")

    rep = {
        "control": "S3 DISTANCE-MATCHED REWIRE",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": PHI_CONTROL_SEED,
        "method": "permute contact values within each diagonal; NaN pattern "
                  "left in place; coarse matrix permuted on the upper triangle "
                  "and mirrored",
        "ps_max_abs_deviation": max_dev,
        "band_cells_unmoved_frac": same,
        "usable_bins_real": int(real_usable.sum()),
        "usable_bins_s3": int(usable.sum()),
        "s3_vs_real_feature_correlation": corrs,
        "elapsed_s": round(time.time() - t0, 1),
    }
    rp = PROCESSED / "s3_validation_report.json"
    rp.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {rp.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
