#!/usr/bin/env python
"""C2 -- is a cross-cell-line DIFFERENTIAL probe powered, and on which feature?

docs/RESEARCH_PLAN_2026-08-26.md C2 says: do NOT probe insulation alone, because
TAD boundaries are largely conserved across cell types (Dixon et al. 2012) and a
model could pass by having memorised GM12878. It directs the probe at (a) A/B
compartments, "substantially more cell-type-variable", and (b) the DIFFERENTIAL,
the regions where two lines actually disagree.

Both halves of that instruction are ASSUMPTIONS about the data. This script
measures them on phi we built ourselves, before any probe is designed against
them, so the design is fixed by numbers rather than by the plan's priors.

No model, no GPU. Reads the chr14 phi built for all three cell lines.

Run: ./3d-gen/bin/python -u scripts/c2_differential_power.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "c2_differential_power.json"
LINES = {"GM12878": "phi_chr14_5000bp.npz",
         "K562": "phi_chr14_5000bp_K562.npz",
         "IMR90": "phi_chr14_5000bp_IMR90.npz"}
RES = 5_000


def main() -> int:
    Z = {}
    for tag, f in LINES.items():
        p = PROC / f
        if not p.exists():
            print(f"missing {p} -- run phase1 for that cell line first")
            return 2
        Z[tag] = np.load(p, allow_pickle=True)

    names = [str(x) for x in Z["GM12878"]["feature_names"]]
    ok = np.ones(len(Z["GM12878"]["phi_raw"]), bool)
    for t in Z:
        ok &= Z[t]["usable"]

    print("=" * 74)
    print("C2 -- cross-cell-line differential power, chr14 at 5 kb")
    print("=" * 74)
    print(f"  jointly usable bins: {ok.sum():,} of {len(ok):,} ({ok.mean():.1%})")
    print()

    raw = lambda t, i: Z[t]["phi_raw"][ok, i]
    res = {"chrom": "chr14", "resolution": RES,
           "jointly_usable_bins": int(ok.sum()),
           "n_bins": int(len(ok)),
           "feature_names": names, "pairwise_r": {}, "differential": {}}

    print("  Pairwise Pearson on jointly usable bins, all 8 features:")
    print(f"  {'feature':22s} {'GM-K562':>9s} {'GM-IMR90':>9s} {'K562-IMR90':>11s}")
    for i, nm in enumerate(names):
        row = {}
        for a, b in (("GM12878", "K562"), ("GM12878", "IMR90"), ("K562", "IMR90")):
            row[f"{a}_vs_{b}"] = float(np.corrcoef(raw(a, i), raw(b, i))[0, 1])
        res["pairwise_r"][nm] = row
        print(f"  {nm:22s} {row['GM12878_vs_K562']:+9.4f} "
              f"{row['GM12878_vs_IMR90']:+9.4f} {row['K562_vs_IMR90']:+11.4f}")

    print()
    print("  DIFFERENTIAL, per pair, on compartment_pc1 and insulation_100kb:")
    for a, b in (("GM12878", "K562"), ("GM12878", "IMR90")):
        for nm in ("compartment_pc1", "insulation_100kb"):
            i = names.index(nm)
            x, y = raw(a, i), raw(b, i)
            d = x - y
            flip = float((np.sign(x) != np.sign(y)).mean()) if nm.startswith("comp") else None
            top = np.abs(d) > np.percentile(np.abs(d), 90)
            k = f"{a}_vs_{b}__{nm}"
            res["differential"][k] = {
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "sign_disagreement_frac": flip,
                "abs_diff_median": float(np.median(np.abs(d))),
                "abs_diff_p90": float(np.percentile(np.abs(d), 90)),
                "top_decile_bins": int(top.sum()),
                "top_decile_mb": float(top.sum() * RES / 1e6),
            }
            v = res["differential"][k]
            print(f"    {a}/{b:8s} {nm:17s} r {v['pearson_r']:+.4f}  "
                  f"sign-flip {('n/a' if flip is None else f'{flip:.1%}'):>6s}  "
                  f"top-decile {v['top_decile_bins']:,} bins "
                  f"({v['top_decile_mb']:.2f} Mb)")

    # The plan's premise, tested rather than assumed.
    print()
    print("  PREMISE CHECK -- 'compartments are more cell-type-variable than")
    print("  insulation'. True only where compartment r < insulation r.")
    for a, b in (("GM12878", "K562"), ("GM12878", "IMR90")):
        rc = res["pairwise_r"]["compartment_pc1"][f"{a}_vs_{b}"]
        ri = res["pairwise_r"]["insulation_100kb"][f"{a}_vs_{b}"]
        holds = rc < ri
        res["differential"][f"{a}_vs_{b}__premise_holds"] = bool(holds)
        print(f"    {a} vs {b:8s}: compartment {rc:+.4f} vs insulation {ri:+.4f}"
              f"  -> {'HOLDS' if holds else 'FAILS -- compartment is MORE conserved'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
