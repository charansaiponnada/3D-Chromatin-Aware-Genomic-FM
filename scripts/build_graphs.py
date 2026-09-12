"""Phase 2 CLI --- inspect and verify the sparse contact graphs.

    python scripts/build_graphs.py --chrom chr20 --arm full
    python scripts/build_graphs.py --chrom chr20 --all-arms

Graphs are built on the fly during training rather than cached: they are cheap
next to the encoder, and caching would multiply disk by the number of arms.

This script exists to VERIFY them --- that target edges stay disjoint from
conditioning edges, that each control changes what it claims to change and
nothing else, and that edge counts stay comparable across arms. A control that
also changed the edge count would confound "wrong structure" with "less
structure", and the comparison would mean nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import data_dir, load_config             # noqa: E402
from chromgraph.data import load_chromosome                     # noqa: E402
from chromgraph.graph import ARMS, build_sample, sample_starts   # noqa: E402


def describe(cfg, chrom_data, cell_line, chrom, start, arm) -> dict:
    s = build_sample(cfg, chrom_data, cell_line, chrom, start, arm=arm)
    s.assert_disjoint()
    distal = ~s.edge_is_local
    return {
        "arm": arm,
        "n_edges": int(s.edge_index.shape[1]),
        "n_local": int(s.edge_is_local.sum()),
        "n_distal": int(distal.sum()),
        "n_targets": int(s.tgt_index.shape[1]),
        "strength_mean": float(s.edge_strength[distal].mean()) if distal.any() else None,
        "sep_mean": float(s.edge_sep[distal].mean()) if distal.any() else None,
        "usable_nodes": int(s.node_ok.sum()),
    }


def mean_of(rows: list[dict], key: str):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def fmt(value, places: int = 3) -> str:
    return "--" if value is None else f"{value:.{places}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cell-line", default=None)
    ap.add_argument("--chrom", default=None)
    ap.add_argument("--arm", default="full", choices=ARMS)
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--n-windows", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    cell_line = args.cell_line or cfg.data.cell_lines[0]
    chrom = args.chrom or cfg.data.pilot_chroms[0]

    chrom_data = load_chromosome(data_dir(), cell_line, chrom, cfg.data.bin_size)
    starts = sample_starts(len(chrom_data["usable"]), cfg.data.nodes_per_sample,
                           chrom_data["usable"])
    if not starts:
        print(f"no usable windows in {cell_line}/{chrom}", file=sys.stderr)
        return 1
    starts = starts[: args.n_windows]
    arms = list(ARMS) if args.all_arms else [args.arm]

    print(f"{cell_line}/{chrom}: {len(starts)} windows inspected\n")
    print(f"  {'arm':<22} {'edges':>7} {'distal':>7} {'targets':>8} "
          f"{'strength':>9} {'sep':>7}")

    summary = []
    for arm in arms:
        rows = [describe(cfg, chrom_data, cell_line, chrom, s, arm) for s in starts]
        agg = {
            "arm": arm,
            "edges": mean_of(rows, "n_edges"),
            "distal": mean_of(rows, "n_distal"),
            "targets": mean_of(rows, "n_targets"),
            "strength_mean": mean_of(rows, "strength_mean"),
            "sep_mean": mean_of(rows, "sep_mean"),
        }
        summary.append(agg)
        print(f"  {arm:<22} {fmt(agg['edges'], 1):>7} {fmt(agg['distal'], 1):>7} "
              f"{fmt(agg['targets'], 1):>8} {fmt(agg['strength_mean'], 4):>9} "
              f"{fmt(agg['sep_mean'], 1):>7}")

    if args.all_arms:
        by_arm = {row["arm"]: row for row in summary}
        full, shuffled = by_arm["full"], by_arm["b3_shuffled_hic"]
        print("\n  Sanity checks")
        same_edges = abs(full["edges"] - shuffled["edges"]) < 1e-9
        same_mean = abs(full["strength_mean"] - shuffled["strength_mean"]) < 1e-6
        print(f"    b3 keeps the edge count:            {'PASS' if same_edges else 'FAIL'}")
        print(f"    b3 keeps the strength distribution: {'PASS' if same_mean else 'FAIL'}")
        print("    (b3 permutes which strength sits on which edge. Same marginals,")
        print("     destroyed pairing -- that is what makes it the decisive control.)")
        flat = by_arm["b2_distance_only"]["strength_mean"]
        print(f"    b2 flattens strength to a constant: "
              f"{'PASS' if flat is not None and abs(flat - 0.5) < 1e-6 else 'FAIL'}")

    out = data_dir() / f"graph_report_{cell_line}_{chrom}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
