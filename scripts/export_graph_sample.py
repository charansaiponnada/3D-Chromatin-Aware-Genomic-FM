"""Export one REAL training sample's graph, per arm, for the website.

    python scripts/export_graph_sample.py
    python scripts/export_graph_sample.py --chrom chr20 --start 1200

Writes website/src/content/graph-sample.json. The website's other graph
picture is generated from the config; this one comes from the built GM12878
data through the same `build_sample` the training loop calls, so what it shows
is exactly what each arm's encoder is handed.

Only a TRAINING chromosome is exported. Val and test windows stay off the site.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from chromgraph.config import data_dir, load_config        # noqa: E402
from chromgraph.data import load_chromosome                 # noqa: E402
from chromgraph.graph import build_sample, sample_starts    # noqa: E402

OUT = REPO / "website" / "src" / "content" / "graph-sample.json"
SHOWN_ARMS = ("full", "b3_shuffled_hic", "b2_distance_only", "b1_random_graph", "b0_dna_only")


def pick_start(cfg, chrom_data) -> int:
    """The fully usable window-run with the most distal edges -- a representative, busy sample."""
    n = cfg.data.nodes_per_sample
    usable = chrom_data["usable"]
    best, best_edges = None, -1
    for start in sample_starts(len(usable), n, usable):
        if usable[start:start + n].mean() < 1.0:
            continue
        s = build_sample(cfg, chrom_data, "", "", start, arm="full")
        e = int((~s.edge_is_local).sum()) + s.tgt_index.shape[1]
        if e > best_edges:
            best, best_edges = start, e
    if best is None:
        raise SystemExit("no fully usable window-run on this chromosome")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-line", default="GM12878")
    ap.add_argument("--chrom", default="chr20")
    ap.add_argument("--start", type=int, default=None, help="start bin; default picks one")
    args = ap.parse_args()

    cfg = load_config()
    if args.chrom not in cfg.data.chroms_train:
        raise SystemExit(f"{args.chrom} is not a training chromosome; val/test stay off the site")

    chrom_data = load_chromosome(data_dir(), args.cell_line, args.chrom, cfg.data.bin_size)
    n = cfg.data.nodes_per_sample
    start = args.start if args.start is not None else pick_start(cfg, chrom_data)

    # The measured, detrended strength for every pixel inside the sample.
    row, col = chrom_data["row"], chrom_data["col"]
    inside = (row >= start) & (col < start + n)
    li = (row[inside] - start).astype(int)
    lj = (col[inside] - start).astype(int)
    ls = chrom_data["strength"][inside]
    matrix = [[int(i), int(j), round(float(s), 2)] for i, j, s in zip(li, lj, ls) if i < j]

    arms = {}
    targets = None
    for arm in SHOWN_ARMS:
        s = build_sample(cfg, chrom_data, args.cell_line, args.chrom, start, arm=arm)
        distal = ~s.edge_is_local
        arms[arm] = [[int(i), int(j), round(float(v), 2)] for i, j, v in
                     zip(s.edge_index[0, distal], s.edge_index[1, distal],
                         s.edge_strength[distal])]
        t = [[int(i), int(j), round(float(v), 2)] for i, j, v in
             zip(s.tgt_index[0], s.tgt_index[1], s.tgt_strength)]
        if targets is None:
            targets = t
        elif t != targets:
            raise SystemExit(f"{arm} holds out different targets from full -- arms would not be paired")

    out = {
        "source": {
            "cell_line": args.cell_line,
            "chrom": args.chrom,
            "start_bin": int(start),
            "start_bp": int(start * cfg.data.bin_size),
            "bin_size": cfg.data.bin_size,
            "control_seed": cfg.control_seed,
        },
        "n": n,
        "local_radius": cfg.data.local_radius,
        "matrix": matrix,
        "targets": targets,
        "arms": arms,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{args.cell_line} {args.chrom} bins {start}..{start + n - 1}: "
          f"{len(matrix)} pixels, {len(targets)} targets, "
          + ", ".join(f"{a} {len(e)}" for a, e in arms.items()))
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
