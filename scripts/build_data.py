"""Phase 1 CLI --- build the dataset from data/manifest.json.

    python scripts/build_data.py --pilot                 # chr20, chr21, GM12878
    python scripts/build_data.py --full                  # every split chromosome
    python scripts/build_data.py --cell-lines GM12878 K562 --chroms chr20

No GPU. Run the pilot on a laptop to prove the pipeline, then --full on the
machine that will train. Nothing built here travels over git: the cluster
rebuilds from the manifest, which is the only data artefact git carries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import data_dir, load_config          # noqa: E402
from chromgraph.data import build_chromosome, load_manifest   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="two small chromosomes, one cell line -- proves the pipeline")
    ap.add_argument("--full", action="store_true",
                    help="every train/val/test chromosome for every configured cell line")
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--cell-lines", nargs="*", default=None)
    ap.add_argument("--band-bp", type=int, default=2_000_000,
                    help="how far from the diagonal to read; beyond this the "
                         "signal is compartment-scale and the pixel count explodes")
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    manifest = load_manifest()
    root = data_dir()

    if args.pilot:
        chroms = cfg.data.pilot_chroms
        cell_lines = cfg.data.cell_lines[:1]
    elif args.full:
        chroms = cfg.data.chroms_train + cfg.data.chroms_val + cfg.data.chroms_test
        cell_lines = cfg.data.cell_lines
    else:
        chroms = args.chroms or cfg.data.pilot_chroms
        cell_lines = args.cell_lines or cfg.data.cell_lines[:1]

    unknown = set(cell_lines) - set(manifest["cell_lines"])
    if unknown:
        print(f"unknown cell line(s): {sorted(unknown)}. "
              f"Manifest has {sorted(manifest['cell_lines'])}.", file=sys.stderr)
        return 1

    print(f"building {len(cell_lines)} cell line(s) x {len(chroms)} chromosome(s) "
          f"at {cfg.data.bin_size} bp into {root}")

    reports = []
    for cell_line in cell_lines:
        for chrom in chroms:
            reports.append(build_chromosome(cfg, manifest, cell_line, chrom, root,
                                            args.band_bp, force=args.force))

    summary = root / "build_summary.json"
    summary.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")
    print("\nThe checksums in each qc.json are the reproducibility gate: build the")
    print("same chromosome on another machine and they must match exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
