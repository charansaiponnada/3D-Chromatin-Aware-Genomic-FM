"""Frozen-representation contact probe for one run (see chromgraph/probe.py).

    python scripts/probe.py --run b0_dna_only_seed0               # val only
    python scripts/probe.py --run full_seed0 --with-test          # final table only

Writes results/<run>/probe_metrics.json. Every arm gets the identical probe:
same head architecture, training pairs, epochs and seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import load_config, results_dir   # noqa: E402
from chromgraph.probe import probe_run                   # noqa: E402
from chromgraph.train import pick_device                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--with-test", action="store_true")
    ap.add_argument("--max-train-samples", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    run_dir = results_dir() / args.run
    cfg = load_config(run_dir / "run_config.yaml")
    splits = {"val": cfg.data.chroms_val}
    if args.with_test:
        splits["test"] = cfg.data.chroms_test

    report = probe_run(run_dir, cfg, pick_device(args.device), splits, cfg.data.cell_lines,
                       max_train_samples=args.max_train_samples, epochs=args.epochs)
    for key, m in report["settings"].items():
        print(f"{key}: long r={m.get('long', {}).get('pearson')}  "
              f"overall r={m.get('overall', {}).get('pearson')}  n={m.get('n_edges')}")
    print(f"wrote {run_dir / 'probe_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
