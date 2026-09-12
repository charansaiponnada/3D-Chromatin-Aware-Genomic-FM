"""Phase 5 CLI --- evaluate a trained run, or collect every run into one table.

    python scripts/evaluate.py --run full_seed0
    python scripts/evaluate.py --run full_seed0 --held-out-cell-line K562
    python scripts/evaluate.py --collect

The headline setting is `hic_free`: every arm is evaluated on sequence alone,
so the only thing separating them is what shaped their weights during
pretraining. Numbers measured `with_hic` are recorded too, but they are an
upper bound with privileged input and must never be compared against a
sequence-only baseline --- a reviewer will catch that in one sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import load_config, results_dir                # noqa: E402
from chromgraph.evaluate import collect_comparison, evaluate_run      # noqa: E402
from chromgraph.train import pick_device                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="directory name under results/")
    ap.add_argument("--collect", action="store_true",
                    help="gather every metrics.json into final_comparison.json")
    ap.add_argument("--held-out-cell-line", default=None,
                    help="evaluate on a cell line the model never trained on")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    if args.collect:
        summary = collect_comparison()
        if not summary:
            print("No metrics.json found under results/. Nothing to collect.")
            print("Every table in the decks and on the website stays ?? until "
                  "this file exists -- which is the intended behaviour.")
            return 0
        print(json.dumps(summary, indent=2))
        print(f"\nwrote {results_dir() / 'final_comparison.json'}")
        return 0

    if not args.run:
        ap.error("pass --run <name> or --collect")

    run_dir = results_dir() / args.run
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        print(f"no run_config.yaml in {run_dir}; was this run started by "
              f"scripts/pretrain.py?", file=sys.stderr)
        return 1

    # Evaluate with the run's OWN config, not the current base.yaml. A result
    # has to be read against the settings that produced it.
    cfg = load_config(cfg_path)
    device = pick_device(args.device)

    cell_lines = ([args.held_out_cell_line] if args.held_out_cell_line
                  else cfg.data.cell_lines)
    splits = {"val": cfg.data.chroms_val, "test": cfg.data.chroms_test}

    print(f"evaluating {args.run}  arm={cfg.train.arm}  cell_lines={cell_lines}")
    report = evaluate_run(run_dir, cfg, device, splits, cell_lines,
                          max_batches=args.max_batches)

    for key, metrics in report["settings"].items():
        overall = metrics.get("overall", {})
        long = metrics.get("long", {})
        print(f"\n  {key}   ({metrics.get('n_edges', 0)} held-out edges)")
        print(f"    overall     r={overall.get('pearson')}  auprc={overall.get('auprc')}")
        print(f"    long-range  r={long.get('pearson')}  auprc={long.get('auprc')}  "
              f"n={long.get('n')}")

    print(f"\nwrote {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
