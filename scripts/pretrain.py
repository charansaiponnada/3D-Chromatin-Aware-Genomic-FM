"""Phases 3-4 CLI --- pretrain one arm, one seed.

    python scripts/pretrain.py --arm full --seed 0
    python scripts/pretrain.py --arm b3_shuffled_hic --seed 0
    python scripts/pretrain.py --arm full --seed 0 --steps 20 --smoke

Every arm goes through this same script. That is the point: matched parameters,
matched data, matched optimiser, matched budget, and only the structural signal
differs. Anything that changes between arms other than `--arm` invalidates the
comparison.

Runs resume automatically from results/<run>/checkpoint.pt, so an interrupted
spot instance or a culled session costs minutes, not the whole run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import load_config, results_dir   # noqa: E402
from chromgraph.graph import ARMS                        # noqa: E402
from chromgraph.train import pick_device, train          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--arm", default="full", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None,
                    help="override train.steps; use with --smoke for a quick check")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model and batch, for proving the loop runs at all")
    args = ap.parse_args()

    overrides = {"train.arm": args.arm, "seed": args.seed}
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.smoke:
        overrides.update({
            "train.batch_size": 1,
            "train.grad_accum_steps": 1,
            "train.log_every": 1,
            "train.ckpt_every": 10_000,      # do not litter checkpoints during a smoke run
            "data.nodes_per_sample": 16,
            "model.d_model": 32,
            "model.n_heads": 2,
            "model.d_ff": 64,
            "model.encoder_layers": 1,
            "model.block_layers": 1,
        })

    cfg = load_config(args.config, **overrides) if args.config else load_config(**overrides)
    device = pick_device(args.device)
    run_name = args.run_name or f"{args.arm}_seed{args.seed}"

    print(f"arm {cfg.train.arm}  seed {cfg.seed}  device {device}")
    print(f"context {cfg.data.nodes_per_sample} windows x {cfg.data.bin_size} bp "
          f"= {cfg.data.context_bp / 1000:.0f} kb")

    out = train(cfg, run_name, device, max_steps=args.steps)
    print(f"\nresults in {out}")
    print(f"next:  python scripts/evaluate.py --run {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
