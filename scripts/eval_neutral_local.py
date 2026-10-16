"""Evaluation-only check: a truly sequence-only Hi-C-free pass.

In the hic_free setting the distal edges are removed, but the local edges
(+/- local_radius) still carry the MEASURED strength from the Hi-C map
(chromgraph/graph.py:_local_edges), which feeds EdgeBias. This script re-runs
the head evaluation and the frozen probe with every local strength replaced by
the neutral 0.5 that _local_edges already uses for missing pixels, so the
model sees DNA only. No weights, losses or configs change.

    python scripts/eval_neutral_local.py --run full_seed1 [--run ...]

Writes results/neutral_local/<run>/{metrics,probe_metrics}.json and the
prediction files; the original results/<run>/ is never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph import model as model_mod                     # noqa: E402
from chromgraph.config import load_config, results_dir        # noqa: E402
from chromgraph.evaluate import evaluate_run                  # noqa: E402
from chromgraph.probe import probe_run                        # noqa: E402
from chromgraph.train import pick_device                      # noqa: E402

NEUTRAL = 0.5
_forward = model_mod.ChromGraphFM.forward


def _neutral_forward(self, batch, *args, use_structure: bool = True, **kwargs):
    if not use_structure:
        batch = dict(batch)
        batch["edge_strength"] = torch.where(
            batch["edge_is_local"], torch.full_like(batch["edge_strength"], NEUTRAL),
            batch["edge_strength"])
    return _forward(self, batch, *args, use_structure=use_structure, **kwargs)


model_mod.ChromGraphFM.forward = _neutral_forward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    for run in args.run:
        src = results_dir() / run
        dst = results_dir() / "neutral_local" / run
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / "checkpoint.pt").exists():
            (dst / "checkpoint.pt").symlink_to(src / "checkpoint.pt")
        cfg = load_config(src / "run_config.yaml")
        splits = {"val": cfg.data.chroms_val}
        rep = evaluate_run(dst, cfg, device, splits, cfg.data.cell_lines)
        prb = probe_run(dst, cfg, device, splits, cfg.data.cell_lines)
        h = rep["settings"]["val/hic_free"].get("long", {}).get("pearson")
        p = prb["settings"]["val/probe_hic_free"].get("long", {}).get("pearson")
        print(f"DONE {run} head_long={h} probe_long={p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
