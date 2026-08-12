#!/usr/bin/env python3
"""Print one line per completed eval across all Phase 3 baseline runs.

Reads results/baselines/baseline_seed*/metrics.json, which train.py writes
atomically (.tmp then os.replace) at every eval. This is the reliable progress
source: train.py's console output is piped through grep inside the supervisor,
and grep block-buffers when its stdout is a file, so the console log can lag
tens of lines behind. metrics.json never lags.

Records are append-only, so a watcher can print the tail beyond what it has
already seen.
"""

import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    for run_dir in sorted(glob.glob(os.path.join(REPO, "results/baselines/baseline_seed*"))):
        path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(path):
            continue
        try:
            records = json.loads(open(path, encoding="utf-8").read())
        except (json.JSONDecodeError, OSError):
            continue  # mid-replace; the next poll will see it
        name = os.path.basename(run_dir)
        for rec in records:
            if "val_bits_per_nucleotide" not in rec:
                continue
            tau = (rec.get("tau_empirical") or {}).get("summary") or {}
            bits = rec["val_bits_per_nucleotide"]
            acc = rec.get("val_accuracy", float("nan"))
            med = tau.get("tau_median", float("nan"))
            mx = tau.get("tau_max", float("nan"))
            ge100k = tau.get("frac_ge_100k", float("nan"))
            print(f"{name} step {rec['step']:>5}  val {bits:.4f} bits  acc {acc:.4f}  "
                  f"tau_med {med:.1f}  tau_max {mx:.1f}  frac>=100k {ge100k:.6f}")


if __name__ == "__main__":
    main()
