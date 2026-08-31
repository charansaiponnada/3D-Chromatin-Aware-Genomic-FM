#!/usr/bin/env python
"""Does a 131,072 bp window fit, and what does a step actually cost?

The window scan (p5_window_scan.json) makes 131,072 bp the indicated target:
4.32x the within-window structural signal for 4x the step time. Before that run
is scheduled, two things have to be measured rather than assumed, because both
have already cost this project time when guessed:

  1. MEMORY. fp32 throughout, no autocast, 44.39 GiB per L40S. Activation
     memory grows linearly in window length, and the scan holds
     (batch, d_inner, L) tensors -- at L = 131,072, batch 2, d_inner 512 that
     is 537 MB per tensor before anything else. train.py already has per-layer
     gradient checkpointing (its CheckpointedLM), so the question is what the
     real peak is with it on, not what the arithmetic says.

  2. STEP TIME. The 4x figure in the window scan is an assumption that cost
     scales linearly in L. Mamba's scan is O(L) so it should, but the measured
     number is what the schedule should be built from. CLAUDE.md 9 already
     records that wall_clock_s understates interrupted runs; an estimated
     s/step would compound that.

Forward + backward on real shapes, both arms, at each width. No training, no
checkpoint, no dataset -- random tokens of the right shape are enough to
measure memory and time, and using them keeps this runnable without touching
the index.

nvidia-smi does not work on this box (CLAUDE.md 3), so memory is read through
torch.cuda.max_memory_allocated / mem_get_info.
"""

import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from chromfm.model import BiMambaLM, ModelConfig, use_scan   # noqa: E402

WIDTHS = [32_768, 65_536, 131_072]
BATCH = 2
VOCAB = 16
D_STRUCT_RAW = 8


def one(width, structural, device, use_ckpt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cfg = ModelConfig(structural=structural)
    model = BiMambaLM(cfg).to(device)
    if use_ckpt:
        try:
            from train import CheckpointedLM
            model = CheckpointedLM(model).to(device)
        except Exception as e:                     # noqa: BLE001
            print(f"      (no CheckpointedLM: {e})")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    tokens = torch.randint(2, 6, (BATCH, width), device=device)
    kw = {}
    if structural:
        kw = {"phi": torch.randn(BATCH, width, D_STRUCT_RAW, device=device),
              "phi_valid": torch.ones(BATCH, width, dtype=torch.bool,
                                      device=device),
              "symmetry": torch.ones(D_STRUCT_RAW, device=device)}
    target = torch.randint(2, 6, (BATCH, width), device=device)

    times = []
    ok, err = True, ""
    try:
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            logits = model(tokens, **kw)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), target.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i > 0:                              # discard warmup
                times.append(time.time() - t0)
    except torch.cuda.OutOfMemoryError as e:       # noqa: PERF203
        ok, err = False, "OOM"
    except Exception as e:                         # noqa: BLE001
        ok, err = False, type(e).__name__ + ": " + str(e)[:80]

    peak = torch.cuda.max_memory_allocated(device) / 2**30
    del model, opt, tokens, target
    torch.cuda.empty_cache()
    return ok, err, peak, (sum(times) / len(times) if times else float("nan"))


def main() -> int:
    if not torch.cuda.is_available():
        print("No CUDA device.")
        return 2
    device = torch.device("cuda", 0)
    use_scan("triton")
    free, total = torch.cuda.mem_get_info(device)
    print("=" * 74)
    print("MEMORY + STEP TIME at wider windows   (fp32, batch 2, one L40S)")
    print("=" * 74)
    print(f"  device total {total / 2**30:.2f} GiB, free now {free / 2**30:.2f} GiB")
    print()
    print(f"  {'window':>9s} {'arm':>11s} {'ckpt':>5s} {'peak GiB':>9s} "
          f"{'s/step':>8s} {'status':>8s}")
    print("  " + "-" * 60)

    base32 = {}
    rows = []
    for width in WIDTHS:
        for structural in (False, True):
            for use_ckpt in (True,):
                ok, err, peak, sec = one(width, structural, device, use_ckpt)
                arm = "structural" if structural else "baseline"
                status = "ok" if ok else err
                print(f"  {width:9,d} {arm:>11s} {str(use_ckpt):>5s} "
                      f"{peak:9.2f} {sec:8.2f} {status:>8s}")
                rows.append({"window": width, "arm": arm, "grad_checkpoint": use_ckpt,
                             "peak_gib": round(peak, 4), "s_per_step": sec,
                             "ok": ok, "error": err})
                if width == 32_768 and ok:
                    base32[arm] = sec
        print()

    print("=" * 74)
    print("READ")
    print("=" * 74)
    print("  Peak is max_memory_allocated on ONE device; DDP over 2 GPUs runs")
    print("  one replica each, so this is the per-GPU figure that matters.")
    print("  s/step here is a single process with no gradient sync -- the")
    print("  real DDP step adds gloo allreduce (NCCL cannot init, CLAUDE.md 3),")
    print("  so treat these as a LOWER bound on wall time, and scale the")
    print("  measured 5.30 / 6.13 s/step at 32,768 by the RATIO below rather")
    print("  than using these numbers directly.")

    # PERSIST. The 2026-08-17 run of this script printed its table and wrote
    # nothing, so no 65 kb memory or throughput number has had a file behind it
    # since -- and under the standing rule that no number may be written into a
    # file unless it came from a command actually run, no schedule could legally
    # be built on them. This closes that.
    out = RESULTS / "p5_memcheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "scripts/phase5_memcheck.py",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(device),
        "device_total_gib": round(total / 2**30, 4),
        "device_free_at_start_gib": round(free / 2**30, 4),
        "batch": BATCH,
        "dtype": "fp32",
        "scan": "triton",
        "phi_granularity": "position",
        "notes": [
            "Single process, no DDP: no gloo allreduce, so s/step is a LOWER "
            "bound on real wall time per step.",
            "Peak is max_memory_allocated on ONE device; DDP runs one replica "
            "per GPU, so this is the per-GPU figure.",
            "Structural arm measured at phi_granularity='position' only. The "
            "T5c window/dual granularities are NOT covered by this run.",
            "3 iterations per cell, first discarded as warmup.",
        ],
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
