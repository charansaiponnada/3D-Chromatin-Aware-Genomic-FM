#!/usr/bin/env python
"""B0(a) -- single-GPU allocator smoke test.

THE QUESTION
------------
Every 65,536 bp launch attempt has died on:
  NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
  at CUDACachingAllocator.cpp:983
under concurrent load from an unrelated project on both devices. CLAUDE.md 3:
nvidia-smi is broken on this box (NVML/kernel-module mismatch); the caching
allocator's own NVML probe fails the same way when it reaches that path under
memory pressure. Single-GPU fallback (no DDP, no gloo, no cross-device
allocator traffic) has been the untried mitigation for over a week.

THIS SCRIPT DOES NOT TRAIN ANYTHING. It restricts the process to ONE device,
builds the real model at the real 65,536 bp window, and runs a FEW forward +
backward + optimizer steps -- allocate, step, free -- to see whether the
allocator's NVML path still fires with only one device visible to this
process. Minutes, not hours. Output is written to disk so the result survives
whether or not it worked.

Run:  CUDA_VISIBLE_DEVICES=0 ./3d-gen/bin/python -u scripts/b0_single_gpu_smoke.py
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

BATCH = 2
VOCAB = 16
D_STRUCT_RAW = 8
N_STEPS = 5          # allocate, step, free -- a smoke test, not training


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=65_536)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    WINDOW = args.window
    OUT = Path(args.out) if args.out else (
        REPO / "results" / f"b0_single_gpu_smoke_{WINDOW}.json")

    result = {
        "purpose": "B0(a) single-GPU allocator smoke test, RESEARCH_PLAN_2026-08-26 Phase B0",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "window_bp": WINDOW, "batch": BATCH, "n_steps": N_STEPS,
    }

    if not torch.cuda.is_available():
        result["outcome"] = "NO_CUDA"
        print("No CUDA device visible to this process.")
        OUT.write_text(json.dumps(result, indent=2))
        return 2

    n_visible = torch.cuda.device_count()
    result["n_visible_devices"] = n_visible
    print(f"CUDA_VISIBLE_DEVICES={result['cuda_visible_devices']!r}, "
          f"{n_visible} device(s) visible to this process")
    if n_visible != 1:
        print(f"WARNING: expected exactly 1 visible device for a single-GPU "
              f"fallback test, got {n_visible}. Proceeding on device 0 anyway.")

    device = torch.device("cuda", 0)

    try:
        free0, total0 = torch.cuda.mem_get_info(device)
        result["mem_get_info_before"] = {"free_gib": free0 / 2**30,
                                          "total_gib": total0 / 2**30}
        print(f"mem_get_info before: free {free0/2**30:.2f} / "
              f"total {total0/2**30:.2f} GiB")
    except Exception as e:                              # noqa: BLE001
        result["mem_get_info_before_error"] = f"{type(e).__name__}: {e}"
        print(f"mem_get_info FAILED before any allocation: "
              f"{type(e).__name__}: {e}")

    try:
        from chromfm.model import BiMambaLM, ModelConfig, use_scan
        use_scan("triton")
        cfg = ModelConfig(structural=True)
        t_build0 = time.time()
        model = BiMambaLM(cfg).to(device)
        if args.grad_checkpoint:
            from train import CheckpointedLM
            model = CheckpointedLM(model).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        result["model_build_s"] = time.time() - t_build0
        result["grad_checkpoint"] = bool(args.grad_checkpoint)
        print(f"model built on {device}, structural=True, "
              f"window {WINDOW:,} bp, grad_checkpoint={args.grad_checkpoint}, "
              f"in {result['model_build_s']:.2f}s")

        tokens = torch.randint(2, 6, (BATCH, WINDOW), device=device)
        target = torch.randint(2, 6, (BATCH, WINDOW), device=device)
        phi = torch.randn(BATCH, WINDOW, D_STRUCT_RAW, device=device)
        phi_valid = torch.ones(BATCH, WINDOW, dtype=torch.bool, device=device)
        symmetry = torch.ones(D_STRUCT_RAW, device=device)

        step_times = []
        for i in range(N_STEPS):
            torch.cuda.synchronize()
            t0 = time.time()
            logits = model(tokens, phi=phi, phi_valid=phi_valid, symmetry=symmetry)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), target.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            step_times.append(dt)
            print(f"  step {i}  loss {loss.item():.4f}  {dt:.2f}s")

        peak = torch.cuda.max_memory_allocated(device) / 2**30
        result["step_times_s"] = step_times
        result["peak_gib"] = peak
        result["outcome"] = "OK"
        print(f"\npeak allocated: {peak:.2f} GiB")
        print("NO nvmlInit_v2_ assertion. Single-GPU fallback WORKS under "
              "current conditions.")

        del model, opt, tokens, target, phi, phi_valid, symmetry
        torch.cuda.empty_cache()

        free1, total1 = torch.cuda.mem_get_info(device)
        result["mem_get_info_after_free"] = {"free_gib": free1 / 2**30,
                                              "total_gib": total1 / 2**30}
        print(f"mem_get_info after free: free {free1/2**30:.2f} / "
              f"total {total1/2**30:.2f} GiB")

    except RuntimeError as e:
        msg = str(e)
        result["outcome"] = "RUNTIME_ERROR"
        result["error"] = msg[:2000]
        result["is_nvml_assertion"] = ("nvmlInit_v2" in msg or "NVML_SUCCESS" in msg)
        print(f"\nRuntimeError: {msg[:500]}")
        if result["is_nvml_assertion"]:
            print("\nTHE NVML ASSERTION STILL FIRES on a single visible device. "
                  "Single-GPU fallback does NOT mitigate this failure mode.")
        else:
            print("\nA RuntimeError occurred, but it is NOT the nvmlInit_v2_ "
                  "assertion this test was checking for.")
    except Exception as e:                              # noqa: BLE001
        result["outcome"] = "OTHER_ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-2000:]
        print(f"\n{type(e).__name__}: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")
    return 0 if result.get("outcome") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
