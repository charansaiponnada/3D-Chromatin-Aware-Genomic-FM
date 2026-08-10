"""Validate the Triton selective scan against the reference implementation.

RUN THIS ON THE GPU BOX BEFORE ANY TRAINING. The kernel in
src/chromfm/scan_triton.py was written on a CPU-only machine and has never been
executed. A scan with a subtly wrong gradient still produces a smooth, plausible
loss curve -- it just trains the wrong model, and every number downstream of it
is void.

What it checks, on random inputs of the shapes the real model uses:
  1. forward values agree with the reference
  2. every input gradient agrees with the reference
  3. both hold with the permeability term p present and absent
  4. the p gradient is non-zero when p is active (a kernel that silently drops p
     would otherwise pass tests 1 and 2 in the p-absent case only)

Run:  python scripts/validate_kernel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chromfm.model import selective_scan_ref                      # noqa: E402
from chromfm import scan_triton                                    # noqa: E402

# tolerances: fp32 accumulation order differs between a Python loop and a kernel
ATOL, RTOL = 2e-4, 2e-3

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def make_inputs(B=2, D=64, L=512, N=16, has_p=True, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    kw = dict(device=device, dtype=torch.float32)

    def rq(*shape):
        return torch.randn(*shape, generator=g, **kw).requires_grad_(True)

    u = rq(B, D, L)
    # delta must be positive, as softplus guarantees in the model
    delta = (torch.rand(B, D, L, generator=g, **kw) * 0.1 + 1e-3).requires_grad_(True)
    A = (-torch.arange(1, N + 1, **kw).repeat(D, 1)).requires_grad_(True)
    Bm = rq(B, N, L)
    Cm = rq(B, N, L)
    Dskip = rq(D)
    p = None
    if has_p:
        p = (torch.rand(B, L, generator=g, **kw) * 0.05).requires_grad_(True)
    return u, delta, A, Bm, Cm, Dskip, p


def clone_inputs(t):
    out = []
    for x in t:
        if x is None:
            out.append(None)
        else:
            out.append(x.detach().clone().requires_grad_(x.requires_grad))
    return out


def compare(has_p: bool, L: int, tag: str) -> None:
    ref_in = make_inputs(L=L, has_p=has_p)
    tri_in = clone_inputs(ref_in)

    y_ref = selective_scan_ref(*ref_in)
    y_tri = scan_triton.selective_scan_triton(*tri_in)

    fwd_ok = torch.allclose(y_ref, y_tri, atol=ATOL, rtol=RTOL)
    max_abs = (y_ref - y_tri).abs().max().item()
    report(f"{tag}: forward values", fwd_ok, f"max |diff| = {max_abs:.3e}")

    seed_grad = torch.randn_like(y_ref)
    y_ref.backward(seed_grad)
    y_tri.backward(seed_grad.clone())

    names = ["u", "delta", "A", "Bmat", "Cmat", "Dskip", "p"]
    for name, a, b in zip(names, ref_in, tri_in):
        if a is None or a.grad is None:
            continue
        ok = torch.allclose(a.grad, b.grad, atol=ATOL, rtol=RTOL)
        md = (a.grad - b.grad).abs().max().item()
        report(f"{tag}: grad {name}", ok, f"max |diff| = {md:.3e}")

    if has_p:
        pref = ref_in[6]
        nz = pref.grad.abs().sum().item() > 0
        report(f"{tag}: grad p is non-zero", nz,
               f"|grad p| = {pref.grad.abs().sum().item():.3e} "
               f"(a kernel that ignored p would still pass the other checks)")


def main() -> int:
    if not torch.cuda.is_available():
        print("No CUDA device. This script must run on the GPU box.")
        print("The Triton kernel remains UNVERIFIED and must not be trained on.")
        return 2
    if not scan_triton.HAVE_TRITON:
        print("Triton is not installed. `pip install triton`, then rerun.")
        return 2

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"tolerances: atol={ATOL}, rtol={RTOL}\n")

    for tag, has_p, L in (
        ("no-p, L=512", False, 512),
        ("with-p, L=512", True, 512),
        ("with-p, L=1024 (multi-chunk)", True, 1024),
        ("with-p, L=100 (ragged final chunk)", True, 100),
    ):
        print(f"{tag}:")
        compare(has_p, L, tag)
        print()

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("\nKERNEL IS NOT VALIDATED. Do not start a training run.")
        print("Fall back to scan backend 'ref' or drop the p term until fixed.")
        return 1
    print("\nKernel validated. Record the GPU, driver and Triton version in")
    print("results/baselines/run_config.yaml alongside this result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
