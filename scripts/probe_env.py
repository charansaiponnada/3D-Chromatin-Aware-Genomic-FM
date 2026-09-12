"""Report what this machine can actually do. Run this FIRST on any new box.

    python scripts/probe_env.py

Answers, in one run, the questions that otherwise cost days: which Python,
which CUDA, how much VRAM, does bf16 work, do the genomics packages import,
how much scratch is there, and what the job scheduler's limits are.

Its output is also exactly what a paper's compute statement needs.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def try_import(name: str) -> str:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "present")
    except Exception as exc:                                     # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       timeout=20).decode().strip()
    except Exception:                                            # noqa: BLE001
        return ""


report: dict = {}

section("python")
report["python"] = sys.version.split()[0]
report["platform"] = platform.platform()
print(f"  version   {report['python']}")
print(f"  platform  {report['platform']}")
print(f"  executable {sys.executable}")
if sys.version_info >= (3, 13):
    print("  NOTE: 3.13 is recent enough that some CUDA wheels may not exist.")
    print("        If torch or a genomics package below is MISSING, try a 3.11 env.")

section("packages")
for pkg in ("torch", "numpy", "yaml", "cooler", "pyfaidx", "h5py", "pandas", "scipy"):
    version = try_import(pkg)
    report.setdefault("packages", {})[pkg] = version
    print(f"  {pkg:<10} {version}")

section("gpu")
try:
    import torch

    report["cuda_available"] = torch.cuda.is_available()
    report["torch_cuda"] = torch.version.cuda
    print(f"  torch       {torch.__version__}")
    print(f"  cuda build  {torch.version.cuda}")
    print(f"  available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / 1024**3
            devices.append({"name": props.name, "vram_gb": round(gb, 1),
                            "capability": f"{props.major}.{props.minor}"})
            print(f"  gpu {i}       {props.name}  {gb:.1f} GB  sm_{props.major}{props.minor}")
        report["devices"] = devices
        supported = torch.cuda.is_bf16_supported()
        report["bf16"] = supported
        print(f"  bf16        {supported}"
              + ("" if supported else "   -> set train.precision: fp16 in the config"))

        # A real allocation, because "available" does not mean "usable".
        try:
            x = torch.randn(4096, 4096, device="cuda")
            y = (x @ x).sum().item()
            del x
            torch.cuda.empty_cache()
            print(f"  matmul      ok ({y:.3e})")
            report["matmul_ok"] = True
        except Exception as exc:                                 # noqa: BLE001
            print(f"  matmul      FAILED: {exc}")
            report["matmul_ok"] = False
except Exception as exc:                                         # noqa: BLE001
    print(f"  torch unavailable: {exc}")

section("optional accelerated kernels")
for pkg in ("mamba_ssm", "causal_conv1d"):
    version = try_import(pkg)
    report.setdefault("optional", {})[pkg] = version
    print(f"  {pkg:<15} {version}")
print("  (Not required. chromgraph.model uses a pure-PyTorch state-space layer,")
print("   so a MISSING here costs nothing.)")

section("storage")
for label, path in (("cwd", Path.cwd()),
                    ("CHROMGRAPH_DATA", Path(os.environ.get("CHROMGRAPH_DATA", "data"))),
                    ("home", Path.home()),
                    ("/scratch", Path("/scratch"))):
    try:
        if not path.exists():
            print(f"  {label:<16} (absent)")
            continue
        usage = shutil.disk_usage(path)
        free = usage.free / 1024**3
        print(f"  {label:<16} {path}  {free:.0f} GB free of {usage.total / 1024**3:.0f} GB")
        report.setdefault("storage", {})[label] = round(free, 1)
    except Exception:                                            # noqa: BLE001
        print(f"  {label:<16} (unreadable)")
print("  Phase 1 needs about 60 GB for hg38 plus one .mcool; the full build more.")

section("scheduler")
for tool in ("sinfo", "squeue", "sbatch", "nvidia-smi"):
    where = shutil.which(tool)
    print(f"  {tool:<12} {where or '-- not found'}")
partitions = run(["sinfo", "-o", "%P %l %G %D", "--noheader"])
if partitions:
    print("\n  partition  timelimit  gres  nodes")
    for line in partitions.splitlines()[:12]:
        print(f"    {line}")
    report["partitions"] = partitions.splitlines()
else:
    print("  No SLURM detected. If this is a rented instance, remember that an")
    print("  interrupted run resumes from results/<run>/checkpoint.pt.")

out = Path("env_probe.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwrote {out}")
