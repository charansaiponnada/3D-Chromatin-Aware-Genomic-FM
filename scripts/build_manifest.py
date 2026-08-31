#!/usr/bin/env python
"""Machine-readable provenance manifest for every logged result.

Written 2026-08-31 in response to peer-review major comment 3: a reader could
not tell which code, index, window or granularity produced each logged number.
This regenerates the manifest from artefacts on disk -- it never hardcodes a
number, so it cannot drift from what it describes (CLAUDE.md rule 1).

Run:  PYTHONPATH=src python scripts/build_manifest.py
Out:  results/MANIFEST.json  (+ a human-readable summary on stdout)
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PROCESSED = REPO / "data" / "processed"


def sha256(p: Path, cap: int = 1 << 30) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        read = 0
        while chunk := f.read(1 << 20):
            h.update(chunk); read += len(chunk)
            if read >= cap:
                return h.hexdigest() + f"-first{cap}B"
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def index_summary(p: Path) -> dict:
    z = np.load(p, allow_pickle=True)
    keys = set(z.files)
    d = {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
         "schema": "multichrom" if "chrom_names" in keys else "single-chrom"}
    if "window" in keys:
        d["window"] = int(z["window"])
    if d["schema"] == "multichrom":
        d["chrom_names"] = [str(c) for c in z["chrom_names"]]
        d["chrom_roles"] = [str(c) for c in z["chrom_roles"]]
        d["phi_standardisation"] = str(z["phi_standardisation"])
        for s in ("train", "val", "test"):
            k = f"{s}_starts"
            if k in keys:
                d[f"n_{s}"] = int(len(z[k]))
    else:
        for s in ("train", "val", "test"):
            if s in keys:
                d[f"n_{s}"] = int(len(z[s]))
    return d


def run_summary(d: Path) -> dict | None:
    mj, rc = d / "metrics.json", d / "run_config.yaml"
    if not mj.exists():
        return None
    rec: dict = {"run": str(d.relative_to(REPO)), "has_checkpoint": (d / "checkpoint.pt").exists()}
    try:
        m = json.load(open(mj))          # a LIST of per-step records
        rec["n_eval_records"] = len(m)
        if m:
            last = m[-1]
            rec["last_step"] = last.get("step")
            rec["last_val_bits"] = last.get("val_bits_per_nucleotide")
        rec["metrics_sha256"] = sha256(mj)
    except Exception as e:
        rec["metrics_error"] = str(e)
    if rc.exists():
        want = ("status", "window", "structural", "seed", "phi_granularity", "n_train_windows",
                "dt_min", "dt_max", "dt_floor", "use_permeability", "steps", "scan_backend")
        got = {}
        for line in open(rc):
            if ":" in line and not line.strip().startswith("#"):
                k, _, v = line.partition(":")
                if k.strip() in want:
                    got[k.strip()] = v.strip()
        rec["run_config"] = got
        rec["run_config_sha256"] = sha256(rc)
    if (d / "checkpoint.pt").exists():
        rec["checkpoint_sha256"] = sha256(d / "checkpoint.pt", cap=1 << 26)
    return rec


def main() -> int:
    man: dict = {
        "generated_by": "scripts/build_manifest.py",
        "git_commit": git("rev-parse", "HEAD"),
        "git_describe": git("describe", "--always", "--dirty"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        man["torch"] = torch.__version__
    except Exception:
        pass

    man["indices"] = [index_summary(p) for p in sorted(PROCESSED.glob("dataset_index*.npz"))]

    runs = []
    for base in (RESULTS / "baselines", RESULTS / "novel_model"):
        if not base.exists():
            continue
        for d in sorted(base.rglob("*")):
            if d.is_dir() and (d / "metrics.json").exists():
                r = run_summary(d)
                if r:
                    runs.append(r)
    man["runs"] = runs

    out = RESULTS / "MANIFEST.json"
    out.write_text(json.dumps(man, indent=2) + "\n")

    print(f"commit {man['git_commit']}  dirty={man['git_dirty']}")
    print(f"\nindices ({len(man['indices'])}):")
    for i in man["indices"]:
        print(f"  {i['file']:<34} {i['schema']:<13} window={i.get('window')} "
              f"train/val/test={i.get('n_train')}/{i.get('n_val')}/{i.get('n_test')}")
    print(f"\nruns with metrics.json ({len(runs)}):")
    for r in runs:
        cfg = r.get("run_config", {})
        print(f"  {r['run']:<44} status={cfg.get('status')} last_step={r.get('last_step')} "
              f"cfg_window={cfg.get('window')} struct={cfg.get('structural')} "
              f"ckpt={'Y' if r['has_checkpoint'] else 'n'}")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
