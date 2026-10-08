"""Apply the pre-registered decision rule (docs/preregistration.md) exactly.

    python scripts/decide.py recipe                     # Section 2, seed 0
    python scripts/decide.py claims --recipe v2         # Sections 3-5, seeds 1-3, val
    python scripts/decide.py claims --recipe v2 --open-test   # Section 7, once

Reads the window-tagged prediction files written by scripts/evaluate.py
(predictions_<split>_hic_free.npz) and scripts/probe.py
(probe_predictions_<split>.npz). Writes results/decision_<mode>_<split>.json.

Every constant below is fixed by the pre-registration. Changing one after a
confirmatory number has been seen is exactly what the rule forbids. Record any
change as an amendment in the document, dated, before running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import results_dir   # noqa: E402

# --- fixed by docs/preregistration.md ----------------------------------------
N_BOOT = 10_000
BOOT_SEED = 20261008          # the date the rule was written
LONG_MIN_SEP = 30             # bins: >= 150 kb at 5 kb, the "long" stratum
CONFIRM_SEEDS = (1, 2, 3)
EXPLORE_SEED = 0
V1_FULL_VAL = 0.2066          # Section 2 threshold, seen before the rule
V1_GAP_VAL = 0.022            # Section 2 threshold, seen before the rule

ARMS = {"full": "full", "b3": "b3_shuffled_hic", "b0": "b0_dna_only", "b1": "b1_random_graph"}

HEADLINES = {
    (True, True): "Measured 3D contacts improve a DNA-only representation, through both "
                  "which regions touch and how strongly.",
    (False, True): "Measured contact topology improves a DNA-only representation; finer "
                   "differences in contact strength among the strongest contacts add no "
                   "detectable benefit.",
    (True, False): "No headline claim of improvement over sequence-only training. Full > B3 "
                   "is reported as a secondary finding with that caveat.",
    (False, False): "No positive claim. Reported as a controlled negative result: under "
                    "matched budgets, Hi-C conditioning during pretraining did not produce a "
                    "better Hi-C-free representation.",
}
# ------------------------------------------------------------------------------


def run_name(arm: str, recipe: str, seed: int) -> str:
    suffix = "" if recipe == "v1" else f"_{recipe}"
    return f"{ARMS[arm]}{suffix}_seed{seed}"


def load(run: str, kind: str, split: str, root: Path) -> dict:
    """kind: "head" (the model's own contact head) or "probe" (frozen probe)."""
    name = (f"predictions_{split}_hic_free.npz" if kind == "head"
            else f"probe_predictions_{split}.npz")
    path = root / run / name
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def window_stats(d: dict, keys: list[str]) -> np.ndarray:
    """Per-window sufficient statistics of long-range edges, aligned to `keys`.

    Rows: n, sum x, sum y, sum xx, sum yy, sum xy. A window with no long edge
    (or absent from this run) is a zero row, so resampling it adds nothing --
    the pairing across runs is by window key, never by position.
    """
    pos = {k: i for i, k in enumerate(keys)}
    m = d["sep"] >= LONG_MIN_SEP
    x = d["pred"][m].astype(np.float64)
    y = d["true"][m].astype(np.float64)
    w = np.array([pos[k] for k in d["keys"][d["window"][m]]], dtype=np.int64)
    out = np.zeros((len(keys), 6))
    for col, v in enumerate((np.ones_like(x), x, y, x * x, y * y, x * y)):
        np.add.at(out[:, col], w, v)
    return out


def pearson_from(s: np.ndarray) -> np.ndarray:
    """Pearson r from summed statistics; s (..., 6)."""
    n, sx, sy, sxx, syy, sxy = np.moveaxis(s, -1, 0)
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


class Bootstrap:
    """One shared set of window resamples, so every comparison is paired."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        rng = np.random.default_rng(BOOT_SEED)
        w = len(keys)
        self.counts = rng.multinomial(w, np.full(w, 1.0 / w), size=N_BOOT).astype(np.float64)

    def r(self, stats: np.ndarray) -> tuple[float, np.ndarray]:
        """Point estimate and N_BOOT resampled values of r for one run."""
        return float(pearson_from(stats.sum(0))), pearson_from(self.counts @ stats)


def compare(boot: Bootstrap, x_runs: list[np.ndarray], y_runs: list[np.ndarray],
            label: str) -> dict:
    """X beats Y: seed-averaged paired-bootstrap 95% CI > 0 AND X > Y in a
    majority of seeds (2 of 3). Seeds pair by position (seed s vs seed s)."""
    xs = [boot.r(s) for s in x_runs]
    ys = [boot.r(s) for s in y_runs]
    point = float(np.mean([p for p, _ in xs]) - np.mean([p for p, _ in ys]))
    dist = np.mean([b for _, b in xs], axis=0) - np.mean([b for _, b in ys], axis=0)
    lo, hi = (float(v) for v in np.nanpercentile(dist, [2.5, 97.5]))
    per_seed = [float(a[0] - b[0]) for a, b in zip(xs, ys)]
    wins = sum(d > 0 for d in per_seed)
    passed = bool(lo > 0 and wins * 2 > len(per_seed))
    return {"comparison": label, "diff": point, "ci95": [lo, hi], "per_seed_diff": per_seed,
            "seeds_ahead": f"{wins}/{len(per_seed)}", "pass": passed,
            "x_r": [p for p, _ in xs], "y_r": [p for p, _ in ys]}


def union_keys(*datas: dict) -> list[str]:
    return sorted(set().union(*(set(d["keys"].tolist()) for d in datas)))


def decide_claims(recipe: str, seeds: tuple[int, ...], split: str, root: Path) -> dict:
    runs = {(arm, kind): [load(run_name(arm, recipe, s), kind, split, root) for s in seeds]
            for arm, kind in (("full", "head"), ("b3", "head"),
                              ("full", "probe"), ("b3", "probe"), ("b0", "probe"),
                              ("b1", "probe"))}
    keys = union_keys(*(d for ds in runs.values() for d in ds))
    boot = Bootstrap(keys)
    st = {k: [window_stats(d, keys) for d in ds] for k, ds in runs.items()}

    a = compare(boot, st["full", "head"], st["b3", "head"], "A: Full > B3 (head)")
    b = [compare(boot, st[x, "probe"], st[y, "probe"], f"B: {x} > {y} (probe)")
         for x in ("full", "b3") for y in ("b0", "b1")]
    claim_a = a["pass"]
    claim_b = all(c["pass"] for c in b)
    return {"claim_A": claim_a, "claim_B": claim_b,
            "headline": HEADLINES[claim_a, claim_b], "comparisons": [a] + b,
            "n_windows": len(keys)}


def decide_recipe(split: str, root: Path) -> dict:
    s = EXPLORE_SEED
    d = {(arm, r): load(run_name(arm, r, s), "head", split, root)
         for arm in ("full", "b3") for r in ("v1", "v2")}
    keys = union_keys(*d.values())
    boot = Bootstrap(keys)
    st = {k: window_stats(v, keys) for k, v in d.items()}
    full_v2 = boot.r(st["full", "v2"])[0]
    gap_v2 = compare(boot, [st["full", "v2"]], [st["b3", "v2"]], "full_v2 > b3_v2 (head)")
    gap_v1 = compare(boot, [st["full", "v1"]], [st["b3", "v1"]], "full > b3, v1 (head)")
    # Section 2: a single seed, so the CI alone carries the decision.
    choose_v2 = bool(full_v2 > V1_FULL_VAL and gap_v2["ci95"][0] > 0
                     and gap_v2["diff"] > V1_GAP_VAL)
    return {"full_v2_r": full_v2, "threshold_full_v1": V1_FULL_VAL,
            "gap_v2": gap_v2, "threshold_gap_v1": V1_GAP_VAL,
            "gap_v1_recomputed": gap_v1, "chosen_recipe": "v2" if choose_v2 else "v1",
            "n_windows": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("recipe", "claims"))
    ap.add_argument("--recipe", choices=("v1", "v2"), help="frozen recipe (claims mode)")
    ap.add_argument("--open-test", action="store_true",
                    help="Section 7: score the sealed test split. Once, at the very end.")
    ap.add_argument("--exploratory-seeds", type=int, nargs="+", default=None,
                    help="run the claims on other seeds (e.g. 0); labelled EXPLORATORY")
    args = ap.parse_args()

    root = results_dir()
    split = "test" if args.open_test else "val"
    try:
        if args.mode == "recipe":
            if args.open_test:
                ap.error("the recipe is chosen on val only")
            out = decide_recipe(split, root)
        else:
            if not args.recipe:
                ap.error("claims mode needs --recipe (chosen by `decide.py recipe`)")
            seeds = tuple(args.exploratory_seeds or CONFIRM_SEEDS)
            out = decide_claims(args.recipe, seeds, split, root)
            out.update(recipe=args.recipe, seeds=list(seeds),
                       status=("EXPLORATORY" if args.exploratory_seeds else "CONFIRMATORY"))
    except FileNotFoundError as e:
        print(f"cannot decide: missing {e}. Evaluate and probe every required run first.")
        return 2

    out.update(split=split, n_boot=N_BOOT, boot_seed=BOOT_SEED, rule="docs/preregistration.md")
    tag = "" if not args.exploratory_seeds else "_exploratory"
    path = root / f"decision_{args.mode}_{split}{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in out.items() if k != "comparisons"}, indent=2))
    for c in out.get("comparisons", []):
        print(f"  {'PASS' if c['pass'] else 'fail'}  {c['comparison']:<26} "
              f"diff {c['diff']:+.4f}  CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
              f"seeds ahead {c['seeds_ahead']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
