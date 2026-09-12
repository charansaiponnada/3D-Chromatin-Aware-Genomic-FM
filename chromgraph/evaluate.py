"""Phase 5 --- evaluation.

Two rules the metrics enforce, both of them the difference between a result and
an artefact:

  * Contact metrics are DISTANCE-STRATIFIED. Hi-C is dominated by short-range
    contacts, so one overall correlation can look strong while long-range
    prediction sits at chance.

  * The headline comparison runs with Hi-C switched OFF for every arm. Our
    model with its graph against a sequence-only baseline without one is not a
    comparison: ours was handed part of the answer. Numbers measured with the
    graph available are reported too, labelled as an upper bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chromgraph.config import Config, results_dir
from chromgraph.graph import NO_DISTAL_ARMS
from chromgraph.train import ShardDataset, collate, to_device

# Separation bands in bins. With 5 kb bins these are roughly
# <50 kb, 50-150 kb, and >150 kb.
STRATA = (("short", 0, 10), ("medium", 10, 30), ("long", 30, 10_000))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return pearson(_rank(a), _rank(b))


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size)
    return ranks


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUPRC by the step-wise definition, no sklearn dependency."""
    if labels.sum() == 0 or labels.size == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels = labels[order]
    cum_tp = np.cumsum(labels)
    precision = cum_tp / np.arange(1, labels.size + 1)
    return float((precision * labels).sum() / labels.sum())


@torch.no_grad()
def collect_predictions(model, cfg: Config, chroms: list[str], cell_lines: list[str],
                        device, use_structure: bool, max_batches: int | None = None,
                        data_root: Path | None = None):
    """Run the contact head over held-out edges and return predictions."""
    dataset = ShardDataset(cfg, chroms, cell_lines, root=data_root,
                           stride=cfg.data.nodes_per_sample)   # no overlap at eval
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    model.eval()

    preds, truth, seps = [], [], []
    for n_batch, batch in enumerate(loader):
        if max_batches is not None and n_batch >= max_batches:
            break
        batch = to_device(batch, device)
        batch["late_fusion"] = cfg.train.arm == "b4_late_fusion"
        out = model(batch, mask_frac=0.0, use_structure=use_structure)

        mask = batch["tgt_mask"]
        if not mask.any():
            continue
        b, _, t = batch["tgt_index"].shape
        bi = torch.arange(b, device=device).view(b, 1).expand(b, t)
        i = batch["tgt_index"][:, 0]
        j = batch["tgt_index"][:, 1]
        zi = out.z[bi[mask], i[mask]]
        zj = out.z[bi[mask], j[mask]]
        sep = (j - i).abs()[mask]
        logit = model.contact_head(zi, zj, sep)

        preds.append(torch.sigmoid(logit).float().cpu().numpy())
        truth.append(batch["tgt_strength"][mask].float().cpu().numpy())
        seps.append(sep.cpu().numpy())

    if not preds:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(preds), np.concatenate(truth), np.concatenate(seps)


def contact_metrics(pred: np.ndarray, true: np.ndarray, sep: np.ndarray) -> dict:
    """Distance-stratified correlation, MSE and AUPRC."""
    out: dict = {"n_edges": int(pred.size)}
    if pred.size == 0:
        return out

    # "Significant contact" = enriched over the distance-matched expectation.
    # strength = oe / (1 + oe), so strength > 0.5 is exactly oe > 1.
    labels = (true > 0.5).astype(np.float64)

    for name, lo, hi in STRATA:
        m = (sep >= lo) & (sep < hi)
        if m.sum() < 2:
            out[name] = {"n": int(m.sum())}
            continue
        out[name] = {
            "n": int(m.sum()),
            "pearson": pearson(pred[m], true[m]),
            "spearman": spearman(pred[m], true[m]),
            "mse": float(((pred[m] - true[m]) ** 2).mean()),
            "auprc": average_precision(pred[m], labels[m]),
            "positive_rate": float(labels[m].mean()),
        }

    out["overall"] = {
        "pearson": pearson(pred, true),
        "spearman": spearman(pred, true),
        "mse": float(((pred - true) ** 2).mean()),
        "auprc": average_precision(pred, labels),
    }
    return out


def evaluate_run(run_dir: Path, cfg: Config, device, splits: dict[str, list[str]],
                 cell_lines: list[str], max_batches: int | None = None,
                 data_root: Path | None = None) -> dict:
    """Load a checkpoint and evaluate it in both structural settings."""
    from chromgraph.model import build_model

    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}")

    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    arm_has_structure = cfg.train.arm not in NO_DISTAL_ARMS
    report: dict = {
        "run": run_dir.name,
        "arm": cfg.train.arm,
        "seed": cfg.seed,
        "step": state.get("step"),
        "cell_lines": cell_lines,
        "settings": {},
    }

    for split_name, chroms in splits.items():
        for setting, use_structure in (("hic_free", False), ("with_hic", True)):
            # An arm with no graph cannot be run "with Hi-C"; recording it twice
            # would imply a distinction that does not exist.
            if setting == "with_hic" and not arm_has_structure:
                continue
            pred, true, sep = collect_predictions(
                model, cfg, chroms, cell_lines, device, use_structure, max_batches,
                data_root=data_root)
            report["settings"][f"{split_name}/{setting}"] = contact_metrics(pred, true, sep)

    report["note"] = (
        "hic_free is the headline setting: every arm runs on sequence alone, so "
        "the only difference between them is what shaped their weights during "
        "pretraining. with_hic is an upper bound measured with privileged input "
        "and must not be compared against a sequence-only baseline."
    )
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def collect_comparison(root: Path | None = None) -> dict:
    """Gather every run's metrics.json into the table the write-up reads.

    Only runs that actually produced a metrics.json appear. A missing arm stays
    missing rather than becoming a zero -- the decks and the website render `??`
    for anything absent, and that is the correct behaviour.
    """
    root = root or results_dir()
    table: dict = {}
    for metrics_file in sorted(root.glob("*/metrics.json")):
        report = json.loads(metrics_file.read_text(encoding="utf-8"))
        arm = report["arm"]
        table.setdefault(arm, []).append(report)

    summary: dict = {}
    for arm, reports in table.items():
        key = "test/hic_free"
        values = [r["settings"].get(key, {}).get("long", {}).get("pearson")
                  for r in reports]
        values = [v for v in values if v is not None and not np.isnan(v)]
        summary[arm] = {
            "n_seeds": len(reports),
            "contact_r_long_hic_free_mean": float(np.mean(values)) if values else None,
            "contact_r_long_hic_free_std": float(np.std(values)) if len(values) > 1 else None,
            "runs": [r["run"] for r in reports],
        }

    out = root / "final_comparison.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
