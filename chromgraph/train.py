"""Phases 3-4 --- dataset, losses, and the training loop.

Everything an experimental arm changes goes through `cfg.train.arm`, so the
seven arms share one code path. That is not tidiness; it is what makes the
comparison a comparison.

Checkpointing and resume are built in from the start rather than retrofitted:
spot instances get interrupted and idle cullers kill sessions, and a run that
cannot resume is a run that has to start over.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chromgraph.config import Config, data_dir, results_dir
from chromgraph.data import load_chromosome
from chromgraph.graph import (NO_DISTAL_ARMS, build_sample,
                              distance_matched_negatives, sample_starts)
from chromgraph.model import build_model


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

class ShardDataset(Dataset):
    """Windows drawn from the built chromosomes of one split."""

    def __init__(self, cfg: Config, chroms: list[str], cell_lines: list[str],
                 root: Path | None = None, stride: int | None = None):
        self.cfg = cfg
        self.root = root or data_dir()
        self.index: list[tuple[str, str, int]] = []
        self._cache: dict[tuple[str, str], dict] = {}

        for cell_line in cell_lines:
            for chrom in chroms:
                try:
                    chrom_data = self._chrom(cell_line, chrom)
                except FileNotFoundError:
                    continue
                starts = sample_starts(len(chrom_data["usable"]),
                                       cfg.data.nodes_per_sample,
                                       chrom_data["usable"], stride=stride)
                self.index += [(cell_line, chrom, s) for s in starts]

        if not self.index:
            raise RuntimeError(
                f"no samples for chroms={chroms} cell_lines={cell_lines}. "
                f"Has scripts/build_data.py run for these?")

    def _chrom(self, cell_line: str, chrom: str) -> dict:
        key = (cell_line, chrom)
        if key not in self._cache:
            self._cache[key] = load_chromosome(self.root, cell_line, chrom,
                                               self.cfg.data.bin_size)
        return self._cache[key]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        cell_line, chrom, start = self.index[i]
        return build_sample(self.cfg, self._chrom(cell_line, chrom),
                            cell_line, chrom, start, arm=self.cfg.train.arm)


def collate(samples: list) -> dict:
    """Pad the per-sample edge lists into fixed-width tensors.

    Padding rather than a flat batch vector keeps the model's indexing simple
    and costs little: edge counts are within a factor of two of each other
    because top-k caps them.
    """
    b = len(samples)
    n = samples[0].n_nodes
    max_e = max(1, max(s.edge_index.shape[1] for s in samples))
    max_t = max(1, max(s.tgt_index.shape[1] for s in samples))

    edge_index = np.zeros((b, 2, max_e), np.int64)
    edge_strength = np.zeros((b, max_e), np.float32)
    edge_sep = np.zeros((b, max_e), np.int64)
    edge_mask = np.zeros((b, max_e), bool)
    edge_is_local = np.zeros((b, max_e), bool)
    tgt_index = np.zeros((b, 2, max_t), np.int64)
    tgt_strength = np.zeros((b, max_t), np.float32)
    tgt_mask = np.zeros((b, max_t), bool)

    for k, s in enumerate(samples):
        e = s.edge_index.shape[1]
        edge_index[k, :, :e] = s.edge_index
        edge_strength[k, :e] = s.edge_strength
        edge_sep[k, :e] = s.edge_sep
        edge_mask[k, :e] = True
        edge_is_local[k, :e] = s.edge_is_local
        t = s.tgt_index.shape[1]
        tgt_index[k, :, :t] = s.tgt_index
        tgt_strength[k, :t] = s.tgt_strength
        tgt_mask[k, :t] = True

    return {
        "seq": torch.from_numpy(np.stack([s.seq for s in samples])),
        "node_ok": torch.from_numpy(np.stack([s.node_ok for s in samples])),
        "node_hic_feat": torch.from_numpy(np.stack([s.node_hic_feat for s in samples])),
        "edge_index": torch.from_numpy(edge_index),
        "edge_strength": torch.from_numpy(edge_strength),
        "edge_sep": torch.from_numpy(edge_sep),
        "edge_mask": torch.from_numpy(edge_mask),
        "edge_is_local": torch.from_numpy(edge_is_local),
        "tgt_index": torch.from_numpy(tgt_index),
        "tgt_strength": torch.from_numpy(tgt_strength),
        "tgt_mask": torch.from_numpy(tgt_mask),
        "n_nodes": n,
    }


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #

@dataclass
class Losses:
    dna: torch.Tensor
    contrast: torch.Tensor
    contact: torch.Tensor
    total: torch.Tensor

    def items(self) -> dict[str, float]:
        # detach first: these are read for logging while the graph is still live
        return {"dna": float(self.dna.detach()), "contrast": float(self.contrast.detach()),
                "contact": float(self.contact.detach()), "total": float(self.total.detach())}


def masked_window_loss(out, batch) -> torch.Tensor:
    """Rebuild a blanked window's embedding from its graph neighbours.

    This replaces per-base masked reconstruction. A d_model-dimensional pooled
    vector cannot carry 5,000 bases of sequence, so a per-base head reading
    only h_i is information-theoretically unable to succeed -- its loss would
    sit flat forever while looking like a training bug. Masking the whole
    window is well posed, and the only route to the answer runs through the
    contacts, which is what we actually want to test.
    """
    mask = out.masked_nodes & batch["node_ok"]
    if not mask.any():
        return out.z.sum() * 0.0
    return F.mse_loss(out.h_recon[mask], out.h[mask].detach())


def contact_loss(model, out, batch) -> torch.Tensor:
    """Binary cross-entropy on the edges the encoder never saw.

    Known limitation, measured in the smoke test rather than assumed: the
    targets are soft strengths whose mean is about 0.5 by construction, since
    strength = oe / (1 + oe) and observed-over-expected averages to 1 at every
    separation. BCE against a 0.5 target floors at the target entropy, about
    0.693, which leaves this term very little dynamic range.

    If Phase 4 shows the contact term is not driving learning, the fix is to
    regress log(O/E) with MSE instead, or to split this into a binary
    "significant contact" label plus a separate strength regression. Both are
    changes that want real data to choose between, so neither is made blind.
    """
    tgt_mask = batch["tgt_mask"]
    if not tgt_mask.any():
        return out.z.sum() * 0.0

    b, _, t = batch["tgt_index"].shape
    z = out.z
    bi = torch.arange(b, device=z.device).view(b, 1).expand(b, t)
    i = batch["tgt_index"][:, 0]
    j = batch["tgt_index"][:, 1]
    zi = z[bi[tgt_mask], i[tgt_mask]]
    zj = z[bi[tgt_mask], j[tgt_mask]]
    sep = (j - i).abs()[tgt_mask]
    logits = model.contact_head(zi, zj, sep)
    return F.binary_cross_entropy_with_logits(logits, batch["tgt_strength"][tgt_mask])


def contrastive_loss(model, out, batch, cfg: Config,
                     rng: np.random.Generator) -> torch.Tensor:
    """Touching pairs should look alike; distance-matched pairs should not.

    Positives come from the HELD-OUT edges, not the conditioning ones. If the
    positive were an edge the encoder had already attended over, the loss could
    be satisfied by the attention having mixed the two nodes, and the objective
    would measure the graph rather than the representation.

    Negatives are other pairs spanning the same number of bins. Sampled freely
    they would sit further apart than the positives on average, and the model
    could minimise this by measuring distance -- which it can already read off
    the input.
    """
    tgt_mask = batch["tgt_mask"]
    if not tgt_mask.any():
        return out.z.sum() * 0.0

    z = F.normalize(model.project(out.z), dim=-1)
    b, n = z.shape[0], z.shape[1]
    tau = cfg.train.temperature
    terms = []

    for k in range(b):
        present = tgt_mask[k].nonzero(as_tuple=True)[0]
        if present.numel() == 0:
            continue
        pick = present[int(rng.integers(present.numel()))]
        pi = int(batch["tgt_index"][k, 0, pick])
        pj = int(batch["tgt_index"][k, 1, pick])
        negatives = distance_matched_negatives(n, pi, pj, count=16, rng=rng)
        if negatives.shape[1] == 0:
            continue
        neg = torch.from_numpy(negatives).to(z.device)
        pos_sim = (z[k, pi] * z[k, pj]).sum() / tau
        neg_sim = (z[k, neg[0]] * z[k, neg[1]]).sum(-1) / tau
        logits = torch.cat([pos_sim.view(1), neg_sim])
        terms.append(-F.log_softmax(logits, dim=0)[0])

    if not terms:
        return out.z.sum() * 0.0
    return torch.stack(terms).mean()


def compute_losses(model, out, batch, cfg: Config, rng) -> Losses:
    arm = cfg.train.arm
    t = cfg.train
    dna = masked_window_loss(out, batch)
    contrast = contrastive_loss(model, out, batch, cfg, rng)
    contact = contact_loss(model, out, batch)

    # B5 uses Hi-C only as an alignment target: the contrastive term survives,
    # the contact-prediction term does not, and the encoder never sees a graph.
    w_contact = 0.0 if arm == "b5_contrastive_only" else t.lambda_contact
    # B0 has no structural signal at all, so neither structural loss is defined.
    if arm == "b0_dna_only":
        w_contact = 0.0
        w_contrast = 0.0
    else:
        w_contrast = t.lambda_contrast

    total = t.lambda_dna * dna + w_contrast * contrast + w_contact * contact
    return Losses(dna, contrast, contact, total)


# --------------------------------------------------------------------------- #
# loop
# --------------------------------------------------------------------------- #

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                            # noqa: BLE001
        return "unknown"


def lr_at(step: int, cfg: Config) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / max(1, t.warmup_steps)
    progress = (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps)
    return t.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg: Config, run_name: str, device: torch.device,
          max_steps: int | None = None, out_root: Path | None = None,
          data_root: Path | None = None) -> Path:
    out = (out_root or results_dir()) / run_name
    out.mkdir(parents=True, exist_ok=True)

    # Written BEFORE the first step. A run that dies halfway still leaves a
    # complete record of what it was.
    cfg.save(out / "run_config.yaml")
    (out / "provenance.json").write_text(json.dumps({
        "commit": git_commit(),
        "arm": cfg.train.arm,
        "seed": cfg.seed,
        "control_seed": cfg.control_seed,
        "device": str(device),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    train_set = ShardDataset(cfg, cfg.data.chroms_train, cfg.data.cell_lines,
                             root=data_root)
    loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0, drop_last=True)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    use_amp = device.type == "cuda" and cfg.train.precision == "bf16"
    steps = max_steps or cfg.train.steps
    start_step = 0

    ckpt_path = out / "checkpoint.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"  resumed from step {start_step}")

    history: list[dict] = []
    model.train()
    iterator = iter(loader)
    t0 = time.time()

    for step in range(start_step, steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)

        opt.zero_grad(set_to_none=True)
        accum = max(1, cfg.train.grad_accum_steps)
        agg = {"dna": 0.0, "contrast": 0.0, "contact": 0.0, "total": 0.0}

        for _ in range(accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = to_device(batch, device)
            batch["late_fusion"] = cfg.train.arm == "b4_late_fusion"

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out_model = model(batch, mask_frac=cfg.train.mask_frac,
                                  use_structure=cfg.train.arm not in NO_DISTAL_ARMS)
                losses = compute_losses(model, out_model, batch, cfg, rng)

            (losses.total / accum).backward()
            for key, value in losses.items().items():
                agg[key] += value / accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0 or step == steps - 1:
            rate = (step - start_step + 1) / max(1e-9, time.time() - t0)
            row = {"step": step, "lr": lr_at(step, cfg), "steps_per_sec": rate, **agg}
            if str(device).startswith("cuda"):
                row["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
            history.append(row)
            print(f"  step {step:>6}  total {agg['total']:.4f}  "
                  f"dna {agg['dna']:.4f}  contrast {agg['contrast']:.4f}  "
                  f"contact {agg['contact']:.4f}  ({rate:.2f} it/s)")
            (out / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")

        if (step + 1) % cfg.train.ckpt_every == 0 or step == steps - 1:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "step": step + 1, "arm": cfg.train.arm}, ckpt_path)

    print(f"  done in {(time.time() - t0) / 60:.1f} min -> {out}")
    return out
