"""Frozen-representation contact probe -- the comparison every arm can enter.

B0 and B5 train no contact head (their contact weight is 0), so their
built-in head is random and its numbers mean nothing. Even for arms that do
train one, the head was optimised jointly with the encoder under each arm's own
objective. The probe removes both differences: freeze the pretrained model,
compute its Hi-C-free representations, fit one fresh contact head of the same
architecture with the same budget and seed, and score it on held-out targets.
What differs between arms is then only the representation.

Every arm is probed in the hic_free setting, the paper's headline setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from chromgraph.config import Config
from chromgraph.evaluate import contact_metrics, save_predictions, window_keys
from chromgraph.model import ContactHead, build_model
from chromgraph.train import ShardDataset, collate, to_device

PROBE_SEED = 1234


@torch.no_grad()
def frozen_pairs(model, cfg: Config, chroms, cell_lines, device, max_samples=None,
                 data_root=None, seed=PROBE_SEED):
    """Hi-C-free representations of every held-out target pair, gathered once.

    Returns (zi, zj, sep, strength, window, keys); `window` indexes `keys`,
    which name the windows of the full (unsubsampled) dataset."""
    ds = ShardDataset(cfg, chroms, cell_lines, root=data_root, stride=cfg.data.nodes_per_sample)
    keys = window_keys(ds)
    order = np.arange(len(ds))
    if max_samples is not None and len(ds) > max_samples:
        order = np.sort(np.random.default_rng(seed).choice(len(ds), size=max_samples,
                                                           replace=False))
        ds = Subset(ds, order.tolist())
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    model.eval()
    zi_all, zj_all, sep_all, y_all, w_all = [], [], [], [], []
    for n_batch, batch in enumerate(loader):
        batch = to_device(batch, device)
        batch["late_fusion"] = cfg.train.arm == "b4_late_fusion"
        batch["fusion_hic"] = False
        z = model(batch, mask_frac=0.0, use_structure=False).z.float()
        m = batch["tgt_mask"]
        if not m.any():
            continue
        b, _, t = batch["tgt_index"].shape
        bi = torch.arange(b, device=device).view(b, 1).expand(b, t)
        i, j = batch["tgt_index"][:, 0], batch["tgt_index"][:, 1]
        zi_all.append(z[bi[m], i[m]].half().cpu())
        zj_all.append(z[bi[m], j[m]].half().cpu())
        sep_all.append((j - i).abs()[m].cpu())
        y_all.append(batch["tgt_strength"][m].float().cpu())
        w_all.append(order[(n_batch * cfg.train.batch_size + bi[m]).cpu().numpy()])
    return (torch.cat(zi_all), torch.cat(zj_all), torch.cat(sep_all), torch.cat(y_all),
            np.concatenate(w_all), keys)


def fit_head(cfg: Config, train_pairs, device, epochs=20, lr=1e-3, batch=4096,
             seed=PROBE_SEED) -> ContactHead:
    """A fresh ContactHead on frozen pairs, regressing log(O/E)."""
    torch.manual_seed(seed)
    head = ContactHead(cfg.model.d_model).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    zi, zj, sep, y = train_pairs[:4]
    target = torch.log(y.clamp(1e-4, 1 - 1e-4)) - torch.log1p(-y.clamp(1e-4, 1 - 1e-4))
    g = torch.Generator().manual_seed(seed)
    n = y.numel()
    head.train()
    for _ in range(epochs):
        for k in torch.randperm(n, generator=g).split(batch):
            pred = head(zi[k].to(device).float(), zj[k].to(device).float(), sep[k].to(device))
            loss = F.mse_loss(pred, target[k].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict(head, pairs, device, batch=8192) -> np.ndarray:
    zi, zj, sep, y = pairs[:4]
    preds = []
    for k in torch.arange(y.numel()).split(batch):
        preds.append(torch.sigmoid(head(zi[k].to(device).float(), zj[k].to(device).float(),
                                        sep[k].to(device))).cpu())
    return torch.cat(preds).numpy()


def probe_run(run_dir: Path, cfg: Config, device, splits: dict, cell_lines,
              max_train_samples: int | None = 1500, epochs: int = 20,
              data_root: Path | None = None) -> dict:
    model = build_model(cfg).to(device)
    state = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    for p in model.parameters():
        p.requires_grad_(False)

    train_pairs = frozen_pairs(model, cfg, cfg.data.chroms_train, cell_lines, device,
                               max_samples=max_train_samples, data_root=data_root)
    head = fit_head(cfg, train_pairs, device, epochs=epochs)

    report = {"run": run_dir.name, "arm": cfg.train.arm, "seed": cfg.seed,
              "step": state.get("step"), "probe": {
                  "train_pairs": int(train_pairs[3].numel()),
                  "max_train_samples": max_train_samples, "epochs": epochs,
                  "seed": PROBE_SEED, "target": "log_oe", "setting": "hic_free"},
              "settings": {}}
    for split, chroms in splits.items():
        pairs = frozen_pairs(model, cfg, chroms, cell_lines, device, data_root=data_root)
        pred = predict(head, pairs, device)
        sep, y, window, keys = pairs[2].numpy(), pairs[3].numpy(), pairs[4], pairs[5]
        report["settings"][f"{split}/probe_hic_free"] = contact_metrics(pred, y, sep)
        save_predictions(run_dir / f"probe_predictions_{split}.npz", pred, y, sep, window, keys)
    (run_dir / "probe_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
