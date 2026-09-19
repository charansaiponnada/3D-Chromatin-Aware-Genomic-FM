"""End-to-end check on synthetic data. No downloads, no GPU, no real Hi-C.

    python scripts/smoke_test.py

This is the gate that says the code is internally consistent: it fabricates a
chromosome in exactly the on-disk format Phase 1 produces, then runs Phase 2,
Phase 3 and Phase 5 over it and asserts the invariants the science depends on.

It cannot tell you whether the model works on real chromatin. It can tell you
that the pipeline runs, the shapes line up, the controls do what they claim,
the held-out edges stay held out, and the loss goes down --- which is what you
want to know before spending money on a GPU.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromgraph.config import load_config                                # noqa: E402
from chromgraph.data import N_CODE, detrend                              # noqa: E402
from chromgraph.evaluate import (average_precision, contact_metrics,     # noqa: E402
                                 pearson)
from chromgraph.graph import ARMS, build_sample, sample_starts           # noqa: E402
from chromgraph.model import build_model, scatter_softmax                # noqa: E402
from chromgraph.train import (ShardDataset, collate, compute_losses,     # noqa: E402
                              pick_device, to_device, train)

failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  --  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


# --------------------------------------------------------------------------- #
# a fake chromosome, in the real on-disk format
# --------------------------------------------------------------------------- #

def write_fake_chromosome(root: Path, cell_line: str, chrom: str, cfg,
                          n_bins: int = 240, seed: int = 7) -> None:
    """Synthesise bins.npz / pixels.npz / seq.npy / qc.json.

    The fake Hi-C carries the same three ingredients real data does --- distance
    decay, compartment preference, and per-locus anchor strength --- because
    without the last two every observed/expected ratio comes out at exactly 1.0,
    the contact bias becomes a per-row constant, and softmax cancels it. That is
    not hypothetical; it is what the explainer's worked example printed on its
    first run.
    """
    rng = np.random.default_rng(seed)
    bin_size = cfg.data.bin_size
    span = cfg.data.nodes_per_sample

    # Anchor windows carry a motif AND attract contacts. Without that link the
    # DNA would be independent of the Hi-C, the contact and contrastive losses
    # would have nothing to learn by construction, and a flat loss would look
    # like a bug rather than an absent signal.
    seq = rng.integers(0, 4, size=(n_bins, bin_size), dtype=np.uint8)
    seq[:3] = N_CODE                                   # a telomere-like N block
    is_anchor = np.zeros(n_bins, bool)
    is_anchor[::7] = True
    motif = np.array([0, 1, 1, 1, 2, 1, 0, 3], np.uint8)          # a CTCF stand-in
    for i in np.flatnonzero(is_anchor):
        for offset in range(0, bin_size - motif.size, bin_size // 4):
            seq[i, offset: offset + motif.size] = motif

    compartment = (np.arange(n_bins) // 8) % 2
    anchor = np.where(is_anchor, 1.4, 0.8) * rng.uniform(0.95, 1.05, size=n_bins)

    rows, cols, raws = [], [], []
    max_sep = min(60, span - 1)
    for i in range(n_bins):
        for step in range(1, max_sep + 1):
            j = i + step
            if j >= n_bins:
                break
            background = 100.0 / (step ** 1.5)
            same = 1.30 if compartment[i] == compartment[j] else 0.75
            rows.append(i)
            cols.append(j)
            raws.append(background * same * anchor[i] * anchor[j]
                        * rng.uniform(0.9, 1.1))
    row = np.asarray(rows, np.int32)
    col = np.asarray(cols, np.int32)
    raw = np.asarray(raws, np.float32)

    # Loops between anchor pairs, at a separation that FITS inside one training
    # window -- otherwise the planted signal is invisible to the model and the
    # test asserts something it cannot possibly observe.
    loop_sep = max(cfg.data.min_separation + 1, span // 2)
    both_anchor = is_anchor[row] & is_anchor[col] & ((col - row) == loop_sep)
    raw[both_anchor] *= 5.0

    # The real detrend, not a copy of it.
    oe, strength, expected = detrend(row, col, raw, max_sep, np.ones(n_bins, bool))

    n_frac = (seq == N_CODE).mean(axis=1).astype(np.float32)
    coverage = np.bincount(row, weights=raw.astype(np.float64), minlength=n_bins)
    coverage += np.bincount(col, weights=raw.astype(np.float64), minlength=n_bins)
    usable = (n_frac <= cfg.data.max_n_frac) & (coverage > 0)

    out = root / "processed" / cell_line / f"{chrom}_{bin_size}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "bins.npz", n_frac=n_frac,
                        coverage=coverage.astype(np.float32), usable=usable)
    np.savez_compressed(out / "pixels.npz", row=row, col=col, raw=raw, oe=oe,
                        strength=strength, expected=expected)
    np.save(out / "seq.npy", seq)
    (out / "qc.json").write_text(json.dumps({
        "cell_line": cell_line, "chrom": chrom, "bin_size": bin_size,
        "n_bins": int(n_bins), "n_pixels": int(row.size),
        "synthetic": True,
    }, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main() -> int:
    device = pick_device("auto")
    tmp = Path(tempfile.mkdtemp(prefix="chromgraph_smoke_"))
    print(f"scratch {tmp}\ndevice  {device}\n")

    try:
        cfg = load_config(**{
            "data.nodes_per_sample": 24,
            "data.bin_size": 256,          # keep the fake DNA small; ratios unchanged
            "data.cell_lines": ["FAKE"],
            "data.chroms_train": ["chrA"],
            "data.chroms_val": ["chrB"],
            "data.chroms_test": ["chrC"],
            "data.pilot_chroms": ["chrA"],
            "model.d_model": 32,
            "model.conv_depth": 3,
            "model.n_heads": 2,
            "model.d_ff": 64,
            "model.encoder_layers": 1,
            "model.block_layers": 2,
            "train.batch_size": 2,
            "train.grad_accum_steps": 1,
            "train.log_every": 5,
            "train.ckpt_every": 10,
            "train.warmup_steps": 2,
            "train.lr": 3e-3,
        })
        for chrom in ("chrA", "chrB", "chrC"):
            write_fake_chromosome(tmp, "FAKE", chrom, cfg)

        print("PHASE 1  detrending")
        # 4 valid bins, 3 pairs at separation 1, only one observed (count 6).
        # Unobserved pairs are zeros: expected[1] = 6/3 = 2, not 6/1 = 6.
        _, _, exp = detrend(np.array([0]), np.array([1]), np.array([6.0], np.float32),
                            1, np.ones(4, bool))
        check("expected counts unobserved valid pairs as zeros",
              abs(float(exp[1]) - 2.0) < 1e-6, f"expected[1] = {float(exp[1])}")
        _, _, exp = detrend(np.array([0]), np.array([1]), np.array([6.0], np.float32),
                            1, np.array([True, True, False, False]))
        check("ICE-filtered bins are excluded from the denominator",
              abs(float(exp[1]) - 6.0) < 1e-6, f"expected[1] = {float(exp[1])}")

        print("PHASE 2  graph construction")
        from chromgraph.data import load_chromosome
        chrom_data = load_chromosome(tmp, "FAKE", "chrA", cfg.data.bin_size)
        starts = sample_starts(len(chrom_data["usable"]), cfg.data.nodes_per_sample,
                               chrom_data["usable"])
        check("windows are produced", len(starts) > 0, f"{len(starts)} windows")

        by_arm = {}
        for arm in ARMS:
            s = build_sample(cfg, chrom_data, "FAKE", "chrA", starts[0], arm=arm)
            s.assert_disjoint()
            by_arm[arm] = s
        check("every arm builds and passes the disjointness assertion", True,
              f"{len(ARMS)} arms")

        full, shuf, dist = by_arm["full"], by_arm["b3_shuffled_hic"], by_arm["b2_distance_only"]
        check("b3 keeps the edge count identical to full",
              full.edge_index.shape == shuf.edge_index.shape,
              f"{full.edge_index.shape[1]} edges")
        check("b3 keeps the strength multiset identical to full",
              np.allclose(np.sort(full.edge_strength), np.sort(shuf.edge_strength)),
              "same marginals, destroyed pairing")
        check("b3 actually changes the pairing",
              not np.allclose(full.edge_strength, shuf.edge_strength))
        d_mask = ~dist.edge_is_local
        check("b2 flattens distal strength to a constant",
              bool(np.allclose(dist.edge_strength[d_mask], 0.5)) if d_mask.any() else False)
        check("b0 has no distal edges at all",
              not (~by_arm["b0_dna_only"].edge_is_local).any())
        check("targets exist to predict", full.tgt_index.shape[1] > 0,
              f"{full.tgt_index.shape[1]} held-out edges")

        # Determinism: the control seed is what makes seeds comparable.
        again = build_sample(cfg, chrom_data, "FAKE", "chrA", starts[0], arm="b3_shuffled_hic")
        check("controls are deterministic under control_seed",
              np.array_equal(shuf.edge_strength, again.edge_strength))

        print("\nPHASE 3  model")
        model = build_model(cfg).to(device)
        dataset = ShardDataset(cfg, ["chrA"], ["FAKE"], root=tmp)
        batch = to_device(collate([dataset[0], dataset[1]]), device)
        batch["late_fusion"] = False

        model.train()
        out = model(batch, mask_frac=cfg.train.mask_frac, use_structure=True)
        b, n = batch["seq"].shape[:2]
        check("z has shape (batch, nodes, d_model)",
              tuple(out.z.shape) == (b, n, cfg.model.d_model), str(tuple(out.z.shape)))
        check("forward output is finite", bool(torch.isfinite(out.z).all()))
        check("masking actually masked something", bool(out.masked_nodes.any()),
              f"{int(out.masked_nodes.sum())} of {b * n} windows")

        rng = np.random.default_rng(0)
        losses = compute_losses(model, out, batch, cfg, rng)
        check("all three losses are finite",
              all(np.isfinite(v) for v in losses.items().values()),
              json.dumps({k: round(v, 4) for k, v in losses.items().items()}))
        losses.total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        check("gradients reach the parameters", len(grads) > 0, f"{len(grads)} tensors")
        check("no gradient is NaN", all(bool(torch.isfinite(g).all()) for g in grads))

        # The bias MLPs are the contribution; if they get no gradient, the
        # mechanism is inert and every result would be meaningless.
        bias_grads = [p.grad for p in model.edge_bias.parameters() if p.grad is not None]
        bias_norm = sum(float(g.abs().sum()) for g in bias_grads)
        check("the contact-bias MLPs receive gradient", bias_norm > 0,
              f"|grad| = {bias_norm:.3e}")

        print("\n     structure dropout")
        model.eval()
        with torch.no_grad():
            z_with = model(batch, use_structure=True).z
            z_without = model(batch, use_structure=False).z
        check("removing the graph changes the representation",
              not torch.allclose(z_with, z_without, atol=1e-5),
              "so the graph is genuinely being used")
        check("the model still runs with no graph at all",
              bool(torch.isfinite(z_without).all()),
              "which is what makes Hi-C-free inference testable")

        print("\n     softmax identities (same ones the explainer proves)")
        scores = torch.randn(50, 2)
        index = torch.randint(0, 8, (50,))
        a1 = scatter_softmax(scores, index, 8)
        a2 = scatter_softmax(scores + 137.0, index, 8)
        check("scatter softmax is shift-invariant",
              torch.allclose(a1, a2, atol=1e-6),
              f"max dev {float((a1 - a2).abs().max()):.2e}")
        sums = torch.zeros(8).index_add_(0, index, a1[:, 0])
        check("weights sum to 1 within every node",
              bool(torch.allclose(sums[sums > 0], torch.ones_like(sums[sums > 0]), atol=1e-5)))

        print("\nPHASE 3  training loop")
        run = train(cfg, "smoke", device, max_steps=40, out_root=tmp / "results",
                    data_root=tmp)
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        check("history was written", len(history) >= 2, f"{len(history)} rows")
        check("the masked-window loss decreases",
              history[-1]["dna"] < history[0]["dna"],
              f"{history[0]['dna']:.4f} -> {history[-1]['dna']:.4f}")
        # The contact term is binary cross-entropy against SOFT targets whose
        # mean is ~0.5 (strength = oe/(1+oe), and oe averages 1 by construction).
        # Its floor is therefore the target entropy, about 0.693, and it reaches
        # that almost immediately. Asserting "total loss decreases" over a short
        # run would be asserting something the objective cannot deliver, so the
        # check is that the term sits AT its floor rather than diverging above it.
        floor = 0.6931
        final_contact = history[-1]["contact"]
        check("the contact term sits at its entropy floor, not above it",
              final_contact <= floor * 1.05,
              f"{final_contact:.4f} vs floor {floor:.4f}")
        check("no loss term diverged",
              all(row["total"] < 10.0 for row in history),
              f"max total {max(r['total'] for r in history):.4f}")
        print(f"     (total {history[0]['total']:.4f} -> {history[-1]['total']:.4f}; "
              f"only the masked-window term has room to move at this scale)")
        check("run_config.yaml written before training",
              (run / "run_config.yaml").exists())
        check("provenance records the commit", "commit" in json.loads(
            (run / "provenance.json").read_text(encoding="utf-8")))
        check("checkpoint written", (run / "checkpoint.pt").exists())

        print("\n     resume")
        run2 = train(cfg, "smoke", device, max_steps=44, out_root=tmp / "results",
                     data_root=tmp)
        state = torch.load(run2 / "checkpoint.pt", map_location="cpu", weights_only=False)
        check("resumed and advanced past the old step", state["step"] == 44,
              f"step {state['step']}")

        print("\nPHASE 5  metrics")
        pred = np.array([0.9, 0.8, 0.2, 0.1, 0.6])
        true = np.array([0.9, 0.7, 0.3, 0.1, 0.55])
        sep = np.array([2, 12, 40, 45, 5])
        m = contact_metrics(pred, true, sep)
        check("metrics are distance-stratified",
              all(k in m for k in ("short", "medium", "long", "overall")))
        check("perfect predictions give r = 1",
              abs(pearson(true, true) - 1.0) < 1e-9)
        check("AUPRC is 1 when ranking is perfect",
              abs(average_precision(np.array([0.9, 0.8, 0.1]),
                                    np.array([1.0, 1.0, 0.0])) - 1.0) < 1e-9)

        from chromgraph.evaluate import evaluate_run
        report = evaluate_run(run, cfg, device, {"test": ["chrC"]}, ["FAKE"],
                              max_batches=2, data_root=tmp)
        check("evaluation produces the hic_free setting",
              "test/hic_free" in report["settings"])
        check("evaluation produces the with_hic setting",
              "test/with_hic" in report["settings"])
        check("metrics.json written", (run / "metrics.json").exists())

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'-' * 62}")
    if failures:
        print(f"{len(failures)} of {checks} checks FAILED:")
        for name in failures:
            print(f"    - {name}")
        return 1
    print(f"all {checks} checks passed")
    print("\nThe pipeline is internally consistent. It says nothing about whether")
    print("the mechanism works on real chromatin -- that is Phase 5 on real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
