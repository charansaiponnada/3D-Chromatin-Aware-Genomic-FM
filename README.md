# ChromGraphFM

An end-to-end 3D chromatin-conditioned genomic representation model. Hi-C
contact graphs bias attention *inside* the sequence encoder during
self-supervised pretraining, rather than being fused in after the sequence
representation has already been learned.


## The question

Does conditioning DNA sequence representation learning on measured 3D chromatin
contacts produce more transferable embeddings than sequence-only pretraining,
late fusion, or Hi-C-guided embedding alignment?

The result is only meaningful against its controls, so the controls are built
into the code, not bolted on at the end: shuffled Hi-C (B3), distance-only edges
(B2), a random graph (B1), and late fusion (B4) all run through the same
training script at matched parameters, data and steps.

## Architecture

`docs/diagrams/chromgraphfm_architecture.drawio` — open at
[app.diagrams.net](https://app.diagrams.net) (File → Open From → Device).

One graph node = one 5 kb Hi-C bin = one 5 kb DNA window. 128 nodes gives
640 kb of context per sample. Attention runs over sparse graph edges only
(top-16 distal + local neighbours), which is what makes that context fit on
one 48 GB GPU.

Three components are deliberate and load-bearing:

- **Contact-biased attention.** `a_ij = softmax(q·k/√d + b_HiC(c) + b_dist(d) + b_scale(r))`
  over graph edges. The contact map decides which distal windows exchange
  information and how strongly.
- **Structure dropout.** The whole graph is dropped with p=0.3 during
  pretraining. Without it, removing Hi-C at inference is out of distribution
  and the Hi-C-free transfer claim cannot be tested at all.
- **Held-out target edges.** The edges the contact head predicts are excluded
  from the edges attention conditions on. Otherwise the model copies its own
  input and the contact metric measures nothing.

## Two machines, one repo

Phases 0–2 run on a laptop (CPU). Phases 3–6 need the college L40S cluster.
Only git moves between them, and git cannot carry the data: hg38 is 3.1 GB and
a single `.mcool` is 5–30 GB, against a 100 MB GitHub file limit.

So **git carries code plus `data/manifest.json`; the cluster rebuilds the data
from it.** Nothing under `data/raw` or `data/processed` is transferable, and no
path is hardcoded — everything resolves from `CHROMGRAPH_DATA` and
`CHROMGRAPH_RESULTS`.

```bash
# laptop
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e .
python tests/test_config.py

# cluster
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e . -r requirements-gpu.txt
export CHROMGRAPH_DATA=/scratch/$USER/chromgraph/data
python scripts/build_data.py --full                  # Phase 1
```

## Phases

| # | Deliverable | GPU |
|---|---|---|
| 0 | Scaffold, config, split validation | no |
| 1 | hg38 windows, streamed Hi-C, ICE-normalised 5 kb bins, chromosome splits, QC | no |
| 2 | Sparse contact graph builder, held-out edge masking, B1–B3 control corruptions | no |
| 3 | Bi-Mamba local encoder, B0 baseline, contact head, eval harness | 1× |
| 4 | Contact-biased attention, edge bias, structure dropout, the three losses | 1–2× |
| 5 | B0–B5 matched-compute sweep; held-out chromosome, cell line, Hi-C-free probe | 2× |
| 6 | Loop AUPRC, E–P linking, CTCF enrichment at high-attention anchors | 1× |

Phase 0 is done.

## Splits

Chromosome-level, never random — Hi-C is autocorrelated over megabases and a
random window split leaks structure across the split boundary.

- train: autosomes except the below
- val: chr10, chr11
- test: chr3, chr13, chr17
- plus a held-out cell line, and a Hi-C-free inference setting

## Honesty rules

1. No number appears in a slide, report or figure unless it traces to a file
   under `results/`. Placeholders stay `??`.
2. Every run writes `run_config.yaml` next to its `metrics.json`, before the
   first step.
3. Controls use a fixed `control_seed` separate from the training seed, so
   seed variance and shuffle variance are never confounded.
4. No claim of superiority over a published model unless the task, data,
   split, metric and resolution all match.
