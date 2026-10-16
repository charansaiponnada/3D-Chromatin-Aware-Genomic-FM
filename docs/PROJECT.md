# ChromGraphFM: project documentation

## What it is
ChromGraphFM is a DNA encoder (7.29 M parameters). Hi-C contacts steer its attention during pretraining. The question it tests: does the model stay better once Hi-C is removed?

## Repository layout
| Path | Contents |
|---|---|
| `chromgraph/` | Code: `model.py` (encoder and contact-biased attention), `graph.py` (contact graph, controls), `data.py` (Hi-C and DNA processing), `train.py`, `evaluate.py`, `probe.py` |
| `configs/base.yaml` | The v1 recipe used for every confirmatory run. Do not edit. |
| `scripts/` | `build_data.py`, `pretrain.py`, `evaluate.py`, `probe.py`, `decide.py`, `eval_neutral_local.py`, `window_labels.py` |
| `results/<arm>_seed<N>/` | Checkpoints and metrics (not in git) |
| `paper/` | `content.js` (text), `build_master.js` (builds `master_copy.docx`), `make_figures.py` |
| `docs/` | Preregistrations, study overview, plain-language explainer |

## Data
- **DNA:** hg38.
- **Hi-C:** GM12878 at 5 kb (4DN `4DNFIXP4QG5B`), distance-detrended.
- **Split by chromosome:**
  - train: 17 autosomes;
  - validation: chr10 and chr11;
  - test: chr3, chr13 and chr17 (sealed).

## Model
- **Sample:** 128 windows of 5 kb each, 640 kb in total.
- **Per-window encoder:** a convolutional tower plus 6 bidirectional SSM layers, giving one vector h_i per window.
- **Graph:** local edges to ±3 windows, plus each window's top-16 distal Hi-C partners. 15% of the distal edges are held out as targets.
- **Attention:** 4 blocks of attention along graph edges only, with a learned bias from Hi-C strength and distance.
- **Structure dropout:** for 30% of samples, all distal edges are removed. The Hi-C-free mode uses local edges only.
- **Losses:** masked-window reconstruction (weight 1), distance-matched InfoNCE (0.5), contact BCE (1).

## Arms
| Arm | 3D signal during pretraining |
|---|---|
| Full | Real contacts |
| B0 | None (DNA only) |
| B1 | Random graph |
| B2 | Real topology, constant strength |
| B3 | Real topology, shuffled strengths |
| B4 | Hi-C fused after the encoder |
| B5 | Hi-C as the contrastive target only |

## How to run
```bash
python scripts/build_data.py                             # build data from data/manifest.json
python scripts/pretrain.py --arm full --seed 1           # ~7.4 h on one L40S
python scripts/evaluate.py --run full_seed1              # head metrics
python scripts/probe.py --run full_seed1                 # frozen probe
python scripts/decide.py claims --recipe v1              # pre-registered decision
python scripts/eval_neutral_local.py --run full_seed1    # sequence-only re-check
cd paper && node build_master.js                         # rebuild the paper
```
To run a sweep, start it with `~/chromgraph_sweep/resume_sweep.sh`, which also resumes it after a server restart.

## Results (validation, long-range r ≥150 kb)
- **With Hi-C:** Full 0.644, B3 0.597, B1 0.175 (mean of seeds 1–3). The model uses real 3D structure strongly.
- **Without Hi-C:** all arms score 0.16–0.18 on the probe. No preregistered claim passes (`results/decision_claims_val.json`).
- **Why:** without distal edges, a window sees only about ±60 kb, so the two windows of a ≥150 kb pair never exchange information.
- **Neutral local strengths:** re-scoring with Hi-C removed from the local edges too changes the scores by ≤0.005 so far.
- **All numbers:** Table 4 of the paper and `results/*/metrics.json`.

## Rules
- Report every number from a file on this machine.
- Do not change the model, the losses or `configs/base.yaml` without approval.
- Open the test split once only, after the decision.
- Never commit `data/`, `results/` or logs.
- Write no attribution lines in commits.
