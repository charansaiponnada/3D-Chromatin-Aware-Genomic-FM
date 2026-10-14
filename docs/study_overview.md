# ChromGraphFM: study overview

A plain-language map of the whole study: the problem, the data, pretraining, what is new, how it is tested, what is claimed, and what comes next. The binding rules are in [`preregistration.md`](preregistration.md); the manuscript is [`../paper/master_copy.docx`](../paper/master_copy.docx). If this file and the preregistration disagree, the preregistration wins.

---

## 1. Research problem

DNA sequence models, such as DNABERT, Nucleotide Transformer, HyenaDNA, Enformer and Evo, read the genome as a line. When they relate one region to another, they use distance along the chromosome. Much gene regulation does not work that way: an enhancer hundreds of kilobases away can act on a gene because chromatin folds and brings the two into physical contact.

Hi-C measures that folding: for every pair of genomic bins, how often they touch. Existing work uses Hi-C in three ways:
- as an **output** to predict from sequence (Akita, Orca, C.Origami);
- as an **input that must be present at inference**;
- as a separate **map to model** in its own right.

**The question:** if measured 3D contacts shape how a DNA encoder learns during pretraining, and are then removed, are the resulting sequence-only representations better?

## 2. Data

| Item | Source |
|---|---|
| Genome | hg38 (UCSC) |
| 3D contacts | GM12878 in situ Hi-C, 4DN `4DNFIXP4QG5B` (72 experiments, the deepest public map), used at 5 kb |
| Cell lines used | GM12878 only (K562 and IMR90 are listed in `data/manifest.json` but not downloaded) |
| Contact processing | balanced contacts divided by the expected contact at each genomic separation (distance detrending); unobserved but valid bin pairs count as zero (fix 2b26d56) |

**Chromosome splits.** Hi-C is correlated over megabases, so the data is split by whole chromosome:

| Split | Chromosomes | Use |
|---|---|---|
| Train | the other 17 autosomes | pretraining (6,284 samples) |
| Validation | chr10, chr11 | every decision |
| Test | chr3, chr13, chr17 | **sealed**; scored once, after the claims are decided |

## 3. Model and pretraining

- **Input:** one sample is 128 consecutive 5 kb windows, 640 kb in total.
- **Sequence stream:** each window is encoded on its own (convolutional tower plus a bidirectional state-space encoder) into one 256-dimensional vector.
- **Structure stream (training only):** the windows become nodes of a graph. Each window links to its neighbours within ±3 windows and to its 16 strongest detrended distal contacts. 15% of distal edges are withheld as prediction targets and never shown to the encoder.
- **Contact-biased attention:** 4 attention blocks. Attention runs only along graph edges, and each edge's contact strength and distance add a learned bias.
- **Pretraining objectives** (self-supervised, no task labels):
  1. masked-window reconstruction;
  2. distance-matched contrastive alignment on held-out contacts;
  3. contact-strength prediction on held-out edges;
  - plus structure dropout: the whole distal graph is removed for 30% of samples.
  - Structure distillation (Hi-C-free pass matches the Hi-C pass) was the v2 recipe. Seed 0 chose v1, which does not use it (`lambda_distill: 0.0`).
- **Budget:** 7.29M parameters, 20,000 steps per run, about 7.4 hours per run on one NVIDIA L40S.

## 4. Novelty

1. **Contact-biased sparse attention.** Measured Hi-C decides *which* distant windows exchange information and *how strongly*. Held-out target edges mean the model can't just copy contact values from its input.
2. **Hi-C as training-time-only information.** Structure dropout (30% of samples train with the distal graph removed) gives a model that needs **only DNA at inference**. Structure distillation was tested in v2 but is not in the chosen v1 recipe.
3. **A controlled, preregistered evaluation.** Six matched control arms isolate what measured contacts contribute, separately from graph regularisation, genomic distance, strength marginals and post-hoc fusion. The decision rule was fixed before any confirmatory result was seen.

## 5. How it is tested

**Setting.** Hi-C is removed at inference for every arm (`hic_free`), so the only difference between arms is what shaped their weights during pretraining.

**Metric.** Pearson r between predicted and measured contact strength on held-out edges at ≥150 kb (long range), on validation. It is read out two ways:
- **Head:** the model's own contact head.
- **Frozen probe:** a fresh head fitted on frozen representations, identical for every arm. This is the only fair read-out for B0 and B5, which don't train a head.

**Arms.** Every arm has the same architecture, data, steps and held-out targets; only the structural signal changes.

| Arm | Signal in pretraining | Role |
|---|---|---|
| **Full** | real contacts and strengths | the model |
| **B3** | real contacts, strengths shuffled | confirmatory control |
| **B0** | DNA only, no graph | confirmatory control |
| **B1** | random graph | confirmatory control |
| B2 | real contacts, constant strength | secondary |
| B4 | Hi-C fused after the encoder | secondary |
| B5 | Hi-C only as a contrastive target | secondary |

**Statistic.**
- 10,000 paired bootstrap resamples of validation windows give a 95% CI for the difference averaged over seeds 1–3.
- **X beats Y** only if the CI lower bound is above 0 **and** X > Y in at least 2 of the 3 seeds.
- Implemented in `scripts/decide.py`.

**Seeds.** Seed 0 was exploratory and chose the recipe (v1, commit 660d20f). Seeds 1, 2 and 3 are confirmatory.

## 6. Claims

- **Claim A, measured strengths (stop gate):** Full beats B3 on the head metric.
- **Claim B, measured topology:** Full and B3 each beat B0 and B1 on the probe metric. All four comparisons must pass.

| A | B | What the paper says |
|---|---|---|
| pass | pass | Measured 3D contacts improve a DNA-only representation, through both which regions touch and how strongly. |
| fail | pass | Measured contact topology improves a DNA-only representation; contact strength adds no detectable benefit. |
| pass | fail | No headline improvement over sequence-only training; Full > B3 reported as secondary. |
| fail | fail | Controlled negative result: Hi-C conditioning did not produce a better Hi-C-free representation. |

**Results so far** (validation, long-range r, from `results/<run>/metrics.json` and `probe_metrics.json`; not a decision, which needs all of seeds 1–3):

| Seed | Full head | B3 head | Full probe | B3 probe | B0 probe | B1 probe |
|---|---|---|---|---|---|---|
| 0 (exploratory) | 0.2066 | 0.1848 | 0.161 | 0.193 | 0.188 | — |
| 1 | 0.1876 | 0.1882 | 0.197 | 0.140 | 0.164 | 0.145 |
| 2 | 0.2061 | 0.1965 | 0.171 | 0.197 | pending | pending |

## 7. Benchmarks and downstream tasks (planned, secondary)

The preregistered claims compare the model only with its own matched controls. They can show that 3D information helps this model; they can't show it beats published models or captures biology beyond the training target. The analyses below would add that.

They are **not part of the preregistration**. They run only after Claims A and B are decided, are labelled secondary or exploratory in the paper, and evaluate existing checkpoints with frozen representations, with no retraining and no architecture changes. Each one needs approval before it starts.

### Downstream tasks (GM12878 labels already listed in `data/manifest.json`)

| # | Task | Label source | Read-out |
|---|---|---|---|
| 1 | **TAD boundary detection:** is this 5 kb window a domain boundary? | 4DN boundary calls `4DNFIVK5JOFU` | classification probe; AUROC / AUPRC |
| 2 | **A/B compartment prediction:** active or inactive nuclear compartment | 4DN compartment track `4DNFILYQ1PAY` | regression/classification probe; Pearson r, accuracy |
| 3 | **Insulation score prediction:** how strongly a window blocks contacts across it | 4DN insulation track `4DNFIBMOGOZC` | regression probe; Pearson r |

Each task is run for every arm (Full, B3, B0, B1) on the same validation and test chromosomes, with the same frozen-probe protocol.

### External baselines

| Baseline | What it tests |
|---|---|
| A public pretrained DNA language model (e.g. Nucleotide Transformer or HyenaDNA), embeddings pooled to 5 kb windows, through the same frozen probe | whether ChromGraphFM representations beat a generic sequence model on the same tasks |
| Akita or Orca, run on our validation and test chromosomes | reference point for contact prediction from sequence; a specialist model, so it is reported as context, not as a target to beat |

### Optional: another cell line

K562 (`4DNFI18UHVRO`), evaluation only on GM12878-trained checkpoints. This tests transfer to an unseen cell type. It needs the download (disk is nearly full) and the shallower map (6 experiments) must be disclosed.

## 8. Timeline

All dates are 2026 and estimated; any server restart pushes them back.

| When | Milestone |
|---|---|
| Sep 19 – Oct 8 | data build, detrend fix, model and loss fixes, seed-0 exploratory runs |
| Oct 8 | preregistration written and committed |
| Oct 9 – 13 | recipe chosen (v1); confirmatory seeds 1 and 2 started |
| **Oct 14, ~20:00** | Full and B3 seed 3 done → **Claim A** inputs complete |
| **Oct 15, ~03:30** | B0 and B1 seed 3 done → `decide.py` → **Claims A and B decided** |
| Oct 15, ~18:00 | secondary arms (B1 seed 0, B4, B5, B2) done |
| Oct 15 – 16 | test split opened once; test numbers recorded |
| Oct 16 – 20 | downstream tasks and external baseline (if approved) |
| Oct 20 – 31 | Results, Discussion and Conclusion written from measured numbers; figures; internal review |
| November | final novelty check, then journal submission (Bioinformatics, NAR Genomics & Bioinformatics, PLOS Computational Biology or BMC Bioinformatics, depending on the outcome) |
