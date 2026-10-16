# Pre-registered decision rule: per-window probes (frozen checkpoints)

**Written:** 2026-10-16, before any label in this document has been computed and
before any checkpoint has been probed on these tasks.
**Fixed:** at the commit that adds this file. Nothing below changes after a
number it would affect has been seen.

This document does not amend `docs/preregistration.md`. That rule is closed and
returned a controlled negative (`results/decision_claims_val.json`). Nothing here
reinterprets it. `docs/preregistration_v3.md` (SIS) is a separate, unapproved draft.

## 1. Motivation (stated before any result)

In the Hi-C-free setting the model keeps only local edges (radius 3) over 4
blocks, so a window can draw information from at most about ±12 windows
(about ±60 kb). The pre-registered contact task scores pairs at ≥150 kb, and
the two windows of such a pair never exchange information. The contact score can
therefore only reflect per-window features, and it measures them indirectly.

This study asks the per-window question directly: **does Hi-C-conditioned
pretraining change what a single window's Hi-C-free representation encodes about
3D genome organisation?**

No model, loss, config or checkpoint changes. Only evaluation runs.

## 2. Labels (GM12878, `4DNFIXP4QG5B.mcool`, ICE-balanced)

All labels are per 5 kb bin, computed by `scripts/window_labels.py` from the mcool
already on disk. Only bins that are usable (`bins.npz["usable"]`) and have a
finite label are scored.

| Task | Label | Type |
|---|---|---|
| T1 insulation | Diamond insulation score at 5 kb, square of 20 bins (100 kb). log2 of the score over the chromosome's mean score. | regression |
| T2 compartment | First eigenvector of the cis O/E Pearson-correlation matrix at 100 kb, oriented so that it correlates positively with GC content. Each 5 kb bin takes the value of its 100 kb bin. | regression |
| T3 boundary | A bin is a boundary if its insulation (T1) is a local minimum within ±5 bins and its boundary strength (mean of the maxima within 5 bins on each side, minus the value) is ≥ 0.1. A bin is labelled positive if it lies within ±1 bin of a boundary. | binary |

## 3. Representation and probe

- **Representation:** `z` from the Hi-C-free forward pass (`use_structure=False`,
  no masking), the same setting as the pre-registered study. Samples of 128
  windows with stride 128, so each window appears once.
- **Probe:** a linear model on standardized `z` (mean and SD from train
  chromosomes).
  - T1 and T2: ridge regression, α = 1.0.
  - T3: L2-logistic regression, C = 1.0, class-balanced weights.
- **Training:** fitted on the 17 training chromosomes and scored on validation
  (chr10, chr11). The test chromosomes stay sealed.
- **Metrics:**
  - T1 and T2: Pearson r.
  - T3: AUPRC.

## 4. Arms and seeds

- **Confirmatory:** Full, B3, B0 and B1, seeds 1–3, the existing checkpoints.
- **Secondary, reported whatever the outcome:** seed 0 of every arm, including
  B2, B4 and B5.
- **Reference baselines, reported whatever the outcome:**
  - per-window GC fraction and N fraction (2 features);
  - the randomly initialised model (Full architecture, seed 1, no training).

## 5. Decision rule

The statistic is unchanged from `docs/preregistration.md`. It is a paired
bootstrap over validation **samples** (128-window blocks), with 10,000
resamples, bootstrap seed 20261008, on the metric averaged over seeds 1–3.

Three tasks are tested, so each CI is a Bonferroni-corrected **98.33%** CI.
"X beats Y on task T" requires:
- the CI lower bound > 0, and
- X > Y in at least 2 of the 3 seeds.

**Claim W (Hi-C pretraining improves per-window 3D features).** Claim W holds
on task T if Full beats B0 **and** Full beats B1 on T. Claim W passes if it
holds on at least one task, and the tasks where it holds are named.

**Claim W′ (the gain needs real strengths).** On each task where Claim W
holds, Full is tested against B3. This is reported, not required.

## 6. Pre-specified wording

| Outcome | Statement |
|---|---|
| W holds on ≥ 1 task | "Hi-C-conditioned pretraining improved the Hi-C-free per-window representation of [tasks], over both a DNA-only and a random-graph control, although it did not improve long-range contact prediction." |
| W holds on no task | "Hi-C-conditioned pretraining did not improve per-window 3D features either; together with the contact result, the Hi-C-free representation is not measurably changed by Hi-C conditioning at this scale." |

The claim may not be strengthened beyond this wording. No task, label
definition, probe or hyper-parameter is changed after a result is seen. If a
label computation fails, it is fixed and the fix is disclosed. No other task is
substituted.

## 7. Test split

Opened once, after Section 6 is decided on validation, for the seeds 1–3 arms
in Claim W, with the same probes refitted the same way.
