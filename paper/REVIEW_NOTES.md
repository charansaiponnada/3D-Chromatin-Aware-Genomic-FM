# Self-review: Sections 1–3 (draft of 2026-10-08)

This review follows the Research-Paper-Writing-Skills output contract: a reverse outline, then the five-dimension checklist, then a claim–evidence map.

## Reverse outline

**Thesis:** measured Hi-C contacts, used *inside* a DNA encoder during self-supervised pretraining and then removed, yield better sequence-only representations.

**Introduction**
1. The task: DNA representations; much regulation is distal and carried by 3D folding.
2. The challenge: sequence models weigh context by linear position.
3. Prior Hi-C work: Hi-C is used as a target, as an input required at inference, or as an external alignment signal. The root difficulty is twofold: a copying shortcut, and dependence on Hi-C at inference.
4. Contribution 1: contact-biased sparse attention, with held-out targets.
5. Contribution 2: structure dropout plus structure distillation.
6. Evaluation design: matched controls; Hi-C-free; chromosome splits. **[PENDING RESULTS]**
7. Contributions list.

**Related work** has four topics, each ending with how this work differs: sequence-to-structure models; DNA language models; learning from Hi-C (including Evo2HiC, the closest concurrent work); and structural attention bias plus training-time-only signals (including Chromoformer, the closest precedent for the bias).

**Methodology** follows the order of Figure 1, from data to graph, encoder, attention, Hi-C-free training, objectives, controls and implementation. Each module states its motivation, its design and its advantage.

## Five-dimension checklist

| Dimension | Status | Open item |
|---|---|---|
| Contribution | The novelty is stated narrowly: contacts bias attention *during self-supervised pretraining*, and dropout plus distillation make the result Hi-C-free. Chromoformer and Evo2HiC are cited and distinguished. | The novelty search was a web search, not a systematic one. Re-run it on bioRxiv and arXiv before submission. |
| Writing clarity | One message per paragraph; the first sentence states it; terms are stable (window, distal edge, target edge, structure dropout, structure distillation). | Read the whole draft aloud once Results exist. |
| Experimental strength | **Not yet assessable.** | Needs 3–5 seeds, the B1 result, a frozen-embedding probe for B0 and B5, and ideally a downstream task. |
| Evaluation completeness | The protocol is defined: Hi-C-free headline metric, distance strata, paired bootstrap, val/test discipline. | The frozen probe described in §3.8 is **not implemented yet**. It must exist before the Results section is written. |
| Method soundness | The held-out target split, distance-matched negatives, detached distillation teacher and fixed control seed are all justified in the text. | B2 and B3 keep the measured topology. The text says so, and the arms table frames what each pair isolates. |

## Claim–evidence map

| Claim (location) | Evidence | Status |
|---|---|---|
| No prior model lets contacts steer a DNA encoder's attention during self-supervised pretraining (Abstract, §1) | Verified literature search (bibliography.json) | supported by search; re-check before submission |
| Raw contact frequency is dominated by distance (§3.2) | Fig. 2a, from the built data | supported |
| Detrended strength is centred on 0.5 at every separation (§3.2) | Fig. 2b; qc.json slopes | supported |
| Held-out targets make the contact objective non-trivial (§1, §3.3) | Code invariant (`assert_disjoint`) plus a smoke test | supported (design property) |
| Distillation adds little computation (§3.6) | Measured: peak VRAM unchanged at 14.3 GiB | supported |
| Regressing log O/E avoids gradient shrinkage while keeping the same optimum (§3.7) | Analytic; contact loss moves early in v2 training | supported (analytic) |
| The model learns structure / outperforms controls | — | **needs evidence: do not write until the v2 and seed results are in** |
