# Self-review: full rewrite of 2026-10-14

This review follows the [Research-Paper-Writing-Skills](https://github.com/Master-cai/Research-Paper-Writing-Skills) output contract:
1. a section outline;
2. paragraph roles;
3. the five-dimension self-review;
4. a claim–evidence map.

It is not part of the paper.

## What changed in this rewrite

- **The paper now describes the model that is actually being confirmed.**
  - The pre-registered recipe rule selected **v1** (`configs/base.yaml`, commit 660d20f). v1 has three objectives, a soft-target BCE contact loss, and structure dropout with `lambda_distill: 0.0`.
  - The previous draft described **v2**: log(O/E) regression plus structure distillation. That was wrong for every confirmatory run.
- **Distillation is no longer a contribution.** It appears only as the screened, unselected v2 recipe, in §2.4 and §3.8.
- **§3.8 now states the protocol in full:**
  - recipe selection;
  - the two read-outs (head and probe);
  - the pre-registered decision rule;
  - disclosure of the seed-0 test read.
- **Sections 4–6 now have a fixed structure** of `[PENDING]` slots, one per pre-registered question. They contain no outcomes.
- **Figures and equations were updated to v1:**
  - Figure 1 is drawn from `configs/base.yaml`, with no distillation arrow.
  - Eq. `l_contact` is now soft-target BCE.
  - `l_total` has three terms.
  - `eq_l_distill.png` is removed.

## 1. Outline (thesis and paragraph roles)

**Thesis:** measured Hi-C contacts, used inside a DNA encoder's attention during self-supervised pretraining and then removed, may yield better sequence-only representations. This is tested against matched controls under a pre-registered rule.

| Section | Paragraph roles |
|---|---|
| Abstract (skill Version 2) | task → challenge → insight (training-time routing signal) → contribution (contact-biased attention, held-out targets, structure dropout) → evaluation design → [PENDING result] |
| §1 Introduction | opening (general → specific setting) → challenge 1 (sequence models weigh by position) → challenge 2 (Hi-C used as output, inference input or external target; copying shortcut; inference dependence) → contribution 1 → remaining challenge → contribution 2 → contribution 3 (evaluation as evidence design) → [PENDING evidence] → contribution list |
| §2 Related work | four topics, each ending with the distinction: seq→structure; DNA LMs; learning from Hi-C (Evo2HiC is the closest); structural attention bias and training-time-only signals |
| §3 Method | overview → data and detrending → graph and held-out targets → local encoder → contact-biased attention → structure dropout → objectives → controls and protocol → implementation; each module is written as motivation, design, advantage |
| §4 Results | setup → Claim A → Claim B → test confirmation → secondary (all PENDING) |
| §5 Discussion | PENDING interpretation; the scope and limitations paragraph is written now |
| §6 Conclusion | PENDING |

## 2. Five-dimension self-review

| Dimension | Status | Open item |
|---|---|---|
| Contribution | needs revision after the results | Contribution 1 (contact-biased attention in SSL pretraining) is narrow and defensible against Chromoformer and Evo2HiC. Contribution 2 is now plain whole-graph dropout, which is weaker as a novelty claim. Contribution 3 (the pre-registered controlled protocol) carries more weight than before. Re-run the novelty search on bioRxiv and arXiv before submission. |
| Writing clarity | pass | One message per paragraph and stable terms (window, distal edge, target edge, structure dropout, head and probe read-out). Read the whole draft aloud once Results exist. |
| Experimental strength | **needs new experiment** (reviewer risk) | Only matched internal controls. There is no external baseline (a DNA LM through the same probe, or Akita/Orca as reference) and no downstream task (TAD boundaries, compartments, insulation). Plan: `docs/study_overview.md` §7, as secondary analyses after the decision. |
| Evaluation completeness | pass for the pre-registered scope | Ablations are covered by B0–B5. There is one cell line only, so K562 transfer would answer "does it generalise". Seed variance is large relative to the arm gaps, and the Discussion must say so. |
| Method soundness | pass | Held-out targets, distance-matched negatives, a fixed control seed, the NaN guard (disclosed) and the seed-0 test read (disclosed). Risk: reviewers may ask why v1 was kept over v2. §3.8 states the rule, which was fixed in advance. |

## 3. Claim–evidence map

| Claim (location) | Evidence | Status |
|---|---|---|
| No prior model lets contacts steer a self-supervised DNA encoder's attention during pretraining (Abstract, §1, §2) | Literature review of 2026-10-08; Chromoformer (supervised) and Evo2HiC (external alignment) are distinguished | supported by the review; needs a systematic search before submission |
| Held-out targets stop the contact objective from being solved by copying (§1, §3.3) | `chromgraph/graph.py` split; `scripts/smoke_test.py` checks that an untrained model cannot beat the mean guess | supported (code and tests) |
| Unobserved valid pairs must count as zero in E(s) (§3.2) | Fix 2b26d56; Figure 2; memory note on the detrend bias (slopes diverge by −0.14 to −0.39 above 500 kb before the fix) | supported |
| Structure dropout makes the Hi-C-free mode in-distribution (§1, §3.6) | Design argument; `model._edges` drops distal edges per sample with p = 0.3 | supported as a design statement; whether it *helps* is what Claim B tests |
| Measured contacts improve the Hi-C-free representation (Abstract, §1 result slot) | Pre-registered Claims A and B, seeds 1–3, val; `scripts/decide.py` | **needs evidence**: decision due after seed 3 (around 2026-10-15) |
| Conditioning inside the encoder beats late fusion or alignment only (§2.3) | B4 and B5 secondary arms | **needs evidence**: queued; secondary only, cannot become the headline |
| Model has 7.3 M parameters and 14.3 GiB peak memory (§3.9) | Run logs and provenance from the timing runs | supported |
