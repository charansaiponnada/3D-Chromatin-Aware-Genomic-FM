# Pre-registered decision rule

**Written:** 2026-10-08, before any v2, B0, B1, B2, B4 or B5 result has been seen.
**Status:** fixed. Any change after this date is recorded below under *Amendments*, with the date and the reason, and must not be made after the number it would affect has been seen.

This rule decides which claim the paper makes. Every outcome leads to a defined claim, so no result has to be explained away and none can be steered toward.

---

## 1. What has already been seen

Disclosed in full, because it limits what can still count as confirmatory.

| Run | Split | Seen | Value (long-range r, Hi-C-free) |
|---|---|---|---|
| full, v1, seed 0 | val | yes | 0.2066 (built-in head); 0.161 (frozen probe) |
| b3_shuffled_hic, v1, seed 0 | val | yes | 0.1848 (built-in head) |
| full and b3, v1, seed 0 | **test** | **yes**: read by an evaluation bug, fixed in c4a697d | 0.160 vs 0.170 |
| v1 seed-0 gap, full − b3 | val | yes | +0.022, paired-window 95% CI [−0.003, +0.046] |
| full_v2, b3_v2 (partial, step ≤ 1,650) | training loss only | yes | no evaluation seen |
| everything else | — | no | — |

Consequences:
- **Seed 0 is exploratory** for every arm. It may be used to choose the recipe, never as confirmatory evidence.
- **Seed-0 test numbers for full and b3 are spent.** They are reported in the paper as an exploratory observation, never as the confirmation.
- **Confirmation uses fresh seeds 1, 2 and 3**, which have never been evaluated on any split.

## 2. Recipe choice (exploratory, seed 0)

Unchanged from the v2 rule fixed earlier on 2026-10-08. **v2 is chosen** only if both of these hold on val:
- full_v2 > full (0.2066)
- the paired-window bootstrap gap full_v2 − b3_v2 has a 95% CI above 0, and is larger than the v1 gap (+0.022)

Otherwise, **v1 is chosen**. Either way, the chosen recipe is then frozen: every arm in Section 3 trains under it, with the same config, steps and hyperparameters. No third recipe is screened before confirmation.

## 3. Confirmatory runs

- **Arms:** Full, B3 (shuffled strengths), B0 (DNA only), B1 (random graph). B2, B4 and B5 are secondary (Section 6).
- **Seeds:** 1, 2, 3 for each arm, under the frozen recipe, at 20,000 steps.
- **Setting:** Hi-C removed at inference (`hic_free`), on the validation chromosomes (chr10, chr11).
- **Metric:** Pearson r between predicted and measured strength on held-out contact edges at ≥ 150 kb (long range).
- **Two read-outs, fixed per claim:**
  - *Head metric:* each model's own trained contact head. Valid only for arms that train one (Full, B3, B1, B2).
  - *Probe metric:* a fresh contact head fitted on frozen Hi-C-free representations (`scripts/probe.py`; same head architecture, training pairs, epochs and probe seed 1234 for every arm). Valid for every arm, and the only valid read-out for B0 and B5.

### Statistic

For a comparison X − Y:
1. Draw 10,000 bootstrap resamples of the validation windows (paired: the same windows for X and Y).
2. In each resample, compute the metric for each seed of X and of Y, average over seeds 1–3, and take the difference.
3. The 95% CI is the 2.5th–97.5th percentile of those differences.

**X beats Y** when **both** of these hold:
- the 95% CI lower bound is > 0
- X − Y > 0 in at least 2 of the 3 individual seeds

The bootstrap is implemented and committed **before** any confirmatory run is evaluated.

## 4. The two claims

**Claim A, measured strengths (the stop gate):** Full beats B3 on the head metric.

**Claim B, measured topology:** all four of these hold, on the probe metric:
- Full beats B0, and Full beats B1
- B3 beats B0, and B3 beats B1

B3 is in Claim B because it keeps the real contact edges and destroys only the strength assignment. If both real-edge arms beat both no-real-edge arms, the gain comes from measured contacts. All four comparisons must pass (an intersection–union test), so no multiple-comparison correction is needed.

## 5. What the paper claims

| Claim A | Claim B | Headline |
|---|---|---|
| pass | pass | **Measured 3D contacts improve a DNA-only representation, through both which regions touch and how strongly.** |
| fail | pass | **Measured contact topology improves a DNA-only representation; finer differences in contact strength among the strongest contacts add no detectable benefit.** The Full-vs-B3 result is reported as a null. |
| pass | fail | No headline claim of improvement over sequence-only training. The Full > B3 result is reported as a secondary finding with that caveat. |
| fail | fail | **No positive claim.** Reported as a controlled negative result: under matched budgets, Hi-C conditioning during pretraining did not produce a better Hi-C-free representation. |

Wording is fixed in substance. Phrasing may be edited, but the claim may not be strengthened.

## 6. Secondary comparisons

These are reported whatever the outcome, are never used to choose the headline, and are labelled secondary.
- Full vs B2 (distance only)
- Full vs B4 (late fusion) and Full vs B5 (contrastive only): does conditioning inside the encoder beat using Hi-C outside it?
- Every arm with Hi-C *present* at inference, as an upper bound
- Distance strata: short (< 50 kb) and medium (50–150 kb)

## 7. Test split

- Opened **once**, after Section 5 is decided on val, for seeds 1–3 of the arms in the chosen claim, using the same statistic.
- Every test number is reported as measured. If test disagrees with val, the paper says so. Nothing is re-selected, retrained or re-tuned after test is opened.
- The seed-0 test numbers already seen (Section 1) are reported as exploratory.

## 8. Stopping

- No further recipes, losses or hyperparameter changes are tried to make Claim A pass once confirmatory runs have started.
- If a confirmatory run crashes, it resumes from its checkpoint. It is never replaced by a rerun with a different seed.

## Amendments

*(none)*
