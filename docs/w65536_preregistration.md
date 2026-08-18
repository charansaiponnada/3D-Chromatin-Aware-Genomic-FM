# Pre-registration — the 65,536 bp decision run

Written **2026-08-18, before the run was launched and before any 65 kb number
existed.** Committed before launch. If any threshold in this file is edited
after a result lands, the edit and its reason must be recorded in
`project-docs/project.tex` in the same commit, and the original values kept
visible. This file exists to stop the goalposts moving; that is its only job.

## What this run is

One seed (0) per arm, warm-started from the finished 32,768 bp checkpoints,
trained 1,000 further steps at a 65,536 bp window.

- structural: `results/novel_model/w32768/structural_seed0/checkpoint.pt`
- baseline:   `results/baselines/w32768/baseline_v2_seed0/checkpoint.pt`

**This run cannot produce a publishable claim.** n = 1 per arm. There is no
seed variance, therefore no test, therefore no p-value, and none will be
reported. Its only purpose is to decide whether to spend the ~150 GPU-hours a
publishable version would cost.

## Why the window changed

`p5_window_scan.json`, no model involved: keep(φ) = 0.0490 at 32,768 bp and
0.1099 at 65,536 bp. The per-position Δ bias can only act on the within-window
share, so widening multiplies its available signal by 2.24×.

**This is an upper bound.** At 65,536 bp, 89.0% of φ's variance is still
between windows. Widening is a partial mitigation and this run tests a
partial mitigation. It is not a fix and must not be written up as one.

## PRIMARY criterion — fixed now

**Δ = (baseline val bits/nt) − (structural val bits/nt) at step 1,000, same
data, same steps, same tokens.**

**PASS iff Δ ≥ +0.0050 bits** (structural better by at least the pre-registered
2σ_real floor from §4.1.3, σ_real = 0.0025 measured over the three Phase 3 v2
seeds).

At 32,768 bp this quantity was **−0.0020** (structural worse). The sign must
change and the magnitude must clear 0.0050.

σ_real was measured at 32,768 bp. It is carried over unchanged because
re-deriving it needs 3 seeds, which is the compute this run exists to decide
about. If the full run happens, σ must be re-measured at 65,536 and the floor
restated.

## SECONDARY criteria — mechanism, fixed now

Both are computed on this run's checkpoint by the existing scripts, unmodified.

- **D1** (`phase5_vars_diagnostic.py`): median pooled
  Var_t(W_Δs·s)/Var_t(dt_proj·δ'). Was 0.0002–0.0004 at 32,768 (inert; the
  §4.1.3 threshold is 0.05). **LIVE iff median pooled D1 ≥ 0.05.**
- **Transfer probe** (`phase5_structure_probe.py`): structural must clear the
  GC/dinucleotide composition floor on `insulation_100kb`. The floor must be
  **recomputed at 65,536 bp** — the 0.1905 figure is a 32,768 bp number and
  does not transfer. At 32,768 the structural arm scored 0.121 against a floor
  of 0.1905, i.e. it lost to counting GC content.

## DECISION RULE — fixed now

| primary | D1 | decision |
|---|---|---|
| PASS (Δ ≥ +0.0050) | any | **Go.** Full run: 3 seeds × 2 arms, multi-chromosome, transfer benchmarks. ~150 GPU-h. |
| FAIL | LIVE (≥ 0.05) | **Narrow.** The mechanism engaged but did not pay. Run the per-window conditioning arm, or one more width. Do not launch the full 3-seed sweep. |
| FAIL | INERT (< 0.05) | **Stop and write.** The diagnosed null, with the keep(φ) scan as the measured cause. |

A result between −0.0050 and +0.0050 is a FAIL on the primary criterion, not an
"encouraging trend". That phrase is banned from the write-up of this run.

## Known cost, accepted in advance

Warm starting overwrites the paired initialisation added on 2026-08-17. The
32,768 checkpoints predate that fix, so **this run inherits reviewer weakness
#3 (unpaired arms)**. Accepted deliberately: at n = 1 there is no pairing to
exploit anyway, and convergence — weakness #1, val loss still descending at
step 2,000 — is the larger threat to interpreting a null. A full run launched
under the "Go" branch must be trained fresh with paired inits.

## Confounds this run does NOT control

- n = 1. No seed variance, no statistics.
- Splits are within chr9; φ is autocorrelated over megabases and leaks toward
  the structural arm (weakness #5). Widening the window makes this worse, not
  better, because each window now spans more of the correlated structure.
- Data order is uncontrolled and confounded with arm (weakness #4).
- 32 kb and 65 kb are different datasets and must never be pooled.
