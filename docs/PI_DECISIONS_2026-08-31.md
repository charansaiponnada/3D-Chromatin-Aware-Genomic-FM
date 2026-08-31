# PI decision memo — six open decisions blocking Phase D

Written 2026-08-31. Every number below came from a command run in this repo and
names the file it is stored in. Projections are labelled as projections and show
their arithmetic.

**Nothing has trained since 2026-08-16.** All Phase A, B, C work is done or
measured. The GPU has been touched only for inference diagnostics. What blocks
the run is not code and not compute — it is these six decisions.

---

## Summary table

| # | Decision | Recommendation | Cost of getting it wrong |
|---|---|---|---|
| D-a | The permeability term `p` | `use_permeability=False` | A null that is partly a handicap, not a finding |
| D-b | Which structural arm(s) | baseline + T5c-dual | Spending 265 GPU-h on the arm that reaches 18% of the signal |
| D-c | Seeds and pairing | 5 paired per arm | p < 0.05 unreachable by design, as in Phase 4 |
| D-d | **Window: 65,536 or 131,072** | 65,536, revisit after | Either 2.4× the cost, or the reachable-variance ceiling again |
| B3 | Second cell line | K562, CNV documented | A cross-cell-line claim on a failed gate, or none at all |
| B2′ | Re-run the 1 kb gate against a matched reference? | Do not rebuild at 1 kb | Either a wasted refetch, or discarding a real 1.95× |

---

## D-a. The permeability term `p`. **Blocking.**

**The finding (2026-08-31, change log entry 4).** `p` was absent from every τ
this project has ever reported. Recomputed from the logged checkpoints, the
structural arm's true τ median is **~51 tokens, not ~487** — about 8.5× shorter
than the baseline's 434.7, because the baseline carries no `p` term at all.
`b_g = -4` gives `p = 0.018150`, so `τ ≤ 1/p ≈ 55` regardless of `dt_min`.
This is the `dt_floor` trap in a second constant, and it means **F4's fix never
reached the structural arm.**

`test_model.py` now guards this: "larger p shortens tau — b_g -4 → 55.1 tokens;
b_g -1 → 3.2 tokens" and "baseline reports no p term — p_bias None in 32/32".

This is an `architecture_spec.md` §7 decision of record (F2 fixed `b_g = -4`
deliberately), so it is not mine to change.

| option | consequence |
|---|---|
| A. Leave `b_g = -4` | Structural arm keeps a ~55-token ceiling the baseline never pays. Any null is then partly a handicap and must be reported that way. |
| **C. `use_permeability=False`** | **Removes the confound. τ means the same thing in both arms.** Drops the "soft reset at a boundary" half of the design — which D2 already measured as never engaging. |
| B. Lower `b_g` to −8 | Ceiling ~2,940 tokens, but F2's own note says this attenuates the gate's gradient ~53×, so the gate becomes even less likely to learn. |
| D. Add `p` to the baseline | **Excluded by standing rule** — never weaken the baseline. |

**Recommendation: C**, with B as a pre-registered secondary arm if budget allows.
D2 already established the gate contributes nothing to defend: p init 0.018150,
measured mean 0.017234/0.017193/0.017108 over 287,309,824 evaluations, full range
[0.0133, 0.0197], all mass in one bin.

---

## D-b. Which structural arm(s). Unresolved since T5c.

Three exist and are implemented: Decision 1 (per-position), T5c (per-window),
T5c-dual (joint).

**The evidence has hardened three times against per-position alone:**

1. keep(φ) = 0.1099 at 65,536 — the per-position mechanism addresses ~11% of
   φ's variance *by construction* (`p5_window_scan.json`).
2. Widening does not rescue it: 89.0% between-window at 65,536, and still 78.8%
   at 131,072.
3. **B2, new today:** refining the bins does not rescue it either. At 1 kb —
   5× finer — **81.9% of φ's variance at 65,536 is still between windows**
   (`b2_phi_resolution_probe.json`).

Three independent measurements, one conclusion: the bulk of the structural
signal is out of reach of a per-position bias at every resolution and every
window this hardware can run.

**Recommendation: baseline + T5c-dual**, two arms. Dual contains per-position as
a strict subset, so it cannot do worse by construction, and it costs +0.429%
params (budget 5%). It is the only arm with access to the other ~82–89%.

*Caveat owed:* the memcheck benchmarked `phi_granularity="position"` only, so
T5c-dual has **no throughput number yet**. It should be measured before the
schedule is fixed.

---

## D-c. Seeds and pairing.

Paired inits work (275/275 tensors bitwise identical) but have never been
exercised. Floors, at n per arm:

| n | unpaired permutation floor | paired sign-flip floor |
|---|---|---|
| 3 | 2/20 = 0.100 | 2/8 = **0.250** |
| 5 | 2/252 = **0.008** | 2/32 = 0.063 |
| 6 | — | 2/64 = 0.031 |

**The trap: at n = 3, pairing is *worse* than the unpaired test already run.**
Phase 4's p = 0.6000 sat against a floor of 0.1000 — p < 0.05 was unreachable
by design, and repeating that would waste the whole run.

**Recommendation: 5 paired seeds per arm.** Report the unpaired permutation test
as primary (floor 0.008) and pairing as variance reduction.

---

## D-d. Window: 65,536 or 131,072? **New, from today's measurement.**

`results/p5_memcheck.json`, measured, one L40S, fp32, batch 2, grad-checkpoint:

| window | baseline s/step | structural s/step | peak GiB | keep(φ) |
|---|---|---|---|---|
| 65,536 | 6.28 | 7.17 | 7.51 | 0.1099 |
| **131,072** | 15.45 | 16.98 | **14.90** | **0.2117** |

**131,072 fits** — 14.90 GiB of 44.39. This was never established before; the
2026-08-17 figures were never persisted and could not be used.

It **doubles** the reachable share of φ's variance for 2.41× the step cost. But
it does not solve the problem — 78.8% is still between windows — and it is the
expensive way to buy what D-b's per-window arm buys for +0.429% params.

**Recommendation: 65,536.** Take the reachable-variance problem with the
architecture (D-b), not with compute. Revisit 131,072 only if the dual arm
shows the window is binding.

---

## B3. Second cell line.

Pre-registered gate in `docs/B3_DEPTH_GATE.md`, written before the numbers
existed:

| | GM12878 | K562 | IMR90 |
|---|---|---|---|
| usable_frac (≥ 0.70) | 0.8039 | 0.8093 | 0.7999 pass |
| r insulation (≥ 0.95) | 0.9971 | 0.9966 | 0.9973 pass |
| r compartment (≥ 0.90) | 0.9711 | 0.9796 | **0.8815 FAIL** |

**The threshold was not moved.** IMR90 fails on `compartment_pc1` — which is
exactly the feature whose GM12878-vs-IMR90 divergence (r = 0.6035, against
K562's 0.8250) was the reason for preferring IMR90. The failure lands on the
feature that motivated the choice. Diagnosis says it is specific to the
compartment eigenvector, not map depth: IMR90's insulation agreement is the
*best* of the three, and a starved map would degrade both.

| option | consequence |
|---|---|
| **K562** | Only gate-passer, passes cleanly. Carries the CNV confound: near-triploid, and `log_contact_density` is a φ feature while compartment calling is CNV-sensitive — so part of any GM12878-vs-K562 difference is karyotype, not regulatory architecture. Also mismatches the GRCh38 reference. |
| Drop the second line | Scope the paper to one cell line. Costs the cross-cell-line claim entirely. |
| Investigate IMR90 further | On the explicit rule that if no computational fault is found, IMR90 stays out by the gate as written. |

**Recommendation: K562, with the aneuploidy documented as a limitation** — a
stated confound is worth more than no cross-cell-line arm, and the K562 chr14
build already exists. This is genuinely a judgement call and reasonable people
differ.

---

## B2′. Should the 1 kb gate be re-run against a matched reference?

**The gate FAILED as pre-registered and that verdict stands.** keep(φ)@65,536 =
0.1809 (1 kb), 0.1690 (2 kb), against the required 0.2198.

But the gate compared a 10 Mb tile with 7 features against a full-chr9,
8-feature reference. The same-tile 5 kb control run today shows the tile
*deflates* keep(φ) (0.0929 vs 0.1099, 0.85×), so the mismatch ran **against**
the candidates. Like-for-like on one tile:

| resolution | keep@65,536 | vs 5 kb | tile entries |
|---|---|---|---|
| 5,000 bp | 0.0929 | 1.00× | 600,161 |
| 2,000 bp | 0.1690 | 1.82× | 1,873,329 |
| 1,000 bp | 0.1809 | **1.95×** | 3,186,781 |

So the true resolution effect is 1.95× — a near-miss on a 2× gate.

**I have deliberately not rescored the gate against this control.** That would
be moving a goalpost after seeing the numbers.

**Recommendation: do not rebuild at 1 kb.** Two independent reasons that do not
depend on rescoring anything: (1) 1 kb buys only **1.071×** over 2 kb for 1.70×
the band entries, so 1 kb is the wrong target even if a rebuild happens; (2)
82% of the variance is still between windows at 1 kb, so a rebuild does not
change the strategic picture — it is D-b's problem, not a data problem.

---

## What is ready the moment these land

- `run_phase4.sh` passes the leakage-safe multichrom index (fixed today; it was
  silently defaulting to the chr9 pilot).
- Provenance defects closed: window, index schema, split roles, guard sentinel.
- Early stopping + `checkpoint_best.pt` implemented and verified.
- `test_model.py` 21/21, `test_phase4_wiring.py` 60/60, `validate_kernel.py` 34/34.
- C1 external benchmark harness wired and validated.

**Projected cost at the recommended settings** (5 paired seeds × 2 arms ×
8,000-step cap, 65,536): **149.4 GPU-h raw / 264.7 GPU-h DDP-scaled.** The raw
column is a measured lower bound; the DDP column applies the measured 32,768
gloo factor (1.753 / 1.788) and is a projection.

**The unsent admin request.** `docs/ADMIN_REQUEST_idle_culler.md` was drafted
2026-08-26 and never sent. At 265 GPU-h against a 10-minute idle culler, this is
the single highest-leverage action available and it costs nothing.
