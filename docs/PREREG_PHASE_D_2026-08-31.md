# Pre-registration — Phase D pretraining run

Written 2026-08-31, **before any run is launched**, per the practice used for
Decisions 5–6. Nothing in this file may be edited after launch except by a
dated amendment that says what changed and why.

**Status: DECISIONS RECORDED 2026-08-31, PI. Launched.** See §0.1 below; §0's
option tables are left unedited as the record of what was weighed.

---

## 0.1 DECISIONS AS TAKEN — PI, 2026-08-31

Recorded **before launch**. The option tables in §0 below are left unedited as
the record of what was weighed.

| # | Decision | Flag |
|---|---|---|
| **D-a** | `use_permeability=False` (option C) | `--no-permeability` |
| **D-b** | T5c-dual, baseline + structural-dual | `--phi-granularity dual` |
| **D-c** | 5 paired seeds per arm | `SEEDS="0 1 2 3 4"` |
| **D-d** | window 65,536 | from the index |
| convergence | prereg §3 criterion, 8,000 hard cap | `--early-stop --steps 8000` |
| split | leakage-safe multichrom | `--index dataset_index_multichrom.npz` |

**D-a is a change to `architecture_spec.md` §7 (failure mode F2 fixed
`b_g = -4`).** It is taken on the grounds that `p` imposes τ ≤ 1/p ≈ 55 tokens
on the **structural arm only** — the baseline has no `p` term — so every
comparison before this ran the hypothesis arm with ~8.5× shorter memory than
the arm it was meant to beat. D2 measured the gate as never engaging (mean
0.0172 over 287,309,824 evaluations, all mass in one bin), so nothing measured
is given up. Recorded as Decision 10 in `architecture_spec.md` §7.

**Parameter accounting under these settings, measured:** baseline
**7,725,312**, structural-dual without `p` **7,758,386** — **+0.428%**, against
a 5% budget.

**Runner:** `scripts/run_phase_d.sh`. Two single-GPU lanes rather than DDP:
NCCL cannot initialise on this box so DDP falls back to gloo, measured 1.75×
slower per step, while a 65,536 run with gradient checkpointing peaks at
7.51 GiB and two independent runs fit trivially in 2×44.39 GiB.

**Unchanged from §1–§6:** primary endpoint, secondary endpoints, falsification
criterion. None were altered after the decisions were taken.

---

## 0. Decisions this run cannot be designed without — PI's call

### D-a. What to do about `p` (the 2026-08-31 finding). **Blocking.**

The permeability term was absent from every τ ever reported. Recomputed from
the logged checkpoints, the structural arm's true τ median is ~51 tokens, not
~487 — roughly 8.5× **shorter** than the baseline's 434.7, because the baseline
carries no `p` term at all. `b_g = -4` gives `p = 0.018`, and `τ ≤ 1/p ≈ 55`
regardless of `dt_min`. This is the `dt_floor` trap in a second constant.

This is an `architecture_spec.md` §7 decision of record (failure mode F2 fixed
`b_g = -4` deliberately, trading init-equivalence against gradient scale). It
is therefore not mine to change. Options:

| option | consequence |
|---|---|
| **A. Leave `b_g = -4`** | The structural arm keeps a ~55-token ceiling the baseline never pays. Any null is then partly attributable to the handicap, and must be reported that way. |
| **B. Lower `b_g` to −8** | `p = 3.4e-4`, ceiling ~2,940 tokens. F2's own note says this attenuates the gate's gradient ~53×, i.e. the gate becomes even less likely to learn. |
| **C. `use_permeability=False`** | Removes the confound entirely; tests the Δ-bias mechanism alone. Drops the "soft reset at a boundary" half of the design, which D2 already measured as never engaging. |
| **D. Add `p` to the baseline too** | Makes the arms comparable by giving both the same floor. Changes the baseline, so it is **not** available: standing rule, never weaken the baseline. |

**Recommendation: C**, with B as a pre-registered secondary arm if budget
allows. C is the only option that makes τ mean the same thing in both arms,
and D2 already established the gate contributes nothing to defend. D is
excluded by the standing rule.

### D-b. Which structural arm(s). Unresolved since T5c.
Decision 1 (per-position) / T5c (per-window) / T5c-dual. keep(φ) = 0.1099 at
65,536 says per-position addresses ~11% of φ's variance by construction;
per-window is the only arm with access to the other 89%.
**Recommendation: baseline + T5c-dual**, two arms, since dual contains
per-position as a strict subset and costs +0.429% params.

### D-c. Seed budget and pairing.
Paired inits now work (275/275 bitwise identical, unexercised). At n per arm:
unpaired permutation floor is 2/C(2n,n); paired sign-flip floor is 2/2^n.
5 unpaired = 0.008; 6 paired = 0.031; **3 paired = 0.250, worse than the
3v3 already run.** **Recommendation: 5 paired seeds per arm** (floor 2/32 =
0.0625 paired, 0.008 unpaired — report the unpaired permutation test as
primary and pairing as a variance reduction, which is the stronger of the two).

---

## 1. Primary endpoint — fixed in advance

**Validation bits/nt on held-out whole chromosomes (chr14, chr15), at matched
steps and matched tokens.** One number per run. Superiority requires the
structural arm to beat the baseline by ≥ 2σ_real, where σ_real is the
across-seed SD of the **baseline arm in this run** (not the 0.0025 measured at
32 kb — that was a different dataset and window and may not carry over).

**Secondary, pre-registered, reported whether or not primary passes:**
1. Probe B within-window insulation_100kb Pearson r, φ withheld, vs the GC
   composition floor. This is the one arm separation the project has.
2. D1 (Var ratio) with the **p-corrected** τ, per depth band.
3. τ median and τ_max, `p`-inclusive, both arms.

**Not endpoints, exploratory, and labelled as such:** compartment/directionality
probes, window scans, per-feature ablations.

## 2. Split — the leakage fix

Primary and only analysis: `dataset_index_multichrom.npz`. chr10–13 train,
chr14–15 val, chr9 held out whole as test and **not touched** until the paper
is written. The chr9-internal split is retired to "pilot, not evidence" and
every number derived from it must carry that label.

Owed and not yet run: a global-standardisation sensitivity analysis, since
per-chromosome z-scoring normalises held-out chromosomes with their own
statistics (transductive).

## 3. Convergence criterion — replaces the fixed step count

Weakness 1 says a null from an undertrained model is not a null. Train until
**either**: val bits/nt improves by < 0.0010 over 1,000 steps (two consecutive
evals), **or** a hard cap of 8,000 steps, whichever comes first. Record which
one fired for every run. All arms get the same cap and the same criterion; if
any arm hits the cap, the comparison is reported at matched steps **and** the
cap is reported as a limitation.

## 4. What would falsify the hypothesis

A result where the structural arm does not beat the baseline by ≥ 2σ on
primary, **with** τ verified `p`-inclusive as ≥ the baseline's and D1 ≥ 0.05,
is a clean negative: the mechanism was expressible, was live, and did not help.
That is the publishable negative result, and it is the outcome this run is
designed to be able to reach. It will not be softened after the fact.

## 5. Not in this run

The CHROME-style post-hoc comparator (step 4) and the DNALongBench downstream
task (step 5) are separate, later, and separately pre-registered. This run
establishes a trustworthy MLM comparison on a leakage-safe split and nothing
more.

## 6. Provenance

`scripts/build_manifest.py` must be run and `results/MANIFEST.json` committed
immediately after the last run completes. `train.py --window` must be passed
explicitly (it defaults to 32768 and is written into `run_config.yaml`,
corrupting the token accounting — still unfixed). `phase4_guard.sh:33`'s
`^ALL SEEDS DONE` sentinel is not configuration-aware and will report false
completion across a width change — archive or namespace the console log first.
