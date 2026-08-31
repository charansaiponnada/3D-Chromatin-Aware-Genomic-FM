# 3D-Chromatin-Aware Genomic Foundation Model

Working document. Read this at the start of every session. Updated 2026-08-25.

**Two claims in the 2026-08-16 version of this file were RETRACTED on
2026-08-17 and are marked as such in §4. If you are working from a summary
that says the structural pathway is "live only in layers 12–15", or that "F1
is confirmed by measurement", that is the withdrawn version.** The
authoritative record is the change log in `project-docs/project.tex`; this file
reflects it through the 2026-08-18 entry.

**If you are picking up work: the ordered task list is
`docs/WORKPLAN_2026-08-25.md`.** T1-T4 there need no GPU and are not blocked by
the culler.

**A newer, phase-lettered plan supersedes the task-list framing above for
sequencing purposes: `docs/RESEARCH_PLAN_2026-08-26.md`.** Phase A (close the
stratification question) is running; do not start Phase B or touch the GPU
until A is closed and reported.

The original pre-research plan is archived at `docs/original_plan.md`.

---

## 1. The question

Does conditioning self-supervised pretraining on Hi-C-derived chromatin structure
improve a DNA language model over a sequence-only baseline, **at matched
parameters and matched compute**?

The contrast is with CHROME, which adds graph attention over *frozen* pretrained
embeddings. Here structure enters *during* pretraining, by biasing the SSM
timescale Δ. If the two turn out equivalent, that is a real finding and gets
reported as one.

---

## 2. Standing rules from the PI

These override anything else in this file.

1. **Never write a number into any file that did not come from a command you
   actually ran. Show the command output.**
2. **Do not change decisions recorded in `docs/architecture_spec.md` §7 (line
   529, "Decisions of record") without telling me first.**
3. **For every substantive change, add a dated entry to the change log in
   `project-docs/project.tex` and rebuild with
   `./3d-gen/bin/python scripts/build_project_doc.py`.**
4. **`project-docs/` must contain only `project.tex` and `project.pdf`.**
5. **Report nothing until a run has actually finished. No estimates.**

Corollary that has mattered in practice: when illustrating a hypothetical
("suppose the result came out at X"), say so explicitly in the same sentence.
An invented illustrative number has already been mistaken for a measured result
once.

---

## 3. Hard constraints

- **Compute:** 2× NVIDIA L40S, 44.39 GiB each. Nothing that needs more.
- **`nvidia-smi` does not work** on this box — NVML/kernel-module version
  mismatch. It does not affect running kernels. Read GPU memory through
  `torch.cuda.mem_get_info`. NCCL cannot initialise (it calls `nvmlInit_v2`), so
  DDP falls back to **gloo**. This is expected, not a fault.
- **The JupyterHub idle culler is the dominant operational constraint.**
  `jupyterhub_idle_culler --timeout=600` kills the *entire user cgroup* after 10
  minutes with no browser activity. `setsid`, `nohup` and reparenting to PID 1
  give no protection. Claude Code tool calls do **not** count as activity — only
  a live browser tab does. Phase 3 took three calendar days for nine hours of
  compute because of this.
  **The single highest-leverage fix available is asking an admin to raise or
  waive that timeout.** Not yet requested.
- **fp32 throughout, no autocast.** The Triton scan is validated for fp32 only.
- **Bitwise reproducibility is False** — `tl.atomic_add` accumulation across
  `d_inner` programs; measured relative spread ~1e-6.

---

## 4. Facts a new session must not re-derive

### Model
| | value |
|---|---|
| baseline params | **7,725,312** |
| structural params | **7,758,354** (+0.43%, budget is 5%) |
| d_model / n_layer / d_state / expand | 256 / 16 / 16 / 2 |
| d_struct / d_struct_raw | 2 / 8 |
| window | **65,536 bp** (1 token = 1 bp) — changed 2026-08-18; **nothing has trained at it.** Every measured number below is at 32,768 |
| Δ init | `dt_min=1e-6`, `dt_max=1e-1`, `dt_floor=1e-7` |

`W_dstruct` is zero-initialised, so the structural model is **numerically
identical to the baseline at step 0**. Anything that develops is learned.

### The F4 finding — do not lose this
τ = 1/(Δ·|A|), and `A = arange(1, d_state+1)` means |A| ≥ 1, so **τ_max at init
is exactly `1/dt_min`**. Mamba's reference `dt_min=1e-3` is a hard 1,000-token
ceiling present before the first gradient step. Measured: 999.5.

Fixed by three constants at **zero parameter cost**. Trained τ median went
14.2 → 434.7; TAD-scale mass ~0 → 4.846e-02; layer-directions reaching it
0/32 → 32/32.

**`dt_floor` is a trap.** It clamps the same quantity. Lowering `dt_min` alone
caps τ at `1/dt_floor` and looks like the change silently failed. `_init_dt`
now raises if `dt_floor > dt_min`.

**Scope limit.** Every measured result was taken at the 32,768 bp window,
where τ ≥ 100 kb is 3× the window and a 385 kb TAD is 11.7× it; at the new 65,536 bp
window, 1.5× and 5.9×. Beyond the window width no τ is behaviourally
distinguishable from any larger one. The supported claim is *"the state retains across the full
window"*, **not** "the model sees a TAD". The `tau_max >= 385_000` assertion in
`test_model.py` is a regression guard on the Δ init, not evidence of capacity.

### Measured results

All at the **32,768 bp** window, 2,000 steps.

| | val bits/nt | τ median |
|---|---|---|
| Phase 3 v1 baseline, 3 seeds (`dt_min=1e-3`) | 1.5210 ± 0.0040 | 14.2 |
| **Phase 3 v2 baseline, 3 seeds** | **1.5197 ± 0.0025** | 434.7 |
| **Phase 4 structural, 3 seeds** | **1.5217 ± 0.0045** | 487.3 |

Per seed — structural 1.5266 / 1.5177 / 1.5209, baseline 1.5196 / 1.5172 /
1.5223. Difference **+0.0020 bits (+0.80σ)**, does not clear the 0.0050 floor,
and the sign is the wrong one. Exact two-sided permutation p = **0.6000**
(minimum attainable at 3v3 = 0.1000, so p<0.05 was unreachable by design).

Every structural τ median exceeds every baseline τ median (+52.6 tokens).
Memory lengthened; loss did not improve.

Uniform-random floor is exactly **2.000** (ACGT; N and PAD never masked, never
contribute loss).

**σ_real = 0.0025.** The §4.1.3 effect-size floor is 2σ = **0.0050 bits**.

**The v1 `baseline_seed*` runs are NOT the Phase 4 baseline.** Different
architecture. They are kept as the evidence that justified the fix. The Phase 4
baseline is `baseline_v2_seed*`.

### P1 gate + D1–D3 — the Phase 4 verdict, and two retractions

`p1_swap_results.json`, 3 seeds, kernel floor **exactly 0.0**, masking floor
1.321e-01 (context only, never a threshold):

| control | Δ bits | KL | flip | KL/S0 |
|---|---|---|---|---|
| S0 removed | +0.0000 | 2.539e-05 | 0.0052 | 1.00 |
| S1 shuffled | +0.0001 | 6.334e-05 | 0.0073 | 2.49 |
| S2 shifted | +0.0001 | 5.528e-05 | 0.0068 | 2.18 |
| S3 distance | +0.0001 | 6.827e-05 | 0.0075 | 2.69 |
| S4 sequence | +0.0000 | 2.874e-05 | 0.0052 | 1.13 |

Benefit null; reliance non-zero but ~0.05% of the masking floor. The ordering
S3≈S1≈S2 ≫ S4≈S0 is what you get if divergence tracks *distance of the
substituted values from the true ones* — S0 is zeros, i.e. the standardised
mean. It shows the input reaches the output, not that structure is understood.
**This stands.**

**D1 = Var_t(W_dstruct·s)/Var_t(dt_proj·δ'), `d1_diagnostic.json`. Threshold
<0.05 ⇒ inert.** Median pooled D1 = **0.0004 / 0.0002 / 0.0002** — INERT in all
three seeds. **That verdict stands.** What was said about its *depth profile*
does not:

> **RETRACTED 2026-08-17 — "the pathway is inert through the encoder and live
> only next to the output head, a late readout correction."**

D1 is a ratio and the retraction is in its denominator. Band medians of the
stored numerator and denominator (`var_*_within`, reproduced 2026-08-25):

| seed | band | Var(W_dstruct·s) | Var(dt_proj·δ') | D1 |
|---|---|---|---|---|
| 0 | L00–L11 | 1.1060e-06 | 2.2285e-02 | 0.0001 |
| 0 | L12–L15 | 2.7630e-07 | 1.4347e-06 | 0.1671 |
| 1 | L00–L11 | 3.8980e-07 | 2.5165e-02 | 0.0000 |
| 1 | L12–L15 | 1.8843e-07 | 3.4627e-03 | 0.0002 |
| 2 | L00–L11 | 1.6919e-07 | 2.5406e-02 | 0.0000 |
| 2 | L12–L15 | 6.6493e-08 | 2.3558e-06 | 0.0280 |

The structural numerator **falls** late (×0.25, ×0.48, ×0.39); the
sequence-driven denominator **collapses** (×6e-5, ×0.1376, ×9e-5). D1 rises in
L12–15 because what it divides by falls away — in absolute terms the structural
pathway is 2–4× *weaker* in the "live" band than in the "dead" one. The
project's sharpest self-criticism, that the mechanism had degenerated into the
post-hoc integration it defines itself against, was **not supported by its own
stored numbers** and is withdrawn.

> **RETRACTED 2026-08-17 — "what they produce is near-constant across positions
> — F1 confirmed by measurement."**

keep(x) = mean Var_t(x) / Var_global(x), all 274 val windows
(`phase5_vars_diagnostic.py`, `p5_vars_diagnostic.json`). For W_dstruct·s,
L00–11 vs L12–15: 0.1945/0.1952, 0.1195/0.1169, 0.0794/0.0752. Early layers
vary with position indistinguishably as much, proportionally, as the live ones.
They produce a small position-varying signal, not a bias.

**D3 stands as a fact, but no longer as evidence for F1:** ‖W_dstruct‖_F
non-zero in 32/32 directions every seed (median 0.19–0.22) against the exact 0
it was initialised to. Gradient pressure reached every layer.

**D2: the permeability gate never engaged (F5). Stands.** p_t init =
softplus(−4) = 0.018150; measured mean 0.017234 / 0.017193 / 0.017108 over
287,309,824 evaluations, full range [0.0133, 0.0197], all mass in one bin. The
"soft reset at a boundary" half of the mechanism did not happen.

**The two pre-registered criteria disagree, and §4.1.3 said in advance that
disagreement is informative.** D_S1 > 0 ⇒ proceed; D1 < 0.05 ⇒ inert. By the
letter of the pre-registered 2×2 that is proceed to Phase 5. The goalpost was
fixed before the numbers existed; do not move it now.

### The binding constraint is the window — measured 2026-08-17

**keep(φ) = 0.0573** on the 274 val windows. Only 5.7% of φ's variance lies
*within* a 32,768 bp window; 94% is between windows. φ is on 5 kb bins, so a
window spans ~6.5 of them. The mechanism biases Δ **per position**, and within a
window its input is near-constant *by construction of the window size*. This is
a sufficient account of the Phase 4 null that does **not** require the mechanism
to be wrong.

**The encoder is exonerated.** keep(s)/keep(φ) = 2.2074 / 2.3464 / 1.2014, mean
**1.9184** — the encoder *increases* the within-window variance fraction of what
passes through it. §4.2bis recommended dropping it on the premise that it was
the bottleneck. It is not. **The `d_struct=8` no-encoder ablation is withdrawn
— do not run it** (~9 GPU-h to confirm nothing).

**Widening is a partial mitigation, not a fix** (`p5_window_scan.json`; φ only,
no model, 300 windows per width):

| window | bins | keep(φ) | between-window |
|---|---|---|---|
| 8,192 | 1.6 | 0.0092 | 99.1% |
| 16,384 | 3.3 | 0.0318 | 96.8% |
| 32,768 | 6.6 | 0.0490 | 95.1% |
| **65,536** | 13.1 | **0.1099** | **89.0%** |
| 131,072 | 26.2 | 0.2117 | 78.8% |
| 262,144 | 52.4 | 0.3499 | 65.0% |
| 524,288 | 104.9 | 0.4932 | 50.7% |

At the adopted 65,536 window **89.0% of φ's variance is still between
windows**; even at 524,288 — far beyond what fits in memory — half of it is.
**Never write "the window was the fault and widening fixes it" without this
table beside it.** The scan's 0.0490 at 32,768 and 0.0573 on the 274 val
windows are different samples of the same quantity, consistent.

**Consequence — PROPOSAL, not a decision (a §4.1.3 change, PI's call):** if
~90% of the structural variance is between windows at *every* width that fits
on this hardware, then a per-position Δ bias addresses the smaller share by
construction, at every reachable width. A complementary arm — one structural
embedding of the window's φ, injected once per window rather than per position
— is the only thing in the design space with access to the other 90%.

### The one clean arm separation in the project

`p5_positional_probe.json`. 500 train / 274 val windows, positions sampled every
512 bp, φ **withheld** at probe time (S0 zeros) so the structural arm cannot
copy its own input. Probe **B** predicts φ_t − mean_w(φ) from h_t − mean_w(h):
it removes window identity — 94% of the variance — and leaves exactly what the
Δ-bias mechanism acts on.

**Within-window insulation_100kb, val Pearson r, per seed:**

| | seed0 | seed1 | seed2 |
|---|---|---|---|
| structural | +0.0254 | +0.0303 | +0.0255 |
| baseline_v2 | +0.0134 | +0.0146 | +0.0221 |

**Every structural seed exceeds every baseline seed** — the maximally extreme
arrangement at 3v3, attaining the exact two-sided permutation floor
p = **0.1000**. It is on precisely the quantity the mechanism was built to move,
and it reproduces across seeds.

**Two caveats that must travel with it in the same breath.** Both arms sit
**below the composition floor** (+0.0802): local GC predicts within-window
insulation better than either trained model, so this is *relative*, not
absolute, structural sensitivity — a contrast between two incompetent
predictors. And the other two targets give nothing: B directionality is negative
for every run, B compartment overlaps between arms.

**The pooling hypothesis was wrong and is recorded as wrong.** The positional
probe was run because mean-pooling over 32,768 positions was suspected of hiding
local structure. Probe A is *lower* than the pooled probe throughout (compartment
+0.1819/+0.2283 against +0.5205/+0.4896 pooled), and on A compartment the
baseline leads. Pooling was denoising a window-level target, not destroying a
local one. The result that survived came from B, which was included for a
different reason.

**Where the evidence stands, net:**
1. A 30× longer memory horizon changed val loss by less than seed noise.
2. The structural arm is +0.0020 bits *worse* at 3 seeds — inside noise, but
   with no hint of benefit in any seed.
3. D1 says the mechanism is inert in all three seeds — but the depth story once
   told about *where* is retracted, and the measured reason it is inert is that
   its input barely varies inside the window it acts on.
4. Against that, the one probe built to isolate what the mechanism acts on
   separates the arms cleanly and reproducibly — below floor, at p = 0.1000,
   which is the most n=3 can attain.

### Data
chr9 only, GM12878, 4D Nucleome experiment set `4DNES3JX38V5` (Rao et al. 2014),
GRCh38. The 27.4 GB `.mcool` is **never downloaded** — read over HTTP range
requests. 27,679 bins at 5 kb, 21,519 usable (77.7%). 50% overlap on train
only, ~89 Mb unique sequence.

**Two indices exist. Do not pool them** — they are different datasets, for the
same reason v1 and v2 baselines must not be pooled.

| index | window | train / val / test | status |
|---|---|---|---|
| `dataset_index_w32768.npz` | 32,768 | 5,422 / **274** / 243 | every measured number in this file |
| `dataset_index.npz` (current) | **65,536** | 2,713 / 137 / 122 | nothing trained on it |

The 32 kb checkpoints and console logs are archived under
`results/{novel_model,baselines}/w32768/`. The `phase5_*.py` scripts glob
`NOVEL.glob("structural_seed*")` / `BASE.glob("baseline_v2_seed*")` at the top
level and will now find **nothing** — point them at `w32768/` to re-run any P5
analysis.

**The 273/242 counts are stale.** `build_index` used to thin the train grid to
get val/test; they are now enumerated independently at `STRIDE_EVAL == WINDOW`
(fixed 2026-08-16). Train starts are byte-identical. The Phase 3/4 *training*
evals ran on the old 273-window val set; P1 and D1 ran on the new 274. That is
why `p1_swap`'s "real" val bits (1.5271 / 1.5187 / 1.5226) differ slightly from
`metrics.json` (1.5266 / 1.5177 / 1.5209). Every P1 comparison is within one
dataset, so the gate is unaffected — but do not mix the two sources in a table.

φ validated against independent 4DN tracks: insulation_100kb r = **0.9969**,
compartment PC1 r = **0.9759**.

φ symmetry under reverse-complement: `[1,1,1,-1,1,-1,1,1]` — `directionality_2Mb`
and `upstream_mass_frac` flip sign. The flip must happen on **raw φ before the
encoder**, not on encoded `s` (failure mode F7).

### Throughput
**5.30 s/step** baseline, **6.13 s/step** structural (+16%, the extra encoder
passes), both at 32,768. 2,000 steps ≈ 2.9 h wall ≈ 5.9 GPU-hours per seed.
Matching is on steps and tokens, not wall clock.

**Throughput and memory at 65,536 and 131,072 are now on disk** —
`results/p5_memcheck.json` + `results/p5_memcheck_console.log`, measured
2026-08-31. The script printed a table and wrote nothing until it was given a
persistence path in the same session; the 2026-08-17 figures quoted in
`docs/NEXT_SESSION.md` §3 still have no file behind them and remain unusable.

Single process, no DDP, fp32, batch 2, `--grad-checkpoint` on, one L40S:

| window | baseline s/step | structural s/step | peak GiB |
|---|---|---|---|
| 32,768 | 3.02 | 3.43 | 3.81 |
| 65,536 | 6.28 | 7.17 | 7.51 |
| **131,072** | **15.45** | **16.98** | **14.90** |

**131,072 bp fits** — 14.90 GiB of 44.39, never established before. It costs
2.41× per step over 65,536 and doubles keep(φ) (0.1099 → 0.2117), so it is a
live Phase D option the pre-registration does not consider.

**These are a LOWER bound and must be scaled before any schedule uses them.**
This benchmark has no gradient sync; the historical real runs at 32,768 are
5.30 / 6.13 s/step against this bench's 3.02 / 3.43 at the same width, a
measured DDP-gloo factor of **1.753 / 1.788**. Applying that factor to wider
windows is a *projection*, not a measurement. Phase D at 5 paired seeds × 2
arms × 8,000 steps: **149.4 GPU-h raw / 264.7 GPU-h DDP-scaled** at 65,536;
**360.3 / 638.3** at 131,072 (projections, arithmetic in the 2026-08-31 change
log entry).

The 65,536 row independently reproduces B0(a)'s 7.51 GiB and 7.16 s/step.

---

## 5. Phase state

| Phase | State |
|---|---|
| 0 Literature | done — `docs/related_work.md` |
| 1 Data | done — `docs/data_card.md`, validated visually and against 4DN |
| 2 Mechanism design | done — `docs/architecture_spec.md` |
| 3 Baselines | **done twice.** v1 failed the F4 gate; v2 passes |
| 4 Structural arm | **done at 32 kb.** 3/3 seeds COMPLETED; P1 swap and D1–D3 run; two D1 conclusions retracted 2026-08-17 |
| 5 Evaluation | **diagnostics done, inference only** — vars, transfer probe, positional probe, window scan. No Phase 5 training. **PI decision pending**, see §6 |
| 6 Paper | not started |

**Nothing has trained since 2026-08-16.** The repo was re-wired for 65,536 bp
on 2026-08-18 — window, paired inits, a working `--grad-checkpoint`,
`--keep-every` — and every launch since has failed on GPU contention, not on
this code. See §6.

Phase 4 ran the **real-structure arm only** — the 3 seeds P1 needs. P2 (S1/S3
pretraining, 6 seeds, ~18 GPU-h) was deliberately not launched: P1 is the gate,
and P1 has now answered. Do not launch P2 to "double-check" the null; it
answers a question P1 already answered.

---

## 6. OPEN DECISION — what to run at 65,536

The previous open decision (separate reliance from benefit) was **resolved**:
pre-registered in `architecture_spec.md` §7 decisions 5–6 *before* any control
touched a trained model, then executed. That amendment stands and is closed.

**What the gate returned.** Loss flat, divergence live-but-tiny, D1 inert in all
three seeds. By the letter of the pre-registered 2×2, proceed to Phase 5. **The
reason it came out that way is now measured:** keep(φ) = 0.0573, i.e. the
mechanism's input barely varies inside the window it acts on. That is a
statement about the experiment, not about the mechanism — and it is not a
licence to assume the mechanism works, since widening leaves 89% of the variance
out of reach.

**Option B of the 2026-08-16 version of this section (one re-run, no encoder,
`d_struct=8`) is WITHDRAWN.** The encoder was measured to *increase*
within-window variance by 1.92×; it was never the bottleneck.

**Repo state as of 2026-08-18 — all verified, none exercised.** Window 65,536,
index rebuilt. **Paired inits** (weakness 3): `train.py` builds a reference
`BiMambaLM(ModelConfig(structural=False))` under the same seed and copies it
into every shape-matching tensor — **275/275 bitwise identical**, W_dstruct
still exactly zero. An earlier per-parameter name-hash reseeding was caught
before launch and discarded (it derived each std from an arm-dependent tensor,
and `hash()` is salted per process; it verified at 90/275).
**`--grad-checkpoint` fixed** — it had never worked: `train.py` called
`torch.utils.checkpoint.checkpoint()` without importing that submodule, and the
flag defaults off, so the fault was invisible for the whole project.
**`--keep-every`** (default 1,000) retains periodic checkpoints, so any later
diagnostic gets a trajectory instead of one endpoint. Post-widening:
`test_model.py` **17/17**, `test_phase4_wiring.py` **39/39**.

**Launch is blocked outside this repository.** Every 65 kb attempt died on
`NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at
CUDACachingAllocator.cpp:983`, under concurrent load from an unrelated project
on both devices — consistent with §3: NVML is broken on this box and the
allocator reaches its NVML path under memory pressure. **Single-GPU fallback is
the untried mitigation.** The culler is the other blocker and is still
unrequested.

**The plan the PI agreed to on 2026-08-17** (`docs/NEXT_SESSION.md` §4): 6–8
chromosomes rather than 22 (4,000 steps × 262k tok ≈ 1 epoch on ~1 Gb, and it
fixes the chr9-internal leakage), window 65,536, 4,000 steps, 3 paired seeds,
6 runs ≈ 80 GPU-h, plus downstream benchmarks (~1 day) and released checkpoints.
The framing is a **model**, not a study, targeting a zero-APC Scopus-indexed
subscription journal. Novelty is not the gap; scale and benchmarks are.

**Immediately actionable, blocked by nothing — the multi-chromosome data
build.** No GPU, so the culler does not touch it. `phase1_acquire.py:59` and
`phase1_features.py:40` hardcode `CHROM = "chr9"`; the `.mcool` is streamed over
HTTP and holds every chromosome. Per-script implementation notes are in
`docs/NEXT_SESSION.md` §5.1. The one genuinely new piece is a
`dataset_index.npz` carrying a chromosome id and splitting **by chromosome**
(train 4, hold out 2) — the current `assign_split`/`build_index` split by
coordinate and must not be reused as-is. Decide and record explicitly whether φ
is standardised per-chromosome or globally across the 6; it changes what
keep(φ) and every φ-derived number mean.

**Still undecided on top of that plan, PI's call:**

| option | cost | what it buys |
|---|---|---|
| Per-window conditioning arm | design + one run | the only proposal reaching the ~90% between-window share; a §4.1.3 change |
| Staged warm start from the 32 kb checkpoints | cheapest | Mamba has no positional embeddings, so those weights load into a 65,536 window unchanged; addresses weakness 1 (not converged) on one budget. But they predate the paired-init fix, so it carries weakness 3 forward. Fresh 65 kb training is the only route that gets both |
| Write up as a negative result now | days | F4 + the null + the keep(φ) curve + the probe-B separation is publishable and honest |

---

## 7. Controls (`architecture_spec.md` §4.1.3)

| | what it does | removes the explanation |
|---|---|---|
| S1 GLOBAL-PERM | φ permuted across all bins | — (primary reliance probe) |
| S2 CIRCULAR-SHIFT | φ rolled 10 Mb | "any smooth auxiliary channel helps" |
| S3 DISTANCE-MATCHED | φ recomputed from contacts resampled under P(s) | "structure is just genomic distance" |
| **S4 SEQUENCE-MATCHED** | φ replaced by 8 aligned 1D sequence covariates | "structure is just GC and gene density" |

S1 and S2 are implemented and tested (`apply_phi_control` in
`scripts/phase1_dataset.py`). **S3 is not built** — it needs the contact matrix,
not a permutation, and currently raises a directing `FileNotFoundError`.

**S4 is proposed, not pre-registered.** `compartment_pc1` correlates strongly
with GC content and gene density, both computable from sequence alone. Without
an *aligned* 1D control, a positive result has an obvious benign explanation.
S2 does not cover this — the objection needs alignment preserved. Adding S4 is a
§4.1.3 change and therefore a PI decision.

Controls draw from a **fixed** seed (20260815), deliberately not the training
seed: every training seed must see the *same* shuffled structure or seed variance
and shuffle variance are confounded.

---

## 8. Known weaknesses (reviewer-grade, all real)

1. **Not converged.** Val loss was still descending at step 2,000 and
   `W_dstruct` was still moving. A null from an undertrained model is not a null.
   Needs one long run per arm (8,000 steps, ~47 GPU-h for both).
2. **n = 3 bootstrap is not a valid procedure.** Ten distinct resampling
   multisets. Needs n ≥ 5, or an exact/permutation test stated as such.
3. **The comparison is unpaired — FIXED IN CODE 2026-08-18, not yet
   exercised.** The structural model instantiates extra modules, consuming RNG,
   so `structural_seedN` and `baseline_v2_seedN` did not share an init despite
   sharing a seed label. `train.py` now copies a same-seed reference baseline
   into every shape-matching tensor (275/275 bitwise identical). **Every
   result currently in this file was measured before that fix**, and only a
   re-run collects the power it buys.
4. **Data order is an uncontrolled nuisance confounded with arm.** Resume does
   not restore the dataloader position (`run_phase3.sh` header documents this),
   and runs were interrupted a variable, unrecorded number of times.
5. **Splits are within chr9.** Hi-C features are autocorrelated over megabases,
   so held-out φ leaks far more than held-out sequence — and it biases *toward*
   the structural arm. Cross-chromosome evaluation is required before any
   positive result is interpretable.
6. **Scale.** 7.7M params, one chromosome, one cell line, ~3 epochs. Claims must
   be scoped to that. Flagship journals will desk-reject on it; realistic targets
   are a Scopus-indexed subscription journal or a workshop.

---

## 9. Operational knowledge

**Convergence stopping and the best checkpoint** (added 2026-08-31). `train.py`
now takes `--early-stop` (`--early-stop-delta 0.0010`, `--early-stop-window
1000`, `--early-stop-patience 2` — the pre-registered values), with `--steps`
as a hard cap. **Off by default**, so every run on disk stays comparable.
Improvement is measured against the eval one window back, not the previous one:
eval-to-eval noise at σ_real = 0.0025 exceeds the 0.0010 threshold and would
stop the run on noise.

`checkpoint_best.pt` is written on every val improvement, separately from
`checkpoint.pt` (the resume point, overwritten every `--ckpt-every`), whether
or not early stopping is on. **Run probes on the best checkpoint, not the
endpoint** — they are not the same model.

`run_config.yaml` now records `stop_reason` (`converged`/`hard_cap`/
`incomplete`), `stopped_at_step`, `final_step`, `hard_cap_steps`,
`best_val_bits_per_nucleotide`, `best_step` and the four settings.

**Do not let the arms early-stop to different budgets and then compare them.**
That confounds mechanism with compute, which is the one thing this comparison
controls. Same cap and same criterion for all arms; `metrics.json` keeps every
eval, so the primary endpoint can be re-read at the step the shortest arm
reached. Not yet wired into `run_phase4.sh` — its `--steps 2000` and 3 seeds
contradict the 8,000 cap and D-c's 5 paired seeds, both PI decisions.

**Restart training** (the one command that restores everything — guard starts
supervisor, supervisor skips completed seeds and resumes partial ones):
```
cd /home/jupyter-238w1a5447/3d-gen && setsid nohup bash scripts/phase4_guard.sh >> results/novel_model/guard.log 2>&1 < /dev/null &
```
Then write the guard PID to `results/novel_model/guard.pid`.
Phase 3 equivalent is `scripts/phase3_guard.sh` — it will exit immediately now,
since its seeds are done.

**Status:** `bash scripts/phase4_status.sh`

**Progress never comes from the console log.** `run_phase4.sh` pipes through
grep, which block-buffers to a file, so `train_console.log` can lag by many
minutes or show nothing at all. `metrics.json` is written atomically by
`train.py` and never lags. Read it with
`scripts/phase3_progress.py '<glob>'`.

**Gotchas that have already cost time:**
- Bash reads shell scripts incrementally by byte offset — **never edit a running
  `.sh`**.
- After a kill, spawn children can survive and hold port 29511. Check with
  `ss -ltnp | grep 29511` before relaunching.
- `train.py --out-dir` exists because `RESULTS` was hardcoded to
  `results/baselines`; a Phase 4 run wrote there while its supervisor watched
  `results/novel_model` and would have retried forever.
- Checkpoints every 120 steps (~11 min), chosen because the culler was firing
  more often than a 200-step interval could save.
- `wall_clock_s` in `run_config.yaml` measures **only the final attempt**. For
  interrupted runs it badly understates. Derive compute from steps × s/step.
- `build_project_doc.py` decodes pdflatex output with `errors="replace"` — one
  non-ASCII byte in the `.tex` used to crash the build while pdflatex exited 0.
- **`phase4_guard.sh` reports false completion across configurations.** Line 33
  greps `results/novel_model/train_console.log` for `^ALL SEEDS DONE`. The 32 kb
  log satisfied that string, so every 65 kb launch exited immediately while
  appearing to have run. Archiving the logs to `w32768/` cleared it. The guard
  has no notion of which configuration a log belongs to — **this will recur at
  the next width change.**
- **`train.py --window` defaults to 32768 and `run_phase4.sh` never passes it.**
  `args.window` is used only at `train.py:769` (the `run_config.yaml` dump) and
  `train.py:718` (`windows_per_optimizer_step`); the real length comes from the
  index, so training is correct but *the record is not*. A 65 kb run writes
  `window: 32768` and **half its true token count** into the one artefact whose
  job is provenance — `results/novel_model/structural_seed0/run_config.yaml`
  already shows `window: 32768` beside `n_train_windows: 2713`. Matching is on
  steps *and tokens*. Pass `--window 65536`, or fix the default, before
  launching.

---

## 10. Verification before compute

- `scripts/test_model.py` — 17/17. Model invariants, param accounting, τ.
- `scripts/test_phase4_wiring.py` — **39/39** after the 65 kb widening
  (27/27 before it). The CLAUDE.md Phase 4 gate:
  shuffling φ changes the output once `W_dstruct` is non-zero, and provably does
  **not** at init.
- `scripts/validate_kernel.py` — 34/34 on the Triton scan.

Run these before launching anything that costs GPU-hours. Two real bugs have
been caught this way and one was caught four minutes into a launch that would
otherwise have looped forever.

---

## 11. Where things are

| what | where |
|---|---|
| **phased plan, current, supersedes the task list for sequencing** | **`docs/RESEARCH_PLAN_2026-08-26.md`** |
| **ordered task list, current** | **`docs/WORKPLAN_2026-08-25.md`** |
| handoff notes, 2026-08-17 | `docs/NEXT_SESSION.md` |
| plain-language project summary | `docs/project_summary_plain.md` |
| literature | `docs/related_work.md` |
| data sources, limitations | `docs/data_card.md` |
| mechanism, failure modes, decisions of record | `docs/architecture_spec.md` |
| GPU operations | `docs/gpu_runbook.md` |
| Phase 3 baselines | `results/baselines/` |
| Phase 3 v2 result, written up | `results/baselines/phase3_report_baseline_v2.txt` |
| Phase 4 structural arm | `results/novel_model/` |
| dated change log | `project-docs/project.tex` → `.pdf` |
| original pre-research plan | `docs/original_plan.md` |

---

## 12. What not to do

- Do not pool v1 and v2 baseline runs. Different architectures.
- Do not read τ for the structural arm from `dt_proj` alone — `W_dstruct` is
  added before the softplus, and the resulting number looks entirely plausible
  while being the baseline's.
- Do not launch P2 (S1/S3 pretraining) before P1 has been read. 18 GPU-hours
  answering a question P1 answers for free.
- Do not edit `train.py` or any supervised `.sh` mid-sweep.
- Do not report a number from a run that has not reached `status: COMPLETED`.
- Do not pool the 32,768 and 65,536 indices. Different datasets.
- Do not run the `d_struct=8` no-encoder ablation — withdrawn 2026-08-17.
- Do not repeat "the pathway is live only in layers 12–15" or "F1 confirmed by
  measurement". Both were retracted on 2026-08-17; see §4.
- Do not describe the window as the fault that widening fixes. 89% of φ's
  variance is still between windows at 65,536.
