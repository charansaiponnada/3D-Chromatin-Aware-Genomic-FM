# 3D-Chromatin-Aware Genomic Foundation Model

Working document. Read this at the start of every session. Updated 2026-08-16.

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
| window | **32,768 bp** (1 token = 1 bp) |
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

**Scope limit.** The window is 32,768 bp, so τ ≥ 100 kb is 3× the window and a
385 kb TAD is 11.7× it. Beyond ~32,768 no τ is behaviourally distinguishable
from any larger one. The supported claim is *"the state retains across the full
window"*, **not** "the model sees a TAD". The `tau_max >= 385_000` assertion in
`test_model.py` is a regression guard on the Δ init, not evidence of capacity.

### Measured results
| | val bits/nt | τ median |
|---|---|---|
| Phase 3 v1 baseline, 3 seeds (`dt_min=1e-3`) | 1.5210 ± 0.0040 | 14.2 |
| **Phase 3 v2 baseline, 3 seeds** | **1.5197 ± 0.0025** | 434.7 |
| Phase 4 structural, seed 0 only | 1.5266 | 473.8 |

Uniform-random floor is exactly **2.000** (ACGT; N and PAD never masked, never
contribute loss).

**σ_real = 0.0025.** The §4.1.3 effect-size floor is 2σ = **0.0050 bits**.

**The v1 `baseline_seed*` runs are NOT the Phase 4 baseline.** Different
architecture. They are kept as the evidence that justified the fix. The Phase 4
baseline is `baseline_v2_seed*`.

**Two negative results already in hand, both worth reporting:**
1. A 30× longer memory horizon changed val loss by less than seed noise.
2. Structural seed 0 finished 0.0069 bits *worse* than the baseline mean
   (+2.74σ), worse than every baseline seed individually. One seed; seeds 1 and
   2 decide whether it holds.

### Data
chr9 only, GM12878, 4D Nucleome experiment set `4DNES3JX38V5` (Rao et al. 2014),
GRCh38. The 27.4 GB `.mcool` is **never downloaded** — read over HTTP range
requests. 27,679 bins at 5 kb, 21,519 usable (77.7%). Windows: 5,422 train /
273 val / 242 test, 50% overlap, ~89 Mb unique sequence.

φ validated against independent 4DN tracks: insulation_100kb r = **0.9969**,
compartment PC1 r = **0.9759**.

φ symmetry under reverse-complement: `[1,1,1,-1,1,-1,1,1]` — `directionality_2Mb`
and `upstream_mass_frac` flip sign. The flip must happen on **raw φ before the
encoder**, not on encoded `s` (failure mode F7).

### Throughput
**5.30 s/step** baseline, **6.13 s/step** structural (+16%, the extra encoder
passes). 2,000 steps ≈ 2.9 h wall ≈ 5.9 GPU-hours per seed. Matching is on
steps and tokens, not wall clock.

---

## 5. Phase state

| Phase | State |
|---|---|
| 0 Literature | done — `docs/related_work.md` |
| 1 Data | done — `docs/data_card.md`, validated visually and against 4DN |
| 2 Mechanism design | done — `docs/architecture_spec.md` |
| 3 Baselines | **done twice.** v1 failed the F4 gate; v2 passes |
| 4 Structural arm | **running.** seed 0 COMPLETED, seed 1 in progress, seed 2 pending |
| 5 Evaluation | not started |
| 6 Paper | not started |

Phase 4 runs the **real-structure arm only** — the 3 seeds P1 needs. P2 (S1/S3
pretraining, 6 seeds, ~18 GPU-h) is deliberately not launched: P1 is the gate,
and if the mechanism is inert the spec says stop.

---

## 6. OPEN DECISION — blocking, and time-critical

**The P1 gate as written cannot distinguish its own alternative hypotheses.**

MLM over 32 kb windows is dominated by local k-mer statistics. A flat loss delta
is consistent with *the mechanism is inert*, *the mechanism works but MLM cannot
express it*, and *structure carries no information* — and §4.1.3 assigns the
same verdict to all three, which is "do not proceed to Phase 5". Phase 5 is
where the claim lives.

Evidence the metric lacks power: a 30× memory-horizon change moved it 0.0013
bits.

**Proposed fix — measure reliance separately from benefit.** Reliance is
prediction divergence (KL between logits under real vs shuffled φ, or argmax
flip rate), not a loss delta. It is an inference-time measurement on existing
checkpoints.

| | loss moves | loss flat |
|---|---|---|
| divergence high | structure helps | used but not useful for MLM → proceed to Phase 5 |
| divergence low | (impossible) | **inert → stop** |

**This must be written into §7 before the P1 swap runs.** It is genuine
pre-registration now; it stops being so the moment the swap executes.
PI decision, not mine.

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
3. **The comparison is unpaired.** The structural model instantiates extra
   modules, consuming RNG, so `structural_seedN` and `baseline_v2_seedN` do not
   share an init despite sharing a seed label. Fixing this costs zero compute and
   buys real power, but requires re-running the structural arm.
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

---

## 10. Verification before compute

- `scripts/test_model.py` — 17/17. Model invariants, param accounting, τ.
- `scripts/test_phase4_wiring.py` — 27/27. The CLAUDE.md Phase 4 gate:
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
