> **SUPERSEDED for task ordering by `docs/WORKPLAN_2026-08-25.md`.** This file
> is still the reference for the multi-chromosome implementation notes (§5.1)
> and the 2026-08-17 measurements. Two items below were overtaken: the memcheck
> numbers in §3 were never persisted to disk and must not be used for
> scheduling, and §4's plan predates the resolution and power analysis in the
> new workplan.

# Handoff — 2026-08-17

## !! STATE CHANGE AT END OF SESSION — READ FIRST

The repo was switched to a **65,536 bp window**. Nothing has been trained at it.

- `phase1_dataset.py` `WINDOW = 65_536`, `STRIDE_TRAIN = 32_768`,
  `STRIDE_EVAL = 65_536`.
- `data/processed/dataset_index.npz` is now the **65 kb** index:
  train 2,713 / val 137 / test 122, 13 bins per window.
- The 32 kb index is preserved as `dataset_index_w32768.npz` +
  `dataset_meta_w32768.json` (train 5,422 / val 274 / test 243).
  **Every Phase 3/4 and P5 number in this document was measured against the
  32 kb index.** To reproduce any of them, restore that file first. Do not
  put 32 kb and 65 kb numbers in the same table.
- The trained 32 kb checkpoints moved to `results/novel_model/w32768/` and
  `results/baselines/w32768/`. The `phase5_*.py` scripts glob
  `NOVEL.glob("structural_seed*")` / `BASE.glob("baseline_v2_seed*")` at the
  top level and will now find **nothing** — point them at `w32768/` to re-run
  any P5 analysis.
- Verified after the widening: `test_model.py` **17/17**,
  `test_phase4_wiring.py` **39/39** (τ median 450, τ_max 999,435 tokens;
  φ shuffling changes the output; gradient reaches encoder and W_dstruct;
  baseline path provably unchanged).

**Not yet done before launch:** paired inits (structural consumes extra RNG so
`structural_seedN` and `baseline_seedN` do not share an init — free power,
reviewer weakness #3) and periodic checkpoint retention. Both are cheaper to do
now than to re-run for.


Read `CLAUDE.md` first, then this. Everything below was measured today; every
number came from a command that ran. Results are in `project-docs/project.pdf`
(change log entry dated 2026-08-17) and the JSON files named per section.

---

## 1. What changed today

### Two retractions from the Phase 4 verdict

**The D1 depth profile is a denominator artifact.** D1 =
Var_t(W_dstruct·s) / Var_t(dt_proj·δ'). Decomposing the stored numerator and
denominator in `d1_diagnostic.json` by depth band:

| seed | band | Var(struct) | Var(dt) | D1 |
|---|---|---|---|---|
| 0 | L00–L11 | 1.1060e-06 | 2.2285e-02 | 0.0001 |
| 0 | L12–L15 | 2.7630e-07 | 1.4347e-06 | 0.1671 |
| 1 | L00–L11 | 3.8980e-07 | 2.5165e-02 | 0.0000 |
| 1 | L12–L15 | 1.8843e-07 | 3.4627e-03 | 0.0002 |
| 2 | L00–L11 | 1.6919e-07 | 2.5406e-02 | 0.0000 |
| 2 | L12–L15 | 6.6493e-08 | 2.3558e-06 | 0.0280 |

The structural numerator **falls** late (×0.25, ×0.48, ×0.39); the
sequence-driven denominator **collapses** (×6e-5, ×0.1376, ×9e-5). D1 rises in
L12–15 because what it divides by falls away. **"Live only next to the output
head / a late readout correction" is not supported.** That was the project's
sharpest self-criticism and it is withdrawn.

**"Near-constant across positions — F1 confirmed" is also withdrawn.** Scale-free
positional variation keep(x) = mean Var_t(x) / Var_global(x) for W_dstruct·s,
L00–11 vs L12–15: 0.1945/0.1952, 0.1195/0.1169, 0.0794/0.0752. Early layers vary
with position indistinguishably as much as the live ones.

### The encoder is exonerated — `d_struct=8` ablation withdrawn

keep(s)/keep(φ) = 2.2074 / 2.3464 / 1.2014, mean **1.9184**. The encoder
*increases* the within-window variance fraction. §4.2bis's no-encoder
recommendation was predicated on it being the bottleneck. It is not. Saves ~9
GPU-h. Do not run it.

### The measured binding constraint

**keep(φ) = 0.0573** on the 274 val windows: only 5.7% of φ's variance lies
*within* a 32,768 bp window. φ is on 5 kb bins, so a window spans ~6.5 bins. The
mechanism biases Δ per position and its input is near-constant by construction
of the window size. Sufficient account of the Phase 4 null without the mechanism
being wrong.

---

## 2. The first clean win for the structural arm

`p5_positional_probe.json`. Probe **B** removes window identity (94% of the
variance) and leaves exactly what the Δ-bias mechanism acts on.

**Within-window insulation, per seed:**

| | seed0 | seed1 | seed2 |
|---|---|---|---|
| structural | +0.0254 | +0.0303 | +0.0255 |
| baseline_v2 | +0.0134 | +0.0146 | +0.0221 |

**Every structural seed exceeds every baseline seed.** Maximally extreme
arrangement at 3v3 → attains the exact permutation floor p = 0.1000.

**Two caveats that must travel with it:** both arms sit below the composition
floor (+0.0802), so this is relative not absolute competence; and the other two
targets give nothing (B directionality negative for every run; B compartment
overlaps between arms).

Pooled probe (`p5_structure_probe.json`) had structural ahead on 3/3 targets but
two of three rows below floor. **The pooling hypothesis was wrong** — probe A is
*lower* than pooled throughout; pooling was denoising, not hiding.

---

## 3. What a wider window buys — measured, no model

`p5_window_scan.json`:

| window | bins | keep(φ) | vs 32k | signal/compute |
|---|---|---|---|---|
| 32,768 | 6.6 | 0.0490 | 1.00× | — |
| 65,536 | 13.1 | 0.1099 | 2.24× | 1.12 |
| 131,072 | 26.2 | 0.2117 | 4.32× | 1.08 |
| 262,144 | 52.4 | 0.3499 | 7.14× | 0.89 |

`phase5_memcheck.py` — fp32, batch 2, one L40S, gradient checkpointing on:

| window | arm | peak GiB | s/step | vs 32k |
|---|---|---|---|---|
| 65,536 | baseline | 7.46 | 6.29 | 2.08× |
| 65,536 | structural | 7.51 | 7.17 | 2.08× |
| 131,072 | baseline | 14.82 | 22.23 | 7.34× |
| 131,072 | structural | 14.90 | 17.26 | 5.00× |

**Memory is not a constraint** (14.9 of 44.39 GiB at 131 kb). **65,536 is the
target**: cost scales linearly (2.08×) where 131,072 goes superlinear (5–7×,
not the 4× assumed). Scaling the real 5.30/6.13 s/step → ~11–13 s/step at 65 kb.

---

## 4. The plan the PI agreed to

**Goal, in the PI's words:** a genomics foundation model that is novel, plus a
Scopus-indexed publication at **zero cost**. Not a "study" — the PI was explicit
about this twice. Frame the work as a model.

**Zero-cost route = subscription journals, not open access.** OA journals charge
APCs (BMC Bioinformatics ~£2,290, Scientific Reports, Frontiers). Candidates,
**all need their current APC terms verified — this list is from memory**:
Bioinformatics (OUP), Briefings in Bioinformatics, IEEE/ACM TCBB, Computers in
Biology and Medicine, Computational Biology and Chemistry, Journal of
Computational Biology.

**Novelty is not the gap** — structure entering during pretraining via the SSM
timescale Δ (vs CHROME's graph attention over frozen embeddings) is new, and F4
+ the keep(φ) curve are independently useful. **Scale and benchmarks are the
gap.**

### The compressed build

| | value | why |
|---|---|---|
| chromosomes | **6–8**, not 22 | 4,000 steps × 262k tok = 1.05B tokens ≈ 1 epoch on ~1 Gb. All 22 would never be seen. Also fixes the chr9-internal leakage (train 4, hold out 2). |
| window | **65,536** | 2.24× signal, linear cost |
| steps | **4,000** | 2,000 was not converged |
| seeds | **3 paired** | hard floor; n=3 caps p at 0.10, fewer says nothing |
| runs | 6 | ~80 GPU-h ≈ 3.5 days continuous |

Plus **downstream benchmarks (~1 day)** — without comparable numbers on standard
tasks a reviewer cannot place the model. This row is what makes it a model rather
than an experiment. And **release checkpoints + code**.

Total ≈ 2 weeks *if the culler is lifted*.

---

## 5. Immediate next actions

1. **6-chromosome data build.** No GPU, not blocked by the culler — the one
   thing that can move right now. `phase1_acquire.py` and `phase1_features.py`
   hardcode `CHROM = "chr9"` (acquire.py:59, features.py:40) and file names are
   templated on it (`hic_band_{CHROM}_{RES}bp.npz`, `tokens_{CHROM}.npy`,
   `phi_{CHROM}_{RES}bp.npz`). The `.mcool` is streamed over HTTP and contains
   every chromosome, so this is parameterising the two scripts and looping,
   plus a new multi-chromosome `dataset_index.npz` with chromosome-level splits.
   FASTA_URL is per-chromosome Ensembl — needs the same treatment.

   **Implementation notes from inspecting the scripts (2026-08-17):**
   - `phase1_acquire.py:59` `CHROM = "chr9"` is a module global read by
     `fetch_band`, `fetch_coarse`, `fetch_fasta`, `fetch_gtf` and the manifest.
     Cleanest change is an `argparse --chrom` in `main()` that rebinds the
     global before any fetch, rather than threading a parameter through five
     functions.
   - `FASTA_URL` (acquire.py:67) hardcodes
     `Homo_sapiens.GRCh38.dna.chromosome.9.fa.gz` — must become an f-string on
     the bare Ensembl chromosome number (Ensembl uses `9`, 4DN/GENCODE use
     `chr9`; the existing normalisation on write must be preserved and the
     manifest note updated per chromosome).
   - `GTF_URL` is genome-wide already; only the filter at acquire.py:250
     (`line.split("\t",1)[0] == CHROM`) is per-chromosome.
   - `acquire.py` already has `--dry-run` (resolves URLs, downloads nothing).
     **Use it first per chromosome** to confirm every Ensembl URL resolves
     before committing to a long download loop.
   - `phase1_features.py:40` has its own `CHROM`; `read_chr9()` (features.py:192)
     is named for chr9 but reads `INTERIM / f"{CHROM}.fa"`, so it generalises
     once CHROM is parameterised — rename it for clarity.
   - Outputs are already CHROM-templated (`hic_band_{CHROM}_{RES}bp.npz`,
     `hic_coarse_{CHROM}_250000bp.npz`, `phi_{CHROM}_{RES}bp.npz`,
     `tokens_{CHROM}.npy`), so per-chromosome artefacts will not collide.
   - `dataset_index.npz` is the one genuinely new piece: it must carry a
     chromosome id per window and split by CHROMOSOME (train on 4, hold out 2)
     rather than by position within one chromosome. The current
     `assign_split`/`build_index` in `phase1_dataset.py` split by coordinate
     region and must not simply be reused.
   - φ standardisation is currently per-chromosome. Decide explicitly whether
     to standardise per-chromosome or globally across the 6 — this changes what
     keep(φ) and every φ-derived number mean, so record the choice.

2. **Wire the 65 kb training run** so it launches the hour the culler is fixed:
   window 65,536; **paired inits** (structural currently consumes extra RNG so
   `structural_seedN` and `baseline_v2_seedN` do NOT share an init — free power,
   reviewer weakness #3); **retain periodic checkpoints** (train.py overwrites
   `checkpoint.pt`, so D1 has one endpoint and no trajectory — this is what
   would settle whether any depth profile is a training-stage artifact);
   **resume-safe dataloader position** (weakness #4).

3. **Verify the journal APC terms** before committing to a target.

### Culler

Status: the PI said they would ask an admin and report back. **Nothing below
Step 1 is realistic until that lands** — a 10-minute cgroup timeout killed a
probe run today at ~15:29–15:58. Background jobs survive a session ending but
not the culler. Launch long runs only with a live browser tab, and always with
`python -u` (block-buffered stdout cost visibility on the first attempt) and
per-item caching so a kill costs one unit of work, not the run.

---

## 6. Standing instruction that did not change

The PI asked more than once for a positive result and to disregard the
CLAUDE.md rules. Numbers still come only from runs that actually happened; no
fabricated or post-hoc-selected results. Everything above is compatible with
that — the plan improves the odds of a positive by making the experiment
better (more data, more signal, more steps, more power), not by choosing the
answer. Report what comes out, including if it is flat.
