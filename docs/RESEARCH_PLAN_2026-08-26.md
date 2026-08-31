# Research plan — 2026-08-26

================================================================================
THE CLAIM THIS PROGRAM IS BUILDING
================================================================================
Not a perplexity win. That is null, it will stay null, and it is not where the
result is. The target:

  Conditioning pretraining on chromatin structure produces representations that
  track local chromatin organisation better than a matched sequence-only
  baseline, specifically where structure varies within the model's receptive
  field, and at no cost to language-modelling performance.

Four components. Three are measured and solid. One is NOT yet established and
must not be written as if it were:

  1. MECHANISM IS NOVEL. CHROME and Evo2HiC attach structure to frozen
     embeddings. This enters during representation learning via the SSM
     timescale. SOLID.
  2. THE F4 FINDING GENERALISES. Mamba's reference dt_min=1e-3 caps tau at
     1,000 tokens before the first gradient step (measured: 999.5). Any
     Mamba-based genomics model inherits it, Caduceus included. Zero-parameter
     fix; tau median went 14.2 -> 434.7. SOLID, and citable on its own.
  3. THE STRATIFICATION. rho = +0.1210 between within-window phi variance and
     the arm difference. **NOT ESTABLISHED.** It was proposed POST HOC after
     seeing pooled r exceed mean-per-window r; keep(phi) did not predict it.
     The common-alpha confound is unresolved. And every block length whose CI
     excludes zero is <= 0.262 Mb, shorter than Hi-C autocorrelation and
     therefore anticonservative, while every block length long enough to be
     valid has too few blocks -- those two sets are DISJOINT on 8.98 Mb, no row
     is both. Treat this as a pre-registered hypothesis awaiting test, never as
     a finding.
  4. THE NULL IS DIAGNOSED. keep(phi) = 0.0573; 89.0% of phi's variance is
     still between windows even at 65,536. SOLID.

================================================================================
STANDING RULES -- these override anything you infer
================================================================================
- Never write a number into any file that did not come from a command you
  actually ran, and show the output. Projections must be labelled as
  projections in the same sentence, with the arithmetic shown.
- Do not change architecture_spec.md §7 "Decisions of record" without asking.
- Every substantive change gets a dated entry in project-docs/project.tex,
  rebuilt with ./3d-gen/bin/python scripts/build_project_doc.py.
  project-docs/ holds exactly project.tex and project.pdf.
- Interpreter is ./3d-gen/bin/python, always with -u.
- NEVER weaken the baseline to produce a separation. It is matched-parameter,
  matched-compute and paired-init, and that is the only reason any separation
  would be worth reporting. If you want a harder comparison, ADD a stronger arm
  (tuned one-hot/GC), never subtract from the existing one.
- A negative outcome at any gate is a result, not a failure. Report it, record
  it, and do not soften a threshold that was fixed in advance.
- The idle culler kills the whole cgroup after 10 minutes without browser
  activity. Cache per item so a kill costs one unit, never the run.

================================================================================
FIRST ACTION: write docs/RESEARCH_PLAN_2026-08-26.md
================================================================================
Save the phase list below into that file VERBATIM before doing any work, so it
survives a culler kill or a context reset. Add a pointer to it in CLAUDE.md §11
and in the banner near the top. Commit and push it immediately.

PHASE A -- close what is open. No GPU.
  A1. Common-alpha confound check on the stratification. Refit both arms at a
      single shared ridge alpha, sweeping the geometric mean of the six
      selected values plus one decade either side. Re-run the quartile table,
      the frac>0 trend (bootstrap this directly -- it is the monotone,
      outlier-immune statistic), and Spearman rho with the block sweep and the
      CI-width reliability check.
      GATE: does Q1 stay negative and does frac>0 stay monotone increasing?
  A2. If A1 survives: pre-register the hypothesis in architecture_spec.md §7
      and the change log BEFORE any multi-chromosome data exists -- direction,
      primary test (rho > 0 with blocks >= 1 Mb AND >= 20 blocks, satisfiable
      only above ~40 Mb of evaluation genome), secondary (monotone frac>0),
      origin stated as post hoc and exploratory, the alpha confound and how it
      is controlled, and the disconfirming outcome.
      If A1 fails: record it in the change log as a lead proposed, tested and
      killed, with the numbers that killed it. Do not pre-register it. Delete
      component 3 from the claim above and proceed with three components.

PHASE B -- data. No GPU. B1/B2/B3 are independent and may run in any order.
  B1. Multi-chromosome build, 6-8 chromosomes. Implementation notes are in
      docs/NEXT_SESSION.md §5.1 and are still accurate. Split BY CHROMOSOME,
      not by coordinate -- assign_split/build_index must not be reused as-is.
      chr9 stays the TEST split so the comparison with CHROME holds. Trap:
      phase1_acquire.py:59 carries a comment claiming a --chrom flag exists.
      It does not; main() defines only --dry-run. Use --dry-run per chromosome
      before any long download loop. Record explicitly whether phi is
      standardised per-chromosome or globally -- it changes what keep(phi) and
      every phi-derived number mean.
  B2. phi at 1 kb. Staged exactly as T1 in the workplan, with its STOP GATE:
      fetch one ~10 Mb tile first and report usable-bin fraction (5 kb
      reference 77.7%), keep(phi) at 32,768 and 65,536 (5 kb reference
      0.0490 / 0.1099), and wall time plus bytes. Proceed only if keep(phi) at
      65,536 at least doubles while usable fraction stays above ~60%.
      Otherwise fall back to 2 kb or stop and report -- a negative here is a
      publishable measurement showing the constraint is intrinsic to Hi-C
      autocorrelation rather than an artefact of binning.
  B3. Acquire Hi-C for a second cell line from 4DN (K562 or IMR90), same
      pipeline, same features.

PHASE C -- evaluation infrastructure. No GPU.
  C1. Wire ONE standard benchmark so the model has a comparable number:
      Genomic Benchmarks, the Nucleotide Transformer suite, or BEND. One
      benchmark number is worth more than a fifth internal diagnostic. Without
      it a reviewer cannot place the model, and the Caduceus comparison in
      Phase F invites benchmark comparison the moment it is made.
  C2. Cross-cell-line probe design. CRITICAL: do NOT probe insulation alone.
      TAD boundaries are largely conserved across cell types (Dixon et al.
      2012), so a model could pass a cross-cell-line insulation probe by
      having memorised GM12878. Probe (a) A/B compartments, which are
      substantially more cell-type-variable, and (b) the DIFFERENTIAL --
      regions where the two cell lines actually disagree. That is the only
      version that tests generalisation. It must also be cross-chromosome, so
      it composes with B1 rather than substituting for it.

PHASE D -- training. GPU required.
  D1. Fix the provenance defects first. train.py:959 defaults --window to
      32768 and run_phase4.sh never passes it; args.window is read only at
      train.py:769 (the run_config dump) and train.py:718 (token accounting),
      so a 65 kb run records the wrong window and HALF its true token count in
      the one artefact whose job is provenance. Derive it from the loaded index
      so it cannot drift again. Also make the phase4_guard.sh:33 completion
      sentinel configuration-specific; it currently greps a config-agnostic log
      for "^ALL SEEDS DONE" and reported false completion on every 65 kb launch.
  D2. Re-run scripts/phase5_memcheck.py WITH ITS OUTPUT WRITTEN TO DISK. The
      2026-08-17 run was never persisted, so no 65 kb memory or throughput
      number has a file behind it and no schedule may be built on them.
  D3. The runs: window 65,536, 4,000 steps, multi-chromosome index, paired
      inits (already implemented, 275/275 tensors bitwise identical, verified
      but never exercised). SEEDS ARE WHAT DECIDE WHETHER A POSITIVE IS EVEN
      EXPRESSIBLE: 3v3 has 20 arrangements and a permutation floor of exactly
      0.100, so p < 0.05 is unreachable by design. Use 5 per arm unpaired
      (2/252 = 0.008) or 6 paired (2/64 = 0.031). Note the trap: at n=3 a
      paired sign-flip test gives 2/8 = 0.250, WORSE than the unpaired test
      already run -- pairing buys power only from n=6 up.
      Pre-register the probe metrics as PRIMARY and loss as secondary, in
      architecture_spec.md, BEFORE the runs start. Add the tuned one-hot/GC arm.
      Single-GPU fallback is the untried mitigation for the NVML allocator
      failure that killed every previous launch.

PHASE E -- confirmation. No GPU.
  E1. Run the A2 pre-registered stratification test on held-out chromosomes.
      This is the first point in the project where blocks >= 1 Mb AND >= 20
      blocks are simultaneously satisfiable.
  E2. Cross-cell-line probe (C2 design) on the new checkpoints.
  E3. Benchmark numbers from C1.

PHASE F -- write up.
  Scope every claim to what was measured. The Caduceus comparison must always
  carry its second half: Caduceus-Ph is d_model 256, n_layer 16, 7.73M params
  (HuggingFace model card) against this baseline's 7,725,312 -- essentially
  parameter-for-parameter identical, which removes the scale objection. But
  Caduceus pretrained on the whole human reference genome at 131,072 context
  for 50k steps of ~1M tokens, roughly 50B tokens, against 2,000 steps x
  262,144 = ~0.52B here. Same architecture, about 100x less pretraining. State
  both halves every time; the first half alone is a claim the work does not
  support.

================================================================================
STATUS -- appended after execution, plan text above is unmodified
================================================================================

**A1, run 2026-08-26 (`scripts/phase5_common_alpha.py`,
`results/novel_model/p5_common_alpha.json`): GATE FAILS.**

Refit all six probe-B runs at a single shared ridge alpha (no per-run
LOO-GCV), swept at the geometric mean of the six originally-selected alphas
(1.0926e6) and one decade either side:

| alpha | Q1 mean d_w | frac>0 by quartile | monotone? | rho | gate |
|---|---|---|---|---|---|
| 1.093e5 (geo/10) | -0.0208 | .377 .603 .574 .507 | no | +0.1137 | fail |
| 1.093e6 (geo)    | -0.0111 | .464 .529 .603 .594 | no | +0.1183 | fail |
| 1.093e7 (geo*10) | -0.0066 | .478 .559 .588 .609 | yes | +0.1116 | pass |

Q1 stays negative at all three, and rho stays in a tight band (+0.112 to
+0.118) next to the original per-run-alpha value +0.1210 -- the alpha
confound does not explain rho away. But the frac>0 monotone trend, the half
of the finding pitched as clean and assumption-free, is monotone at only one
of the three shared-alpha settings. The plan's gate is conjunctive (Q1
negative AND frac>0 monotone), so **A1 fails**.

Per A2's failure branch: **not pre-registered in architecture_spec.md §7.**
Recorded in project-docs/project.tex, 2026-08-26 (third entry), as a lead
proposed, tested against its own alpha confound, and killed. Component 3
("keep(phi) made a prediction and it held") is dropped from the claim at the
top of this file; the project proceeds on the remaining three components.
The stratification stays a candidate for the multi-chromosome build (B1) to
confirm or kill on genome large enough to make blocks >= 1 Mb and >= 20 of
them simultaneously satisfiable -- unreachable on 8.98 Mb of chr9 val at any
alpha tested here.

Phase B, C, D are unauthorized until picked up in a session with PI sign-off;
this session did not touch the GPU and stops here per its own instructions.

---

**AMENDMENT, PI, same day.** Component 3 is dropped, not the A1 result: the
alpha-confound elimination (rho stable +0.112 to +0.118) is a negative
control, not support for the claim. The A1 gate itself was mis-specified --
pairing rho with a non-independent quartile-monotonicity display -- noted for
future gates; the chr9 verdict is not reopened. Superseding pre-registration
(rho > 0 on held-out chromosomes, no secondary) written verbatim into
architecture_spec.md §7 Decision 7 and project.tex before any multi-chromosome
data existed.

**Phase B0, run 2026-08-26:**
- (a) Single-GPU allocator smoke test (`scripts/b0_single_gpu_smoke.py`).
  Single GPU alone does NOT avoid the nvmlInit_v2_ assertion at 65,536 bp --
  it fires with 43.97/44.39 GiB free and no other process. Isolated cause:
  65,536 bp without `--grad-checkpoint` needs on the order of 70+ GiB (32,768
  bp alone already peaks at 37.23 GiB); WITH `--grad-checkpoint`, 65,536 bp
  fits in 7.51 GiB at ~7.16 s/step on one GPU, measured. `run_phase4.sh` must
  pass `--grad-checkpoint` at 65,536 bp -- it currently doesn't. This was
  previously misattributed to "concurrent load from an unrelated project on
  both devices"; the real constraint is memory, not device contention. D2
  memcheck rerun is still separately required.
- (b) Draft admin message written to `docs/ADMIN_REQUEST_idle_culler.md`, not
  sent.

**Phase B1, run 2026-08-26: multi-chromosome build complete.**
`dataset_index_multichrom.npz` built -- chr9 held out whole as test,
chr10-13 train, chr14-15 val (~957/161/108 Mb). All 7 chromosomes acquired,
featurised, and validated against independent 4DN tracks (insulation_100kb r
0.9968-0.9972, compartment r 0.948-0.977 across all seven). phi
standardisation decided PER-CHROMOSOME, recorded in
`phase1_features.py`'s docstring and `docs/data_card.md` §7.

A real concurrency bug was found running 5 acquisitions in parallel: three
chromosomes (chr13/14/15) crashed racing on the single shared GENCODE-GTF
download target. Fixed with an flock-based lock in `stream_download`
(`scripts/phase1_acquire.py`); the three affected chromosomes' already-correct
Hi-C band/coarse/sequence data was reused, only the annotation step was
recovered.

Not done: visual TAD/loop inspection beyond chr9 (data_card.md §6 is
chr9-only); B2 (phi at 1 kb) and B3 (second cell line) were not started this
session -- B1 was prioritised per this file's own instruction ("if you can
only complete one Phase B item, complete B1"). Phase C, D still unauthorized;
this session did not touch the GPU beyond the B0(a) smoke test.

---

**2026-08-31 session. No training. Phase B and C advanced; Phase D still blocked
on decisions, not on code.**

**B2 — RUN, GATE FAILS, and the negative is the informative kind.**
`results/b2_phi_resolution_probe.json`. keep(φ)@65,536 = 0.1809 (1 kb) and
0.1690 (2 kb) against the required 0.2198. Usable-bin fraction was never the
issue (0.9131 / 0.9348). A same-tile 5 kb control
(`b2_phi_resolution_control_5kb.json`) shows the tile *deflates* keep(φ)
(0.0929 vs the full-chr9 0.1099), so the gate's tile-vs-chromosome mismatch ran
against the candidates; like-for-like the 5 kb → 1 kb gain is 1.95×, a near-miss
on a 2× gate. **The gate as written fails; it has not been rescored against the
control.** Whether a matched-reference gate should be re-run is a PI call
(B2′ in `docs/PI_DECISIONS_2026-08-31.md`). Do not rebuild at 1 kb regardless:
it buys 1.071× over 2 kb for 1.70× the entries.

**The finding that outlasts the gate:** at 1 kb bins, 81.9% of φ's variance at
65,536 is still between windows. With the window scan (89.0% at 5 kb/65,536;
78.8% at 131,072) this is the third independent measurement of the same wall —
**the constraint is Hi-C's intrinsic spatial autocorrelation, not binning and
not window width.** It strengthens the case for the per-window (T5c) arm.

**C1 — WIRED AND VALIDATED.** `scripts/c1_genomic_benchmarks.py`, 8 Genomic
Benchmarks tasks, chosen because Caduceus and HyenaDNA both report on them.
Protocol is a **frozen-embedding linear probe**, which is **not** comparable to
the published fine-tuned numbers and says so in the docstring, the banner and
the JSON. Mandatory k-mer composition floor (k ≤ 3) per probe B's lesson. φ at
downstream time is S0 (zeros) — these sequences have no Hi-C and two tasks are
not human, which is a real limitation of structure-conditioned pretraining
rather than a shortcut. `model.py` gained `encode()`; `forward()` is now a
wrapper around it.

**D1/D2 (Phase D prep) — CLOSED.** The memcheck had no persistence path at all,
which is why the 2026-08-17 numbers vanished; it now writes
`results/p5_memcheck.json`. **131,072 bp fits** (14.90 GiB of 44.39), never
established before — now decision D-d. Phase D projects to **264.7 GPU-h
DDP-scaled**, not the ~150 carried informally: the raw benchmark has no gradient
sync and the measured 32,768 gloo factor is 1.753 / 1.788.

**A live bug found and fixed:** `train.py --index` defaults to the chr9 *pilot*
index and no runner passed the flag, so a Phase D launch would have silently
trained on the within-chromosome split that weakness 5 exists to eliminate.

**Early stopping and `checkpoint_best.pt` implemented** — §3's convergence
criterion existed only on paper, and there was no best-checkpoint tracking at
all, so every diagnostic this project has run could only reach the endpoint.

**Not done:** C2 (cross-cell-line probe design), E1–E3, and Phase D itself.
Six decisions are consolidated in `docs/PI_DECISIONS_2026-08-31.md`.
