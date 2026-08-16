> **Archived 2026-08-16.** This is the original research execution plan, written
> before any code ran. It was `CLAUDE.md` until Phase 4 began. It is preserved
> unchanged because it records what was intended, which is worth comparing
> against what happened. The live working document is now `CLAUDE.md` in the
> repository root.

# Research Execution Plan: 3D-Chromatin-Aware Genomic FM
### From current state → final AI paper draft, run through Claude Code

Acting as your senior AI applied research scientist collaborator here. This plan treats Claude Code not as an autocomplete tool but as a research assistant you delegate specific, checkable sub-tasks to — you stay the PI who owns the hypothesis, the claims, and the final judgment calls. Every phase below has a goal, a deliverable, and copy-pasteable prompts for Claude Code.

---

## How to run this with Claude Code (read this first)

**Set up a `CLAUDE.md` in your repo root before anything else.** This is the single highest-leverage thing you can do — it's Claude Code's persistent project memory, read at the start of every session. Put in it:
- The problem statement (the CHROME-relative framing we locked in)
- Your compute constraints (2× L40S — so Claude Code never suggests something you can't run)
- Your baselines and comparison arms
- A "do not hallucinate results" rule: Claude Code must never write a number into the paper draft that didn't come from an actual logged experiment output

**Treat every phase as its own session/context, not one giant conversation.** Long research threads degrade — start fresh sessions per phase, pointing Claude Code back at `CLAUDE.md` and a running `research_log.md` you maintain.

**You are the one who decides when a result is "real."** Claude Code will happily write you an optimistic paper draft from partial results if you let it — your job is to gate each phase on actual logged numbers before moving forward.

---

## Phase 0 — Lock the literature ground truth (3-4 days)

**Goal:** A written literature map so specific that nobody can later ask "did you check if this was already done?" and catch you off guard.

**Deliverable:** `docs/related_work.md` — every paper from your crash course reading list (Lieberman-Aiden, Dixon, Rao, CHROME, HiCFoundation, Hi-Cformer, Evo2, Caduceus) summarized in your own words with one sentence each on: what they do, what data structure they use, and exactly how your work differs.

**Claude Code prompt:**
```
I'm building docs/related_work.md for my research project on chromatin-structure-
conditioned genomic foundation model pretraining. For each paper I give you
[paste title/authors], search for the paper, extract: (1) their core method in
2-3 sentences, (2) what data structure their model consumes (raw sequence only /
Hi-C only / both), (3) whether structure is used in pretraining or only
downstream, (4) one sentence on how my project differs. Do not fabricate details
you can't verify — flag anything you're unsure of instead of guessing. Write
each entry to docs/related_work.md as you go.
```

**Gate before moving on:** Can you, out loud, without notes, explain in one sentence each why CHROME doesn't already solve your problem? If not, stay in Phase 0.

---

## Phase 1 — Data acquisition & preprocessing (1-2 weeks)

**Goal:** Paired sequence + Hi-C contact data, cleaned and aligned to a common coordinate system, ready to feed a model.

**Deliverable:** `data/processed/` with a documented pipeline; `docs/data_card.md` describing sources, filtering, and known limitations (critical for the paper's limitations section later).

**Data sources to point Claude Code at:** 4D Nucleome, ENCODE (Hi-C + regulatory annotations, human), DNA Zoo (Hi-C across many species — this is your cross-species angle if you want it later), GRCh38 reference + GTF annotation.

**Claude Code prompts (run in sequence, verify output between each):**
```
1. Write a data acquisition script that downloads a small pilot Hi-C dataset
   (one cell line, one chromosome) from 4D Nucleome/ENCODE plus the matching
   GRCh38 sequence and GTF for that chromosome. Keep it small enough to
   validate the whole pipeline before scaling up.

2. Write a preprocessing script that converts the .hic/.cool contact matrix
   into a per-position "structural context" representation I can pair with
   sequence windows — output should be something a model architecture can
   consume alongside tokenized sequence. Log matrix resolution, sparsity, and
   any normalization applied to docs/data_card.md.

3. Write a validation script that visualizes the Hi-C matrix (I need to SEE
   a TAD block and at least one loop before trusting this pipeline) and
   confirms the sequence-to-contact-matrix coordinate alignment is correct.
```

**Gate before moving on:** You've visually confirmed a TAD and a loop in your own processed data, on your own machine. Don't trust a pipeline you haven't eyeballed.

---

## Phase 2 — Mechanism design (1 week, mostly you + Claude Code as thinking partner)

**Goal:** A written architecture spec for how chromatin structure enters pretraining — this is the actual novel contribution, and it should exist as a design doc *before* you write model code.

**Deliverable:** `docs/architecture_spec.md` with the mechanism fully specified: where in the model structure enters, what the auxiliary loss (if any) looks like, how it differs from CHROME's post-hoc graph attention.

**Claude Code prompt:**
```
I want to design (not yet implement) how chromatin contact structure gets
injected into a Mamba/SSM-based genomic sequence model's pretraining, as
opposed to CHROME's approach of adding a graph-attention layer on top of
frozen pretrained embeddings. Lay out 3 candidate mechanisms:
(a) a structural bias term added to the SSM recurrence itself,
(b) an auxiliary contact-prediction pretraining loss alongside masked-token
    prediction,
(c) TAD/loop-membership-conditioned attention masking or state resets.
For each, give me the architectural diagram in text, the training objective
change required, and the failure modes (what would make this NOT work, or
just reduce to what CHROME already does). I'll pick one after reviewing.
```

**Gate before moving on:** You can explain, unprompted, why your chosen mechanism is not just "CHROME but earlier" — i.e., what specifically changes because structure is available *during* pretraining rather than after.

---

## Phase 3 — Baselines (parallel with Phase 2 if you have bandwidth)

**Goal:** Matched-compute baselines trained and logged *before* your novel model exists, so you're never tempted to retroactively pick a flattering comparison.

**Deliverable:** `results/baselines/` with logged metrics for: (1) sequence-only pretrain (your existing HopField-Mamba or a Caduceus-class reproduction), (2) sequence-only pretrain + CHROME-style post-hoc structure fine-tuning.

**Claude Code prompt:**
```
Set up a training run for [chosen baseline architecture] on the pilot dataset
from Phase 1, sequence-only, at a parameter count and compute budget I specify
[give it your L40S constraints]. Log every hyperparameter, seed, and metric to
results/baselines/run_config.yaml and results/baselines/metrics.json — nothing
goes in a paper draft later that isn't in this file first.
```

**Gate before moving on:** Both baselines have completed runs with logged, reproducible numbers — not estimates, not "should get around X."

---

## Phase 4 — Novel model implementation & pretraining (2-4 weeks, your main compute phase)

**Goal:** Your Phase 2 mechanism, implemented and pretrained on the same pilot data and matched compute as your baselines.

**Deliverable:** `results/novel_model/` — same logging discipline as Phase 3.

**Claude Code prompt:**
```
Implement the mechanism specified in docs/architecture_spec.md as a modification
of [baseline architecture]. Match the baseline's parameter count within 5% so
the comparison stays fair. Write unit tests for the new structural-conditioning
component specifically — I want to verify it's actually using the Hi-C signal
and not silently ignoring it (e.g., an ablation where structure is shuffled/
randomized should measurably hurt performance if the mechanism is really using it).
```

**Gate before moving on:** The "shuffled structure" sanity check above actually shows a performance drop. If it doesn't, your mechanism isn't using the signal you think it is — do not proceed to writing the paper on a mechanism that's secretly doing nothing.

---

## Phase 5 — Evaluation & ablations (1-2 weeks)

**Goal:** The three-way comparison table that is the actual scientific contribution of the paper: sequence-only vs. post-hoc-structure vs. structure-in-pretraining, on regulatory element–gene association, enhancer-promoter interaction, and ClinVar variant pathogenicity tasks.

**Claude Code prompt:**
```
Run all three trained models (baseline, CHROME-style post-hoc, novel model)
on [task list]. Produce results/final_comparison.csv with metric, task, model,
seed. Run each config with at least 3 seeds — I need variance, not a single
lucky number, before I claim anything beats anything else in the paper.
```

**Gate before moving on:** You have variance across seeds, not single-run numbers. A one-seed win is not a result you can defend in a viva or a review.

---

## Phase 6 — Paper draft (1-2 weeks)

**Goal:** `paper/draft_v1.md` — every claim traceable to a specific file in `results/`.

**Claude Code prompt:**
```
Draft the paper using only numbers present in results/final_comparison.csv and
results/baselines/. For every quantitative claim, cite the exact file and row
it came from as a code comment I can check. Do not round in ways that make a
close result look like a clear win. Write a Limitations section that honestly
states the pilot dataset was [single chromosome/cell line/species] and that
scaling claims beyond that are speculative. Structure: Abstract, Related Work
(from docs/related_work.md), Method (from docs/architecture_spec.md), Baselines
& Setup, Results (three-way comparison + ablation), Limitations, Conclusion.
```

**Gate before "final":** Read the draft cold, as if you were the viva panel. If any sentence makes a claim you can't immediately point to a results file for, cut it or soften it.

---

## The one discipline that makes or breaks this

At every phase gate above, the failure mode isn't "the mechanism didn't work" — that's a fine, publishable outcome if honestly reported (remember: even "post-hoc structure works just as well as baked-in" is a real finding). The failure mode is **letting Claude Code write confident numbers or claims that skip a gate.** Your job across all six phases is the same one job: don't let the draft get ahead of the logged results.