# What this project is and what has happened so far

Written 2026-08-16. Plain language, for anyone picking this up cold.

Every number in this file comes from a file in the repository. Nothing is
estimated or remembered. Where a number is worked out from another one, it says
so.

---

## 1. The question

DNA is a string of four letters. But inside a cell it is not a straight line —
it is folded up, and pieces that sit far apart along the string end up touching
each other in space. That folding is not random. It controls which genes turn on.

Nearly every AI model built for DNA reads the letters and ignores the folding.
A few bolt the folding on at the end, after the model has already finished
learning.

This project asks a different question:

> **If you show a model the folding while it is learning, does it learn better
> than a model that only ever sees the letters?**

The answer might be no. That is fine — "no" is a real finding, and the whole
project is set up so that a "no" can be reported honestly instead of buried.

---

## 2. How the project is organised

Six phases. Each one has to produce something checkable before the next starts.

| Phase | What it is | State |
|---|---|---|
| 0 | Read the existing literature, write down exactly how this differs | done |
| 1 | Get the data, clean it, check it by eye | done |
| 2 | Design the mechanism on paper before writing any model code | done |
| 3 | Train the plain letters-only model — the yardstick | **done, twice** |
| 4 | Train the folding-aware model | **running now** |
| 5 | Test both on real biology tasks | not started |
| 6 | Write the paper | not started |

The rule that holds the whole thing together: **no number goes into the paper
that did not come out of a run that actually finished.**

---

## 3. The data

One chromosome, one cell type, one species. Deliberately small — the point was
to get the whole pipeline working before scaling anything.

- **Chromosome 9 of the human genome** (GRCh38 reference)
- **Folding data:** in situ Hi-C from the 4D Nucleome project, experiment set
  `4DNES3JX38V5`, cell line GM12878. This is the Rao et al. 2014 dataset — the
  most detailed human folding map available.
- The folding file is 27.4 GB. It was **never downloaded**; it is read over the
  internet a piece at a time.

From the folding map, 8 numbers are computed for every 5,000-letter chunk of the
chromosome, describing things like "is this a boundary between two folded
domains" and "is this region in the active or inactive compartment".

**The data was checked against an independent source before being trusted.**
The 4D Nucleome project publishes its own version of one of these 8 numbers.
Ours agrees with theirs at a correlation of **0.9969** — near perfect, on a
track we never looked at while building the pipeline. A second one agrees at
0.9759.

The chromosome splits into 27,679 chunks, of which **21,519 are usable** (77.7%).
The rest are gaps and repetitive regions where the folding data is meaningless.

Training uses **5,422 windows** of 32,768 letters each, with 273 held back for
testing and 242 never touched at all.

---

## 4. The model

Same model in both arms, so the comparison is fair:

- **7,725,312 numbers** (parameters) in the letters-only version
- **7,758,354** in the folding version — **0.43% bigger**, well inside the 5%
  fairness budget agreed in advance

The task it learns is fill-in-the-blank: hide 15% of the letters, predict them.
Score is in "bits per letter" — **lower is better, and pure guessing scores
exactly 2.0**.

The folding version differs in one place. The model has an internal dial that
controls how far back it remembers. In the folding version, that dial is nudged
by the 8 folding numbers at each position. At the start of training the nudge is
set to exactly zero, so the two models are *identical* on step one. Anything
that develops after that is learned, not built in.

---

## 5. What actually happened

### 5.1 The first attempt at the yardstick (7–12 August)

Three training runs of the letters-only model finished. Score: **1.5210 bits**,
give or take 0.0040 between runs.

But a check built into the run failed. The model could only remember about
**14 letters** at a time by the end of training. The folding features describe
chunks of 5,000 letters. The model physically could not hold enough to use what
we were about to feed it.

That check existed precisely to catch this before spending real compute. It did.

### 5.2 Finding the cause (13 August)

The cause was one constant in the code.

The memory length works out to `1 / (Δ × |A|)`, and because of how `A` is set
up, `|A|` is never less than 1. So the longest possible memory at the start of
training is exactly `1 / dt_min`. The standard value everyone copies from the
original Mamba paper is `dt_min = 0.001`, which puts a hard ceiling of
**1,000 letters** on memory — before a single training step has run.

Measured ceiling before the fix: **999.5 letters**. That is not a coincidence,
it is the formula.

Three constants were changed. **No parameters were added.** There was a trap
along the way: a second constant clamps the same quantity, and lowering only the
first would have capped memory at 10,000 letters and looked like the fix had
silently failed. The code now refuses to run if those two constants disagree.

### 5.3 Re-doing the yardstick properly (13–15 August)

All three runs were redone on the fixed model. The old runs were kept, not
overwritten — they are the evidence that justified the change.

| | before the fix | after the fix |
|---|---|---|
| score (3 runs) | 1.5210 ± 0.0040 | **1.5197 ± 0.0025** |
| typical memory | 14 letters | **435 letters** |
| fraction of the model's internal state that remembers past 100,000 letters | about zero | **4.8%** |
| parts of the model reaching that range | 0 out of 32 | **32 out of 32** |

**The single most important result so far, and it is a negative one:** memory
got 30 times longer and the score did not improve at all. The difference is
smaller than the difference between two runs of the same model.

So "longer memory helps" is not the story. Whatever the folding buys, it has to
buy it some other way.

One limit worth stating plainly: the model reads 32,768 letters at a time. A
folded domain can be 385,000 letters. So the honest claim is "the model now
remembers across its whole reading window", **not** "the model can see a folded
domain". It never sees one. The 8 folding numbers are how long-range information
gets in — that is the entire point of the design.

### 5.4 The folding model (15 August – now)

Before spending any compute, 27 automated checks were written to confirm the
folding data actually reaches the model and is actually used. All 27 pass. The
important one: scrambling the folding data changes what the model predicts —
except at the very start, where it provably does not, exactly as designed.

One real bug was caught four minutes into the first launch. The training script
wrote its results to the wrong folder, and the supervising script was watching
the right folder for a "finished" signal that would never appear. The run would
have trained perfectly and then been restarted forever. Fixed, relaunched.

Three folding runs are now going. Where the first one stands:

| step | folding model | the three yardstick runs at the same step |
|---|---|---|
| 200 | 1.9814 | 2.8445 / 2.3335 / 2.5400 |
| 400 | 1.8254 | 1.8249 / 1.7971 / 1.8898 |
| 600 | 1.7528 | 1.7296 / 1.7631 / 1.7424 |
| 800 | 1.6364 | 1.5936 / 1.6104 / 1.6180 |
| 1000 | 1.5663 | 1.6080 / 1.5541 / 1.5687 |
| 1200 | 1.5494 | 1.5505 / 1.5381 / 1.5498 |
| 1400 | 1.5466 | 1.5340 / 1.5309 / 1.5430 |

Read that honestly: it looked ahead early, then settled inside the spread of the
yardstick runs and has stayed there, drifting to either side. **No signal in
either direction yet.** One run is not a result — the yardstick's own three runs
finish anywhere between 1.5172 and 1.5223.

One thing does differ. The folding model's memory has grown steadily —
454 → 460 → 466 → 470 → 472 → 474 — while the yardstick runs sit between 405 and
456. That means the folding pathway is doing *something*. Whether it helps is a
different question, and right now it doesn't.

---

## 6. The test that actually decides this

Beating the yardstick would not be enough on its own. The model might be doing
better by luck, or because any 8 extra numbers help.

So the real test is to feed the trained model **deliberately wrong** folding
data and see if it still performs. Three kinds of wrong, each removing something
different:

- **S1** — the folding numbers shuffled so they belong to the wrong parts of the
  chromosome.
- **S2** — the folding numbers slid 10 million letters along, so they are still
  smooth and realistic but pointing at the wrong place. This catches "the model
  just likes having any smooth extra input."
- **S3** — the hardest one. Folding is partly just a function of distance: two
  points close along the string touch more often, everywhere, always. S3 rebuilds
  the folding data keeping only that distance effect. **If the model does just as
  well on S3 as on the real thing, then "structure" was only distance in
  disguise and the central claim is false**, no matter how good the headline
  number looks.

The bar was fixed in advance: the drop from real folding to S1 has to be at
least **0.0050 bits**, with statistical confidence across runs. For scale, that
is about the gap between two yardstick runs.

S1 and S2 are built and tested. S3 still needs building — it can't be made by
shuffling, it needs going back to the raw contact data.

---

## 7. Compute used

Two NVIDIA L40S graphics cards.

Measured speed: **5.30 seconds per training step**. Each run is 2,000 steps, so
about **2.9 hours per run**, using both cards — roughly **5.9 card-hours each**.

| what | runs | card-hours (worked out from the speed above) |
|---|---|---|
| yardstick, first attempt | 3 | ~18 |
| yardstick, redone after the fix | 3 | ~18 |
| folding model | 3, one part-done | ~18 when finished |
| **total when phase 4 ends** | | **~53** |

That is the *useful* compute. Real elapsed time has been much longer, because
the machine shuts down training whenever the browser tab is closed for 10
minutes. This has happened well over a dozen times.

**No work has ever been lost to it.** The training saves itself every 120 steps
(about 11 minutes), and a supervisor script restarts it from the last save. The
worst case is losing 11 minutes.

---

## 8. What is honestly still unknown

- Whether the folding helps at all. Seven checkpoints in, there is no signal.
- Whether the fill-in-the-blank score is even the right thing to measure. A
  model can score identically and still have learned better internal
  representations — and the real biology tests are in Phase 5.
- **This creates a trap that needs deciding now, not later.** If the pass/fail
  test is read on the fill-in-the-blank score, then "folding doesn't help
  fill-in-the-blank" and "folding is doing nothing at all" produce the same
  number — and the rules as written say a fail means stop before Phase 5, which
  is where the actual claim lives. Two honest ways out: measure the test on a
  biology task instead, or write down in advance that a flat score with proven
  reliance still proceeds. Either is fine. Choosing after seeing the numbers is
  not.
- Everything here is one chromosome, one cell type, one species, and a short
  training run. Nothing about scaling can be claimed from it.

---

## 9. Things worth reporting regardless of the outcome

1. **The memory ceiling.** The standard Mamba setting caps memory at exactly
   1,000 letters. Almost everything biologists want to model is bigger than
   that. The fix costs nothing. This is useful to other people whatever happens
   here.
2. **Longer memory did not improve the score.** 30 times more memory, no gain.
   Worth knowing before anyone else spends compute assuming it would.
3. **A clean negative is publishable.** "Building the folding in early works no
   better than bolting it on afterwards" is a real answer to a question people
   currently assume they know the answer to.

---

## 10. Where the real files are

| what | where |
|---|---|
| literature review | `docs/related_work.md` |
| data sources and limitations | `docs/data_card.md` |
| the mechanism design and every decision | `docs/architecture_spec.md` |
| yardstick results | `results/baselines/` |
| the written-up yardstick result | `results/baselines/phase3_report_baseline_v2.txt` |
| folding model results | `results/novel_model/` |
| dated log of every change and why | `project-docs/project.tex` (and `.pdf`) |
| the 27 checks on the folding wiring | `scripts/test_phase4_wiring.py` |

To see what is happening right now: `bash scripts/phase4_status.sh`

If training has stopped, this one command restarts it and picks up from the last
save:

```
cd /home/jupyter-238w1a5447/3d-gen && setsid nohup bash scripts/phase4_guard.sh >> results/novel_model/guard.log 2>&1 < /dev/null &
```
