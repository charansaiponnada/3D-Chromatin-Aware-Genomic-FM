# C2 — cross-cell-line probe design

Written 2026-08-31, **before any model is trained on multi-chromosome data**, so
the design is fixed by measurement rather than chosen after seeing a result.
Every number is from `scripts/c2_differential_power.py` →
`results/c2_differential_power.json`, chr14 at 5 kb, 17,059 jointly usable bins
(79.7% of 21,409).

---

## 1. The plan's premise, tested rather than assumed

`RESEARCH_PLAN_2026-08-26.md` C2 instructs: do **not** probe insulation alone,
because TAD boundaries are largely conserved across cell types (Dixon et al.
2012) and a model could pass by having memorised GM12878. It directs the probe
at A/B compartments, "substantially more cell-type-variable".

That is an assumption about the data. Measured on φ we built ourselves:

| feature | GM–K562 | GM–IMR90 | K562–IMR90 |
|---|---|---|---|
| insulation_100kb | +0.7530 | +0.7675 | +0.7928 |
| insulation_250kb | +0.7900 | +0.8003 | +0.7599 |
| insulation_500kb | +0.8132 | +0.8049 | +0.7955 |
| directionality_2Mb | +0.7924 | +0.8026 | +0.7765 |
| log_contact_density | **+0.4881** | **+0.4799** | **+0.4871** |
| upstream_mass_frac | +0.8340 | +0.8412 | +0.8202 |
| short_long_ratio | +0.7535 | +0.7625 | +0.7953 |
| compartment_pc1 | **+0.8389** | +0.6906 | +0.6158 |

**The premise holds for IMR90 and FAILS for K562.**

- GM12878 vs **IMR90**: compartment +0.6906 < insulation +0.7675 → holds.
  Compartment is the more cell-type-variable feature, as the plan assumed.
- GM12878 vs **K562**: compartment +0.8389 > insulation +0.7530 → **fails.**
  Compartment is *more* conserved than insulation between these two.

The likely reason is lineage: GM12878 (lymphoblastoid) and K562
(erythroleukemia) are both haematopoietic; IMR90 is lung fibroblast. It
reproduces the direction of the independent 4DN-track measurement made for B3
(GM–K562 compartment +0.8250, GM–IMR90 +0.6035).

**Consequence for the design: with K562 as the second line, compartment is the
wrong discriminating feature.** Probing it would test conservation, not
generalisation — the exact failure mode C2 was written to avoid, arrived at from
the other direction.

## 2. This collides with the B3 decision, and the collision is on one feature

| | K562 | IMR90 |
|---|---|---|
| B3 depth gate (`docs/B3_DEPTH_GATE.md`) | **passes** all three | **FAILS** on compartment (0.8815 < 0.90) |
| C2 premise (compartment more variable than insulation) | **fails** | **holds** |

`compartment_pc1` is at the centre of both, in opposite directions. The line
whose compartment track is trustworthy enough to pass the gate is the line whose
compartment is too conserved to make an interesting probe, and vice versa.

**This is new information for B3 and is recorded in
`docs/PI_DECISIONS_2026-08-31.md`.** It does not reopen the gate: IMR90 still
fails it as written.

## 3. What the probe must therefore be

**Primary: the DIFFERENTIAL, not the raw feature.** Predict `φ_A − φ_B` at
matched bins, where A is the pretraining line and B the held-out line. A model
that memorised GM12878 scores zero on this by construction — the target is
exactly what memorisation cannot supply.

Powered, measured on chr14 alone:

| pair | feature | sign disagreement | top-decile differential |
|---|---|---|---|
| GM/K562 | compartment_pc1 | **17.8%** (3,033 bins) | 1,688 bins, 8.44 Mb |
| GM/IMR90 | compartment_pc1 | **20.3%** | 1,685 bins, 8.43 Mb |
| GM/K562 | insulation_100kb | n/a | 1,706 bins, 8.53 Mb |

~8.4 Mb of strongly differential sequence on a single chromosome is enough to
fit a probe. Power is not the constraint here; **feature choice is.**

**Secondary: A/B compartment sign flips** — 17.8% (K562) / 20.3% (IMR90) of
jointly usable bins change compartment sign between lines. A binary,
composition-floor-comparable target, and the cleanest statement of "this region
is A in one cell type and B in the other".

**Report insulation alongside, as a CONTROL, never as the headline.** Its role
is the one it played in B3: if a differential result appears on compartment but
insulation degrades in parallel, the effect is pipeline or depth, not biology.

**Excluded: `log_contact_density`.** It is the least conserved feature across
*every* pair (0.4799–0.4881) including the two haematopoietic lines, which is
the signature of a technical rather than biological difference — sequencing
depth and, for K562, copy number. It is the φ feature most exposed to K562's
aneuploidy and must not carry a cross-cell-line claim.

## 4. Constraints that must travel with the design

- **Cross-chromosome AND cross-cell-line.** Held-out chromosomes (chr14/15 val,
  chr9 test) so this composes with B1 rather than substituting for it. The
  chr9-internal split is retired to pilot status.
- **Composition floor is mandatory.** Probe B's lesson: local GC beat both arms
  on within-window insulation. Every target here gets a k-mer/GC floor, and a
  result below floor is reported as relative, not absolute, sensitivity.
- **Second-line data is currently chr14 only.** K562 and IMR90 were acquired for
  chr14 as the B3 gate chromosome. A cross-chromosome version needs the second
  line on chr15 at minimum, and chr9 for the test split. That is a data cost to
  budget, not something the existing files cover.
- **φ standardisation is per-chromosome** (`phase1_features.py`, data_card §7).
  A differential between two lines on the same chromosome is unaffected by a
  shared per-chromosome shift, but this must be stated: the differential is
  computed on `phi_raw`, not on the standardised `phi`, for exactly that reason.

## 5. Disconfirming outcome, fixed in advance

The structural arm shows no advantage over the matched baseline on the
differential target, at or above the composition floor. That is a clean negative
and will be reported as one: structure-conditioned pretraining did not produce
representations that transfer across cell types.
