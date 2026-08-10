# Architecture Spec — 3D-Chromatin-Aware Genomic FM

**Status: mechanism selected — (a), structural bias in the SSM recurrence (§4.1).** Candidates (b) and (c) are documented but not pursued (§4.2, §4.3); (c) is logged as future work with an explicit empirical trigger (§6).

**Phase 2 gate reminder:** the selection recorded in §4.4 is yours to confirm. Nothing in §4 has been implemented or run — the only thing executed so far is the parameter accounting in §4.1.2.

Companion document: [`docs/related_work.md`](related_work.md). Section references of the form §D1, §H3 point there.

---

## 1. Primary comparator: Evo2HiC, not CHROME

**This supersedes the framing in `related_work.md` §D1.** CHROME remains the comparator for *post-hoc structure integration* and stays in the three-way results table. But it is no longer the work this project's novelty is defined against.

**Why the comparator changed.** CHROME's sequence encoder is trained to convergence with no structure present (§D1, stage 1), so "structure earlier" was a clean differentiator. Evo2HiC (§H3) removes that cleanliness: its second contrastive objective aligns the DNA encoder's 2D embeddings with Hi-C patch embeddings, explicitly to *"transfer structural knowledge, such as chromatin loops and topologically associating domains (TADs), from Hi-C contact maps into the DNA encoder."* Structure shapes a **sequence** encoder during its representation-learning stage. That is the same axis this project claims.

**What Evo2HiC actually is, stated precisely** — because the differentiation below depends on the details, not the summary:
- A frozen Evo 2 (7B) teacher distilled into a **3.6M-parameter, 7-layer CNN** student, trained on 1.2M 2 kb human bins.
- Two **SigLIP-style contrastive** objectives, applied in stages: (1) student ↔ frozen Evo 2 embeddings; (2) student 2D embeddings ↔ Hi-C patch embeddings, positives being same-pixel pairs.
- **No masked or generative objective over nucleotides.** Verified keyword counts over the full preprint text: `masked` 0, `MLM` 0.
- **No SSM.** `Mamba` 0, `state space` 0. CNN throughout.
- **No variant-level evaluation.** `ClinVar` 0, `pathogenic` 0, `eQTL` 0, `variant` 0.

**The one differentiator it kills.** §D1 differentiator #3 — "structure at training time, sequence-only at inference" — is **no longer ours alone**. Evo2HiC's DNA-only encoder needs no Hi-C at inference. Do not assert #3 unqualified against Evo2HiC in any draft. It still holds against CHROME, which requires the contact graph as a GAT input.

---

## 2. Mechanism-by-mechanism: could Evo2HiC's approach get there instead?

The question for each candidate is narrow and adversarial: **could a reviewer say "just run Evo2HiC-style contrastive distillation and you'd get the same effect"?**

The distinction that decides most of these: contrastive distillation is a constraint on the **output embedding space** — it says *these representations should be near those representations*. It cannot express a constraint on **how information propagates between positions inside the model**. Mechanisms that change the computational graph are not reachable by an output-space alignment loss; mechanisms that only change what the embedding encodes are.

| Mechanism | Reachable by Evo2HiC-style distillation? | Nature of the difference |
|---|---|---|
| (a) Structural bias in SSM recurrence | **No** | Architectural — changes information flow, not embedding content |
| (b) Auxiliary contact loss alongside MLM | **Partially — this is the exposed one** | Representational — same injection point, different objective class |
| (c) TAD-conditioned state resets | **No** | Architectural + input-conditional at inference |

### (a) Structural bias term added to the SSM recurrence

**Could distillation achieve the same effect? No.**

Evo2HiC's Hi-C signal acts on 2 kb-bin embeddings at the encoder's *output*, pulling them toward Hi-C patch embeddings. It never touches how the encoder mixes information across positions — the student is a fixed-receptive-field CNN whose propagation structure is identical with or without the structural loss.

A structural bias inside the recurrence modifies the state transition itself: how much of position *i*'s state reaches position *j* becomes a function of the structural prior. That is a change to the computational graph, and no output-space alignment loss induces it. You could distill a CNN to *imitate the embeddings* of a structurally-biased SSM, but that is downstream of having built one — it is not an alternative route to the mechanism.

**Why our version is still a distinct contribution.** The claim is not "structure improves embeddings" (Evo2HiC has that) but "structure belongs in the propagation operator of a long-range sequence model." The scientific content is whether a structurally-modulated recurrence generalizes differently — particularly at distances beyond the training contact range, where an embedding-space constraint has nothing to say but a modified recurrence still acts.

**Honest exposure.** A reviewer can concede the mechanism differs and still ask whether the *downstream benefit* differs. That is empirical, and the ablation has to answer it: an Evo2HiC-style distillation arm at matched parameters is the comparison that makes (a) defensible. Without it, "architecturally distinct" is an assertion about code, not a result.

### (b) Auxiliary contact-prediction loss alongside masked-token prediction

**Could distillation achieve the same effect? Partially — and this is the mechanism most exposed to the objection.**

Both inject Hi-C into the sequence encoder's training signal at the same point. A reviewer's version of the objection is fair on its face: contrastive alignment to Hi-C patch embeddings and predictive reconstruction of contacts are both "make the sequence representation carry structural information."

Three things keep it distinct, in descending order of strength:

1. **Evo2HiC has no self-supervised objective for the auxiliary loss to interact with.** `masked` 0, `MLM` 0 — there is no nucleotide-level learning signal in their model at all. The contribution in (b) is not the contact loss; it is the **interaction** between masked-token prediction and contact prediction — whether structural supervision changes *what sequence features MLM learns*. Evo2HiC cannot exhibit that interaction because one of the two terms is absent. **State the contribution as the multi-task interaction, never as the contact loss alone.**
2. **A distilled student's ceiling is its teacher.** Evo2HiC's student can only recover features already present in frozen Evo 2 embeddings, filtered toward Hi-C relevance. A from-scratch model has no teacher ceiling and can in principle learn structure-relevant features Evo 2 never encoded. Testable: compare against an Evo 2–distilled encoder on cases where Evo 2's representation is known to be weak.
3. **Predictive vs. relational supervision.** A contact-prediction head reconstructs contact values; SigLIP contrastive alignment only enforces same-pixel/different-pixel relational structure. The predictive target is strictly more informative — but this is the weakest of the three arguments and should not carry weight on its own.

**Also note** this mechanism's *other* prior-art exposure, from §D3: the auxiliary objective is close to Akita's training objective used as a head. Combined with the Evo2HiC exposure, (b) is doubly encumbered on mechanism and rests almost entirely on the transfer result (§3 below). That is a defensible position, but it is a narrower one than (a) or (c).

### (c) TAD/loop-membership-conditioned attention masking or state resets

**Could distillation achieve the same effect? No — the least reachable of the three.**

A state reset is a discrete, structure-conditional intervention: at a TAD boundary the recurrent state is cleared or gated, changing the model's effective receptive field per position. Contrastive distillation has no mechanism for a discrete, position-dependent intervention on propagation. Evo2HiC's structural signal is continuous, global, and applied to outputs.

This is also the mechanism that most directly operationalizes the biology in §A2 and §A3 — TAD insulation as a real constraint on which enhancers reach which promoters — rather than treating Hi-C as a generic similarity signal.

**Design consequence you need to decide before implementing (c).** State resets require TAD boundary information *at inference*. If those come from called TADs, this mechanism **loses the sequence-only-at-inference property**, and Evo2HiC's DNA-only encoder becomes strictly better on that axis. To stay competitive, (c) needs a **learned boundary predictor** trained during pretraining and used at inference in place of ground-truth calls. That is an additional component with its own failure mode — if the boundary head is inaccurate, resets fire in the wrong places and the mechanism actively hurts. Budget for it, and treat "reset positions from learned head vs. from ground-truth calls" as a required ablation.

---

## 3. What survives regardless of which mechanism is chosen

Two claims are untouched by Evo2HiC and should carry the paper's contribution independent of the mechanism decision:

1. **The transfer evaluation.** No sequence→structure model — not Akita, not Orca, not Evo2HiC — has ever had its representations evaluated on variant or regulatory tasks (§H1, §H3). Evo2HiC's entire eval surface is contact maps, epigenomic tracks, motifs, and resolution enhancement. **The transfer claim is the contribution that no current work contests.** It is also the one that requires no architectural novelty at all to be publishable — which makes it the safe core of the paper.
2. **SSM + structure is unoccupied.** `Mamba` 0 and `state space` 0 in Evo2HiC; CHROME is CNN/MLP + GAT; Akita and Orca are CNN. Mechanisms (a) and (c) have no counterpart anywhere in the reviewed literature.

Corollary for the results table: the comparison arms should now be **sequence-only baseline** / **CHROME-style post-hoc** / **Evo2HiC-style contrastive distillation** / **our mechanism**. The distillation arm is what converts §2's "architecturally distinct" arguments from assertions into results.

---

## 4. Mechanism design

### 4.1 Mechanism (a) — structural bias in the SSM recurrence — **SELECTED**

**Dependency.** This spec assumes Phase 1 delivers, per genomic bin *b*, a raw structural feature vector φ_b ∈ ℝ^8. Working definition: insulation score at three window sizes, directionality index, log contact density, upstream and downstream normalized contact mass, and the compartment (A/B) eigenvector — all computed on distance-corrected (observed/expected) contact matrices. **The exact composition of φ is Phase 1's call, not this document's**; §4.1.1 only requires that φ exist per bin and be standardized genome-wide.

#### 4.1.0 Design constraint: reverse-complement equivariance

Settle this before writing code, because it changes the parameter count and the tying scheme.

Caduceus offers two variants: **PH** (RC data augmentation, no equivariance guarantee) and **PS** (RC-equivariant by construction). Under reverse-complement, bin order reverses, and the components of φ do **not** all transform the same way: insulation, contact density and compartment eigenvector are **symmetric** (invariant under reversal), while the directionality index and the upstream/downstream mass pair are **antisymmetric** (they swap or flip sign).

Consequently:
- **If building on PS**, the structural pathway must respect the equivariance construction or it silently breaks the model's guarantee. Required tying: `W_Δs^rev = W_Δs^fwd · diag(σ₁…σ_{d_s})` where σ_k = +1 for symmetric coordinates and −1 for antisymmetric ones. This *reduces* parameter count relative to §4.1.2.
- **If building on PH**, the two directions may hold independent parameters, which is what §4.1.2 counts.

**Recommendation: start on PH.** RC equivariance is orthogonal to the hypothesis under test, and coupling the two makes a null result uninterpretable — you would not know whether the mechanism failed or the equivariance tying starved it. §4.1.2 therefore reports the PH (untied) case, which is the parameter **upper bound**.

#### 4.1.1 Mathematical formulation

**Baseline selective SSM.** Per layer, per direction, with channel index *i* ∈ [1, d_inner] and state index *n* ∈ [1, N]. Layer input u_t ∈ ℝ^{d_model}:

```
(z_t, v_t)          = split(in_proj(u_t))                    ∈ ℝ^{d_inner} × ℝ^{d_inner}
x_t                 = SiLU(DepthwiseConv1d(v)_t)             ∈ ℝ^{d_inner}
(δ'_t, B_t, C_t)    = split(x_proj(x_t))                     ∈ ℝ^{dt_rank} × ℝ^N × ℝ^N
Δ_t                 = softplus(W_dt δ'_t + b_dt)             ∈ ℝ_{>0}^{d_inner}
A[i,n]              = −exp(A_log[i,n])                       < 0
Ā_t[i,n]            = exp(Δ_t[i] · A[i,n])                   ∈ (0,1)
B̄_t[i,n]            = Δ_t[i] · B_t[n]
h_t[i,n]            = Ā_t[i,n] · h_{t−1}[i,n] + B̄_t[i,n] · x_t[i]
y_t[i]              = Σ_n C_t[n] · h_t[i,n] + D[i] · x_t[i]
out_t               = out_proj(y_t ⊙ SiLU(z_t))
```

**Structural context.** Let E_θ : ℝ^8 → ℝ^{d_s} be a shared two-layer MLP (d_s = 2), and b(t) the bin containing token t:

```
s_t = E_θ(φ_{b(t)}),     with φ standardized genome-wide to zero mean, unit variance per coordinate
```

Standardization is not cosmetic — it is the primary defence against failure mode **F1**.

**The two modifications.**

*(i) Per-channel timescale bias.* An additive structural term in pre-softplus space:

```
Δ_t = softplus( W_dt δ'_t + b_dt + W_Δs s_t ),      W_Δs ∈ ℝ^{d_inner × d_s}, no bias
```

*(ii) Scalar permeability penalty.* A non-negative, structure-dependent additive damping of the log-decay:

```
p_t      = softplus( w_g^⊤ s_t + b_g )     ≥ 0,     w_g ∈ ℝ^{d_s}, b_g ∈ ℝ
Ā_t[i,n] = exp( Δ_t[i] · A[i,n] − p_t )
```

B̄, C, D, h and y keep their baseline form, with Δ_t taken from (i).

**Initialization.** `W_Δs = 0`, `w_g = 0`, `b_g = −8`. Then p_t = softplus(−8) ≈ 3.4×10⁻⁴ and the model is numerically indistinguishable from the baseline at step 0, so any divergence during training is attributable to learned structural signal rather than an initialization shift. Zero-init on W_Δs does **not** block learning: ∂L/∂W_Δs = (∂L/∂Δ_pre) s_t^⊤, which is non-zero whenever s_t ≠ 0.

**Text diagram (one layer, one direction).**

```
                    φ_{b(t)} ──► E_θ ──► s_t ─┬──► W_Δs ──┐
                                              │           │  (per-channel, ℝ^{d_inner})
                                              │           ▼
   u_t ──► in_proj ──► conv ──► SiLU ──► x_proj ──► W_dt ─(+)──► softplus ──► Δ_t
                                  │                                            │
                                  │                                            ▼
                                  │            A ──────────────────────► Δ_t·A ─(−)──► exp ──► Ā_t
                                  │                                            ▲
                                  └──► B_t, C_t                    p_t ─────────┘
                                                                    ▲   (scalar, all channels)
                                              s_t ──► w_g, b_g ─────┘

   h_t = Ā_t ⊙ h_{t−1} + Δ_t B_t x_t          ← structure enters BOTH retention and drive
   y_t = C_t·h_t + D x_t
```

**Why this is not "just concatenating structural features to the input" — the crux of the novelty argument.** Appending s_t to x_t would add only to the drive term B̄_t x_t. It cannot alter Ā_t, which governs *retention*. Here Δ_t enters Ā_t inside an exponential, so the structural signal sets the state's **effective memory horizon**: for channel *i*, the time constant is

```
τ_t[i] = −1 / ( Δ_t[i]·A[i,n] − p_t )      (in tokens)
```

Structure therefore controls *how far information propagates*, not merely *what is encoded*. This is precisely the class of constraint that §2 argued contrastive distillation cannot express, and it is the sentence the Method section should be built around.

**Relationship to mechanism (c).** As p_t → ∞, Ā_t → 0 and h_t = B̄_t x_t — a hard state reset. Mechanism (a) is thus the **soft, continuous, learned relaxation** of mechanism (c), and (c) is its discrete limit. See §6.

#### 4.1.2 Parameter accounting — computed, not cited

Computed by instantiating both backbones as `nn.Module`s and summing parameter tensors: [`scripts/param_accounting.py`](../scripts/param_accounting.py). Reproduce with:

```bash
python scripts/param_accounting.py
```

Config: `d_model=256, n_layer=16, d_state=16, d_conv=4, expand=2 → d_inner=512, dt_rank=16, vocab=16, d_s=2`. BiMamba shares `in_proj`/`out_proj` across directions; conv, `x_proj`, `dt_proj`, `A_log`, `D` are per-direction.

| | Parameters |
|---|---:|
| Baseline, per layer — `in_proj` | 262,144 |
| — `out_proj` | 131,072 |
| — `fwd` inner | 44,544 |
| — `rev` inner | 44,544 |
| — RMSNorm | 256 |
| **Baseline layer total** | **482,560** |
| Structural add, per direction — `W_Δs` | 1,024 |
| — `w_g` (+ `b_g`) | 3 |
| **Structural add, per layer (both directions)** | **2,054** |
| Structural encoder E_θ (shared, once) | 178 |

| Model | Total |
|---|---:|
| Baseline | **7,725,312** |
| Structural-bias | **7,758,354** |
| Added | 33,042 |
| **Delta** | **+0.4277%** |

**Matched-parameter constraint (≤5%): PASS**, with ~11× headroom. The margin means d_s can grow to ~8–16, or a small per-layer structural encoder can replace the shared one, without breaching the constraint — but **re-run the script rather than extrapolating** if the config changes.

Two notes. First, this supersedes the `[UNVERIFIED]` Caduceus parameter figure flagged in `related_work.md` §C2 and §G *for our own purposes* — the constraint is now satisfied against a model we instantiate, so no citation is load-bearing. It does **not** verify Caduceus's published counts, which still need checking if the paper cites them. Second, if §4.1.0 resolves toward PS, the tied `W_Δs` reduces the added count; the constraint passes a fortiori.

#### 4.1.3 Shuffled-structure sanity check (Phase 4 gate)

Per `CLAUDE.md`, Phase 4 does not exit until a shuffled-structure ablation shows a measurable performance drop. Specified concretely:

**What gets shuffled — four controls.** All operate on φ *before* E_θ, within the training split, leaving sequence untouched:

| ID | Control | What it destroys | What it preserves | Rules out |
|---|---|---|---|---|
| **S0** | `ZERO` — s_t := 0 ∀t | all structural signal | — | establishes the sequence-only floor at *identical* parameter count |
| **S1** | `GLOBAL-PERM` — permute φ_b uniformly at random across all bins genome-wide | sequence↔structure correspondence **and** local autocorrelation | exact marginal distribution of φ | — (primary reliance probe) |
| **S2** | `CIRCULAR-SHIFT` — per chromosome, circularly shift φ by ≫ max memory horizon (≥10 Mb) | alignment only | marginal **and** local autocorrelation | "the model just benefits from any smooth, slowly-varying auxiliary channel" |
| **S3** | `DISTANCE-MATCHED REWIRE` — resample contacts preserving the distance-decay curve P(s), then recompute φ | locus-specific structure | the 1D-distance-explainable component | **"structure is just genomic distance in disguise"** |

**S3 is the control that matters most.** Per `related_work.md` §A1, contact probability decays smoothly with genomic distance under the fractal-globule model. If S3 performs like the real-structure run, the mechanism has learned a re-parameterized positional prior, not a structural one — and the paper's central claim is false regardless of how good the headline numbers look. It mirrors CHROME's degree- and distance-matched controls (§D1), so it is also the control a reviewer familiar with that paper will expect.

**Two protocols, measuring different things.**

- **P1 — swap at inference (cheap; the gate).** Take the real-structure-pretrained model, feed S0–S3 at evaluation only. Measures **reliance**: does the trained model's behaviour depend on the structural input?
- **P2 — shuffle at pretraining (expensive; confirmatory).** Pretrain separate models under S1 and S3 at identical seeds, data order, and compute. Measures whether structure **helped learning**. This is the actual scientific claim; P1 alone cannot establish it, because a model could rely on a signal at inference that conferred no benefit during pretraining.

**Metrics.** Primary: masked-token validation loss in bits/nucleotide on the held-out chromosome. Secondary: the Phase 5 downstream suite. ≥3 seeds per configuration, per the Phase 5 gate.

**Decision rule.** For a lower-is-better metric, let Δ_S = metric(S) − metric(real), and σ_real = across-seed standard deviation of the real-structure runs.

> **The gate passes iff, on P1 with control S1:** Δ_S1 > 0 with a 95% bootstrap CI over seeds excluding 0, **and** Δ_S1 ≥ 2·σ_real.

A statistically significant but tiny effect is not sufficient — the 2σ_real floor is what stops a p-value from standing in for an effect size.

**Predicted ordering, as a falsifiable ladder:**

```
metric(real)  <  metric(S0)  ≤  metric(S3)  ≤  metric(S1) ≈ metric(S2)
```

| Observation | Interpretation | Action |
|---|---|---|
| real < S0 | structure beats no structure | necessary, not sufficient |
| S3 ≈ real | "structure" is just distance decay | **stop** — reframe or abandon the claim |
| S3 > real, and S3 < S1 | locus-specific structure carries signal beyond distance | the result you want |
| S1, S2 > S0 | wrong structure actively misleads → strong reliance | strongest evidence |
| S1 ≈ S0 | model learned to gate out uninformative structure | acceptable; weaker |
| S1 ≈ real | **mechanism is inert** | **do not proceed to Phase 5** |

**Cheap mechanistic diagnostics — run continuously during pretraining, not just at the gate.** These catch a dead mechanism in hours rather than after a full run:

- **D1** — ratio Var_t(W_Δs s_t) / Var_t(W_dt δ'_t), per layer. **< 0.05 ⇒ pathway inert** (F1/F8).
- **D2** — histogram of p_t. Mass concentrated at ~0 ⇒ gate unused (F5). Bimodal ⇒ see §6.
- **D3** — ‖W_Δs‖_F trajectory. Flat near 0 ⇒ no gradient pressure (F6).
- **D4** — probe R² predicting s_t from the *baseline* model's hidden states. High R² ⇒ redundancy (F3), which is a finding, not a bug.

#### 4.1.4 Failure modes — how this reduces to the baseline in practice

The question this section answers: the mechanism is architecturally distinct on paper, so what would make it behave identically to sequence-only in a real run?

**F1 — Bias absorption into b_dt.** *(highest-probability failure)* If s_t is near-constant across positions — because φ is poorly normalized, or E_θ collapses — then W_Δs s_t is a constant vector, absorbed exactly into the existing b_dt. The model is then the baseline with a re-parameterized bias, and will train to identical loss. **Detect:** D1. **Mitigate:** genome-wide standardization of φ (§4.1.1); optionally a variance floor penalty on s_t across a batch.

**F2 — Softplus saturation.** If pre-softplus activations sit far positive, softplus is locally linear and the structural term is a small relative perturbation of Δ; if far negative, softplus ≈ 0 and its gradient vanishes, so W_Δs receives no learning signal. **Detect:** distribution of `W_dt δ'_t + b_dt` per layer. **Mitigate:** retain Mamba's standard dt-bias initialization, which targets a usable Δ range; do not re-tune it away.

**F3 — Structure is predictable from sequence.** ⚠️ **Substantially defused by external evidence, 2026-08-07.** CTCF motifs and CpG content substantially determine boundaries — the premise of Akita and Orca — so a model with enough capacity might infer s_t from sequence alone, driving gradient pressure on W_Δs to zero. **Detect:** D4.

However, Lee (arXiv:2604.07196; `related_work.md` §I3) probed **Evo2-7B** directly and found it **does not** encode higher-order 3D organisation: TAD boundary deletions and CTCF motif inversions/deletions were penalised *less* than GC- and size-matched random controls (boundary deletions paired Wilcoxon p = 0.405), and generated sequences failed to produce convergent loops (median enrichment 0.054 vs reference 0.388). Their conclusion: Evo2 *"has learned local CTCF grammar but misses higher-order 3D organization."*

If a 7B model with 1 M context trained on 9.3T tokens cannot infer TAD/loop-scale structure from sequence, a 7.7 M-parameter model will not either — so the structural input is carrying information the sequence pathway does not already hold. **F3 remains live for *local* structure** (they find local CTCF grammar *is* learned), but not at the scale mechanism (a) targets.

**This does not remove the diagnostic.** Still run D4, and still report a redundancy result honestly if it appears — per `CLAUDE.md` a negative result is publishable. It does mean F3 should no longer be treated as the most likely explanation for an inert mechanism; F1, F4 and F7 are.

**F4 — Resolution mismatch.** ⚠️ **MEASURED, 2026-08-07 — the precondition is present at initialization.** See [`scripts/f4_memory_horizon.py`](../scripts/f4_memory_horizon.py).

At Mamba's standard initialization for our config (d_model=256, d_inner=512, d_state=16, Δ ∈ [10⁻³, 10⁻¹], A = −(1…16)):

| quantity | value |
|---|---|
| τ median | 14.9 tokens (0.015 kb) |
| τ p90 | 108.8 tokens |
| τ p99 | 497.4 tokens |
| **τ max (slowest channel)** | **994 tokens (0.99 kb)** |
| (channel, state) pairs with τ ≥ 5,000 tokens | **0.0000** |

**No channel's memory horizon reaches even one 5 kb bin.** τ_max/bin = 0.199. Depth composition helps — 16 × τ_max ≈ 15,900 tokens ≈ 3.2 bins — but that is a relay heuristic, not a bound.

**Two consequences, and the second is bigger than F4.**

1. *For the mechanism:* φ assigned as a step function per 5 kb bin is constant across everything any single layer's state can see. **Mitigation now mandatory, not optional: interpolate φ to token resolution** (linear between bin centres) so s_t varies continuously at every position rather than jumping every 5,000 tokens. This is cheap and should be done regardless of anything else.
2. *For the project:* if τ tops out near 1 kb per layer and ~16 kb across the stack, **a d_model=256 backbone may not represent TAD-scale (100 kb–1 Mb) dependencies at all** — and that limitation applies to the **baseline** just as much as to the structural arm. Conditioning on TAD structure is only meaningful if the architecture can express dependencies at TAD scale in the first place.

**This is a risk flag, not a proof of failure.** Δ is learned and selectivity can drive it far below its initialization range; Caduceus's reported long-range VEP results suggest such models do capture *something* long-range. But the mechanism starts in the degenerate regime and has to climb out.

**New gate — Phase 3 must measure this, and it blocks Phase 4.** Add to the baseline run's logged output the distribution of learned Δ and the implied τ per layer. Then:

| Trained baseline τ | Reading | Action |
|---|---|---|
| τ reaches ≳100 kb in upper layers | architecture can express TAD scale | proceed to Phase 4 as specified |
| τ saturates ≪ TAD scale | structure-at-TAD-scale is not expressible at this size | re-scope to sub-TAD structural signal, or raise d_model/d_state, **before** spending Phase 4 compute |

**Do not skip this.** It is a cheap measurement on a run that Phase 3 requires anyway, and it can invalidate the mechanism's premise before the expensive phase begins.

**Other mitigations if needed:** the pilot `.mcool` already carries 1 kb and 2 kb resolution levels (τ_max/bin = 0.994 and 0.497 respectively), so changing resolution needs no new download — but it costs comparability with CHROME's 5 kb and buys noisier contacts. Treat resolution as an ablation axis, not a default change.

**F5 — Permeability gate never leaves zero.** If p_t stays ≈ 0, Ā is unchanged and modification (ii) contributes nothing; only (i) is live. Not fatal on its own, but it must not be reported as if both terms were active. **Detect:** D2.

**F6 — Gradient starvation.** The structural pathway is 0.43% of parameters and sits behind a softplus and an exponential. The MLM gradient may simply dominate. **Detect:** D3, plus gradient-norm ratio of structural to non-structural parameters. **Mitigate:** a separate, higher learning rate for the structural pathway — but declare it, since it is a tuning advantage the baseline does not receive and a reviewer will ask.

**F7 — Direction/RC sign conflict.** In BiMamba both directions consume the same s_t, but the antisymmetric coordinates of φ (directionality index, upstream/downstream mass) mean the correct response has *opposite sign* in the forward and reverse passes. With shared or symmetrically-initialized parameters the two contributions can cancel, yielding an apparently inert mechanism that is in fact two live pathways destructively interfering. **Detect:** compare sign and magnitude of W_Δs^fwd vs W_Δs^rev on antisymmetric coordinates; ablate to symmetric-only φ and see whether the effect returns. **Mitigate:** independent per-direction parameters (already assumed in §4.1.2), or the explicit sign-tying of §4.1.0.

**F8 — Structural encoder collapse.** E_θ maps all inputs to a constant or to a one-dimensional subspace, discarding most of φ. Reduces to F1. **Detect:** rank/singular values of s_t over a large batch. **Mitigate:** d_s is small (2) by design, which limits how much there is to collapse; consider bypassing E_θ entirely and feeding two standardized raw coordinates as an ablation.

> **Standing instruction:** F1, F4 and F7 all produce the *same* symptom — a mechanism that trains to baseline loss with no error. Do not conclude "structure doesn't help" until D1–D4 have distinguished which of them is responsible. That distinction is the difference between a real negative result (F3) and a bug (F1/F4/F7).

---

### 4.2 Mechanism (b) — auxiliary contact-prediction loss — documented, not pursued

Not selected because it is **doubly encumbered on mechanism and therefore rests almost entirely on the transfer result**. Per `related_work.md` §D3 the objective is close to Akita's training target relocated to an auxiliary head, and per §2 above it shares an injection point with Evo2HiC's structure-distillation stage — so the defensible claim reduces to the MLM×contact-loss *interaction*, a narrower and harder-to-demonstrate contribution than (a)'s architectural one. It also does nothing that (a) does not: both put Hi-C into the pretraining signal, but only (a) changes information propagation, which is the part no competitor can reach by distillation. **It is not discarded** — (b) remains the natural third comparison arm alongside the Evo2HiC-style distillation arm (§3), and if (a) fails via F3 (structure redundant with sequence), (b) is the cheapest way to test whether a *predictive* rather than *modulatory* use of the same signal fares better.

### 4.3 Mechanism (c) — TAD-conditioned state resets — documented, not pursued

Not selected because **(a) already contains it as a limiting case**: p_t → ∞ gives Ā_t → 0, a hard reset (§4.1.1). Mechanism (a) therefore tests the same underlying hypothesis — that TAD boundaries should constrain information propagation — with strictly fewer moving parts, and lets the model *learn* where and how sharply to insulate rather than committing to called boundaries. (c) additionally carries the inference-time liability flagged in §2: hard resets need boundary positions at inference, so either we ship a dependency on called TADs (losing the sequence-only-at-inference property that Evo2HiC's DNA-only encoder has) or we build and validate a learned boundary predictor, which is an extra component with its own failure mode — an inaccurate head fires resets in the wrong places and actively degrades the model. Deferred to §6 with an explicit trigger.

### 4.4 Selection rationale

(a) is selected on four grounds, in descending order of weight:

1. **It is the mechanism no competitor can reach by their existing method.** §2 established that contrastive distillation constrains the output embedding space and cannot express a constraint on inter-position propagation. (a) is squarely on the unreachable side; (b) is not.
2. **It subsumes (c).** The soft relaxation tests the same hypothesis with fewer components and no inference-time structural dependency, and its learned p_t distribution *tells us empirically* whether the hard version is worth building (§6).
3. **Cost.** +0.43% parameters, one extra ℝ^{d_inner×2} matrix and a scalar gate per direction per layer. It fits the 2× L40S budget with the comparison arms intact.
4. **It has a clean falsification path.** §4.1.3's S3 control can kill the claim outright, and D1–D4 can diagnose a dead mechanism within hours. A mechanism that is cheap to disprove is the right one to try first.

**Against it, recorded honestly:** (a) is the mechanism most vulnerable to silent inertness. Three distinct failure modes (F1, F4, F7) all present as "trains to baseline loss, no error raised." The diagnostics in §4.1.3 exist specifically because of this, and they are not optional.

---

## 4.2bis Implementation findings — 2026-08-07

Three things surfaced when the mechanism was actually built ([`src/chromfm/model.py`](../src/chromfm/model.py), tested by [`scripts/test_model.py`](../scripts/test_model.py), 16/16 checks passing). All three change Phase 3 or Phase 4 planning, and none was visible from the spec alone.

### The permeability term does not fit the stock Mamba kernel

`mamba_ssm`'s `selective_scan_fn` takes `delta` as an argument, so **modification (i), the Δ bias, is free** — add `W_Δs s_t` before the call and the fused kernel runs unmodified.

**Modification (ii), p_t, is not expressible through that interface.** It needs `Ā = exp(ΔA − p)`, and folding p into Δ would require `Δ' = Δ + log(g)/A_n`, which depends on the state index *n*. A single per-channel Δ cannot produce a decay term uniform across n.

The reference scan in `model.py` handles both but is sequential in time, so it is unusable at L = 32,768 (that would be 32,768 × 16 layers × 2 directions Python iterations per forward pass).

**This forces a decision before Phase 3, not after.** Whatever scan the structural arm uses, the baseline must use the same one, or matched-compute is confounded. Three options:

| Option | Cost |
|---|---|
| Drop p, keep only the Δ bias | Stock kernel, full speed. Loses the soft-reset semantics and the formal link to mechanism (c) in §6. |
| Write a Triton scan carrying p | Keeps the mechanism whole. Costs implementation and correctness-testing time, and both arms must then run on it. |
| Chunked pure-PyTorch scan for both arms | Fair and portable, but slower for the baseline too, which eats the compute budget twice. |

Unresolved. It is the first thing Phase 3 has to settle.

### The structural encoder is gradient-bottlenecked

`∂L/∂s = W_Δs^⊤ · ∂L/∂Δ_pre`, and `W_Δs` is zero-initialised, so the encoder E_θ receives **exactly zero gradient on step 0**. It unblocks once `W_Δs` moves, but measured magnitudes after one step are stark: `|∂L/∂W_Δs| ≈ 7×10⁻⁴` against `|∂L/∂E_θ| ≈ 3×10⁻⁹`. The encoder trains roughly five orders of magnitude slower than the matrix in front of it.

**Recommendation: delete E_θ and feed the eight standardised φ features directly, with d_struct = 8.** Computed by instantiation:

| Configuration | Total | Added | Delta |
|---|---:|---:|---:|
| d_struct = 2, with encoder (current spec) | 7,758,354 | 33,042 | +0.428% |
| d_struct = 8, with encoder | 7,856,952 | 131,640 | +1.704% |
| **d_struct = 8, no encoder** | **7,856,672** | **131,360** | **+1.700%** |

Still comfortably inside the 5% budget. Removing E_θ also eliminates failure mode **F8** (encoder collapse) outright and removes a component whose only job is compressing eight already-standardised numbers into two.

**Not applied unilaterally**, because it changes parameter figures recorded in `data_card.md`, the primer and the PDF. It is a one-line config change when approved.

### The permeability gate's initialisation was throttling its own gradient

`b_g = −8` gives `p = 3.4×10⁻⁴`, which is the near-exact baseline equivalence §4.1.1 wanted. But the gradient through softplus is `σ(b_g)`, so `b_g = −8` also attenuates the gate's learning signal by a factor of ~3000 — **failure mode F2, introduced by our own initialisation choice**.

Changed to `b_g = −4`: p = 0.018, still negligible against the decay term, with 53× more gradient. Measured effect on the gate's gradient in the unit test: 2.06×10⁻⁷ → 7.69×10⁻⁶.

### What the tests now guarantee

Parameter counts match `param_accounting.py` exactly; the structural model is numerically identical to the baseline at initialisation (max gap 0.0); different φ produces different output; gradient reaches `W_Δs` and the gate despite zero-init; masked positions stay finite; the reverse-complement sign flip changes the reverse pass; and `tau_stats()` reproduces the F4 measurement (τ_max ≈ 1000 tokens, median 14.5).

One bug the tests caught that inspection had missed: the reverse-complement sign flip was being applied to the **encoded** s rather than to raw φ. The encoder mixes all eight coordinates, so there is no per-feature correspondence left to flip afterwards — the operation was meaningless. It now happens on φ before encoding, at the cost of a second encoder pass.

A second bug: `A` was broadcast against `(b, d, l, 1)` by trailing-axis alignment, which silently matches `n` against `l` and only agrees when `l == d_inner`. The first test passed because it happened to use L = 64 with d_inner = 64.

---

## 5. Standing check — Evo2HiC is a live target

**Evo2HiC is an active bioRxiv preprint (10.1101/2025.11.18.689171, posted Nov 2025) from the Noble Lab at UW — the same group behind HiCFoundation, with H100-class compute and an established Hi-C modeling program.** It is not a settled citation. It is a moving comparator.

**The specific risk.** Their third stated limitation is, verbatim: *"in the current work we extracted embeddings from Evo 2 by freezing its parameters, as the model's size makes fine-tuning computationally challenging. We plan to develop efficient fine-tuning strategies that enable adapting Evo 2 with Hi-C data, potentially unlocking its full capacity for chromatin structure analysis."*

They have publicly stated the intent to adapt a sequence foundation model with Hi-C, and named compute — not ideas — as what stopped them. They are better resourced than this project on exactly that axis.

**Two distinct failure scenarios, with different consequences:**

| If a v2 / journal version adds… | Consequence | Response |
|---|---|---|
| **Variant evaluation** (ClinVar / eQTL / regulatory transfer) | **Severe.** §3's core contribution — the unclaimed transfer result — closes. The paper would rest entirely on mechanism novelty. | Re-scope to mechanism (a) or (c) with the transfer result demoted to confirmation, or pivot the framing to matched-compute small-model analysis. |
| **Unfrozen / fine-tuned Evo 2 with Hi-C** | **Moderate.** Establishes "structure adapts the sequence FM" at scale, but still by distillation/fine-tuning, not by structure inside a self-supervised objective, and still with no SSM. | Sharpen the framing to the objective class and the recurrence, and lean harder on matched-compute fairness. |

**Re-search protocol — run before Phase 4 commits compute, and again before the Phase 6 draft:**

- [ ] Check bioRxiv 10.1101/2025.11.18.689171 for a **v2 or later revision**; diff the evaluation section against the v1 keyword baseline recorded in §H3 (`ClinVar` 0, `pathogenic` 0, `eQTL` 0, `variant` 0, `masked` 0, `MLM` 0, `Mamba` 0, `state space` 0).
- [ ] Check for a **journal version** — Noble Lab published HiCFoundation in *Nature Methods*, so expect a high-profile venue and an expanded evaluation on the way.
- [ ] Forward-citation sweep on Evo2HiC and on HiCFoundation for follow-ups from the same group.
- [ ] Re-run the general sweep for **SSM/Mamba genomic LM + Hi-C**. As of 2026-08-07 this combination was unoccupied; that is the assumption mechanisms (a) and (c) rest on, and it has a shelf life.

> Use the browser, not WebFetch — bioRxiv returns 403 to automated fetches but loads normally in the browser pane.

**Gate.** Do not spend Phase 4 compute on a mechanism whose novelty argument has not been re-checked against the current version of this preprint within the preceding two weeks.

---

---

## 6. Future work — mechanism (c), hard TAD-conditioned state resets

Logged explicitly so it is not lost, and so the decision to revisit it is driven by evidence rather than by whichever mechanism happens to be in front of us later.

**What it is.** The discrete limit of §4.1's permeability term: instead of a learned continuous penalty p_t, the recurrence is hard-reset at TAD boundaries — h_t := B̄_t x_t, discarding carried state — with boundaries taken from called TADs or from a learned boundary head.

**Why it is deferred rather than rejected.** It encodes the biology of `related_work.md` §A2–§A3 more literally than (a) does: TAD boundaries are genuine discrete constraints on which enhancers reach which promoters, not soft preferences. If the true function is closer to a step than a slope, (a) will approximate it imperfectly and (c) will fit it better. That is an empirical question, and (a) answers it as a side effect.

**Empirical trigger — revisit (c) if any of these hold after Phase 4:**

1. **The learned p_t distribution is strongly bimodal** (diagnostic D2), with mass near 0 and mass at large values. That is the model asking for hard resets through a soft parameterization, and it is the single most direct evidence for (c).
2. **p_t saturates at its upper range** at a substantial fraction of positions, and those positions align with called TAD boundaries above chance. Same signal, different symptom.
3. **(a) passes the §4.1.3 gate but the effect size is small**, while D2 shows the gate is doing most of the work relative to the Δ-bias term — i.e. the useful part of (a) is the part (c) sharpens.

**Conversely, do not revisit (c) if** p_t stays unimodal and near zero (F5) — that is the model declining to insulate at all, and a hard version of a term the model does not want will not help.

**What building it would require, beyond (a):**
- A learned boundary predictor, trained jointly during pretraining, so inference does not depend on called TADs. Without this, (c) forfeits the sequence-only-at-inference property and loses on that axis to Evo2HiC's DNA-only encoder (§2).
- A gradient path through a discrete decision — straight-through estimator, or a temperature-annealed relaxation of §4.1's p_t, which is the natural continuous→discrete bridge and reuses (a)'s implementation.
- An ablation separating "resets at learned boundaries" from "resets at ground-truth boundaries", to establish how much of any gain is the mechanism versus the quality of the boundary calls.

**Sequencing.** (c) is a Phase 4-extension or follow-up-paper item, not a parallel track. Running it concurrently would split compute across two mechanisms and leave neither with the ≥3 seeds Phase 5 requires.

---

---

## 7. Decisions of record

Taken 2026-08-07 under explicit delegation from the PI ("you take everything"). **Each is reversible and each is the PI's to override** — they are recorded here so the reasoning is inspectable rather than buried in a chat log.

| # | Decision | Choice | Reasoning | Reversal cost |
|---|---|---|---|---|
| 1 | Mechanism | **(a), structural bias in the SSM recurrence** | §4.4. Unreachable by contrastive distillation; subsumes (c) as its soft limit; +0.43% params; cheap to falsify. | Low before Phase 4 — (b) and (c) remain specified in §4.2/§4.3/§6. |
| 2 | RC handling | **Caduceus-PH** (RC data augmentation), not PS | §4.1.0. Equivariance is orthogonal to the hypothesis; coupling them makes a null result uninterpretable. Note Lee (§I3) cites RC-equivariance as a *fix* for Evo2's failure — so PS becomes interesting later, as a follow-up, not a confound now. | Moderate — PS requires the sign-tying of §4.1.0 and a re-run. |
| 3 | Hi-C resolution | **5 kb default**, 1 kb/2 kb as an ablation axis | CHROME comparability (§D1). F4 argues for 1 kb (τ_max/bin 0.994 vs 0.199) but 1 kb buys noisier contacts. Both levels are in the same `.mcool`, so switching costs one band re-extraction and no download. | Low — re-run `phase1_acquire.py` with `RESOLUTION` changed. |
| 4 | Cross-species scope | **Out of scope for this paper** | Evo2HiC uses 177 DNA Zoo species, HiCFoundation 316 (§H3, §B1). Entering that lane against two better-resourced groups splits compute and weakens the single-chromosome pilot's already-narrow claim. Keep as future work. | Low — additive, not structural. |

**Consequential update from §I3.** Adopt **DNALongBench** (`related_work.md` §I2) as the Phase 5 evaluation suite rather than assembling tasks ad hoc, and add **Lee's perturbation-sensitivity protocol** as a head-to-head evaluation: does our structurally-conditioned model penalise TAD boundary deletions and CTCF inversions more than matched controls, where Evo2-7B does not? That is a direct quantitative claim against a published negative result, and a stronger contribution than another benchmark table.

**Additions to the §5 standing check:**
- [ ] Lee arXiv:2604.07196 is a **preprint, not peer reviewed**. Re-check status before citing it as load-bearing in the Method section, and watch for an Evo2-40B follow-up that could weaken the F3 argument.
- [ ] Confirm DNALongBench's published Caduceus-Ph numbers and splits are reusable as-is before building Phase 5 around them.

---

*Created 2026-08-07; §4, §6, §7 added same day. §4.1.2 parameter counts computed by executing [`scripts/param_accounting.py`](../scripts/param_accounting.py) — instantiated modules, not cited figures. §4.1.4 F4 numbers computed by executing [`scripts/f4_memory_horizon.py`](../scripts/f4_memory_horizon.py). Nothing in §4 has been trained or run. All Evo2HiC claims verified against the v1 full text (§H3 of `related_work.md`); keyword counts computed over retrieved full text, not abstracts.*
