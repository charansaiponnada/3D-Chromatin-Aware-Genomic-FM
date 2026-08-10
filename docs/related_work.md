# Related Work — 3D-Chromatin-Aware Genomic Foundation Model

**Phase 0 deliverable.** Literature ground truth for the claim: *chromatin 3D structure should condition genomic FM **pretraining**, not be bolted on afterwards.*

**Status legend** — every entry is tagged with how it was verified:
- `[V-FULL]` — read from full text (PMC / publisher HTML)
- `[V-ABS]` — verified from abstract + official repo/landing page only
- `[UNVERIFIED]` — plausible but *not* confirmed from a primary source; must be checked before it appears in a paper draft

Nothing in this file may be cited in `paper/draft_v1.md` while still tagged `[UNVERIFIED]`.

---

## A. Foundational 3D genome biology (what the data *is*)

### A1. Lieberman-Aiden et al. 2009 — "Comprehensive Mapping of Long-Range Interactions Reveals Folding Principles of the Human Genome" `[V-ABS]`
*Science* 326(5950):289–293. DOI: 10.1126/science.1181369

**What they do.** Introduce Hi-C: crosslink the genome in place, digest, re-ligate spatially proximal fragments, sequence the junctions. Produces the first genome-wide spatial proximity map of the human genome, at 1 Mb resolution.

**Data structure consumed.** None (it is an assay paper) — it *produces* the contact matrix, the data structure everything downstream in this list depends on.

**Structure in pretraining or downstream.** N/A.

**Two findings that matter to us.** (1) The genome partitions into two spatial compartments (A/B) tracking chromatin state. (2) At megabase scale, folding is consistent with a *fractal globule* — locally dense, knot-free, easy to unfold. The fractal-globule scaling is the reason contact probability decays smoothly with genomic distance, which is exactly the "expected" curve any structural signal must be normalized against before it carries information beyond 1D distance.

**How this project differs.** We consume Hi-C rather than produce it. The relevance is methodological: the distance-decay baseline from this paper is the null our structural conditioning must beat, or our "structure" term is just a re-parameterized positional prior.

---

### A2. Dixon et al. 2012 — "Topological domains in mammalian genomes identified by analysis of chromatin interactions" `[V-ABS]`
*Nature* 485:376–380.

**What they do.** Hi-C in human and mouse ES cells and differentiated cells; identify megabase-scale self-interacting domains ("topological domains", now TADs) as a pervasive, largely cell-type-**invariant** feature of genome organization.

**Data structure consumed.** Hi-C contact matrices; derived per-bin directionality/insulation statistics.

**Structure in pretraining or downstream.** N/A (descriptive genomics).

**Why it matters to us.** TAD boundaries are enriched for CTCF, housekeeping genes, tRNAs and SINEs. The relative cell-type invariance of TADs is the single most important practical fact for this project: it is what makes a *pretraining-time* structural prior defensible at all. If TADs were wholly cell-type-specific, conditioning a general-purpose pretrained model on one cell line's Hi-C would be baking in a cell-line artifact rather than a genome-intrinsic prior.

**How this project differs.** We treat TAD/boundary structure as a conditioning signal on a learned sequence model, not as an object of description. Mechanism (c) in `docs/architecture_spec.md` (state resets / masking at TAD boundaries) descends directly from this paper's insulation concept.

---

### A3. Rao et al. 2014 — "A 3D Map of the Human Genome at Kilobase Resolution Reveals Principles of Chromatin Looping" `[V-ABS]`
*Cell* 159(7):1665–1680. PMID: 25497547

**What they do.** In situ Hi-C across nine cell types; densest map (GM12878) has 4.9B contacts at 1 kb resolution. Identify ~10,000 loops, contact domains (median ~185 kb), and six subcompartments.

**Data structure consumed.** Hi-C contact matrices at kilobase resolution.

**Structure in pretraining or downstream.** N/A.

**Why it matters to us.** Loops frequently link promoters to enhancers, correlate with activation, and are conserved across cell types and species. Loop anchors sit at domain boundaries and bind CTCF in **convergent orientation** — orientation, not just presence. This is the concrete mechanistic hook for the project's core claim: a sequence-only model sees the CTCF motif but has no supervision that makes motif *orientation pairing across a 500 kb gap* matter. GM12878 from this paper is also the de facto highest-quality Hi-C target and a natural pilot dataset.

**How this project differs.** We use these loop calls as ground-truth structure to condition on and to evaluate against, rather than deriving them.

---

## B. Hi-C-native foundation models (structure in, sequence out — or no sequence at all)

### B1. HiCFoundation — "A generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics analysis across species" `[V-ABS]`
*Nature Methods* (2026), s41592-026-03097-8; preprint bioRxiv 10.1101/2024.12.16.628821. Noble Lab. Code: github.com/Noble-Lab/HiCFoundation

**What they do.** Self-supervised pretraining on Hi-C submatrix patches — hundreds of Hi-C assays, 81 human cell lines/tissues, ~118M contact submatrices. Emits three levels of embedding (whole chromosome / contiguous loci / single locus). Fine-tuned decoders handle reproducibility scoring, resolution enhancement, loop detection, single-cell Hi-C, and — notably — **prediction of epigenomic activity from Hi-C input**. Cross-species validated on 316 species.

**Data structure consumed.** **Hi-C contact matrices only.** DNA sequence is not an input.

**Structure in pretraining or downstream.** Structure is the *entire* pretraining signal. Sequence is absent from both stages.

**How this project differs.** HiCFoundation is the mirror image of our problem, not a competitor for it: it learns a representation *of* contact maps; we learn a representation *of sequence* that has been shaped by contact maps. HiCFoundation cannot score a variant, because a point mutation does not change its input. Our model must — that is why ClinVar is on our task list and not on theirs.

**Watch item.** Their "predict epigenomic activity from Hi-C" result is the closest thing in the literature to evidence that structure alone carries regulatory signal. If that transfer is strong, a reviewer will ask why we need sequence at all; the honest answer is variant resolution, and we should have the ablation to back it.

---

### B2. Hi-Cformer — "Hi-Cformer enables multi-scale chromatin contact map modeling for single-cell Hi-C data analysis" `[V-ABS]`
bioRxiv 10.1101/2025.08.04.668453 (posted Aug 2025). Code: github.com/Xiaoqing-Wu02/Hi-Cformer

**What they do.** Transformer with a multi-scale attention mechanism over blocks of single-cell Hi-C contact maps, capturing dependencies across genomic distances and across scales simultaneously. Produces low-dimensional per-cell representations.

**Data structure consumed.** **Single-cell Hi-C contact maps only.** No DNA sequence.

**Structure in pretraining or downstream.** Structure is the sole input; the model is a scHi-C representation learner.

**Downstream tasks.** Cell clustering, cell-type annotation, imputation of sparse 3D signal, recovery of TAD-like boundaries and A/B compartments from sparse data.

**How this project differs.** Same axis as HiCFoundation — no sequence channel, therefore no variant-level or motif-level claims possible. Hi-Cformer's real relevance to us is *methodological borrowing*: their multi-scale block attention is a candidate design for how we summarize a contact submatrix into a per-position structural context vector in Phase 1, step 2.

> `[UNVERIFIED]` The specific pretraining objective (masked patch reconstruction vs. contrastive vs. purely supervised) is **not** confirmed — the repo and abstract do not state it. Resolve from the bioRxiv full text before citing any objective-level claim. bioRxiv blocks automated fetch; download the PDF manually.

---

## C. Sequence-only genomic foundation models (our baselines and our architectural lineage)

### C1. Evo 2 — "Genome modelling and design across all domains of life with Evo 2" `[V-ABS]`
*Nature* (2026), s41586-026-10176-5; preprint bioRxiv 10.1101/2025.02.18.638918. Arc Institute + NVIDIA. Code: github.com/ArcInstitute/evo2

**What they do.** Autoregressive (next-nucleotide) pretraining on **OpenGenome2**, 8.8–9.3T tokens from >128,000 genomes spanning all three domains of life, at single-nucleotide resolution. Architecture: **StripedHyena 2**, a convolutional multi-hybrid. Two sizes: **7B** and **40B** parameters, context up to **1M tokens**. Demonstrates chromosome-scale generation and >90% accuracy on BRCA1 benign-vs-pathogenic variant classification.

**Data structure consumed.** **Raw sequence only.** No Hi-C, no chromatin structure, at any stage.

**Structure in pretraining or downstream.** Neither. Any 3D awareness is emergent-at-best from sequence statistics.

**How this project differs.** Evo 2 is the strongest existing demonstration that long-context sequence-only pretraining buys real regulatory and clinical signal — which makes it the sharpest possible statement of our null hypothesis: *at 1M-token context, does explicit 3D structure still add anything sequence alone hasn't already absorbed?* Note that Evo 2 is also **CHROME's third input modality** (frozen 7B embeddings), which makes it the shared reference point across our whole comparison.

**Compute reality check (2× L40S, 96 GB total).** Evo 2 at 7B/40B is not reproducible here and is not a training baseline for us. It enters this project only as (a) a frozen embedding source, matching CHROME's setup, and (b) a scale caveat in the Limitations section: our conclusions are about the small-model regime.

---

### C2. Caduceus — "Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling" `[V-ABS]`
ICML 2024, PMLR 235. arXiv:2403.03234. Schiff, Kao, et al. (Kuleshov group). Code: github.com/kuleshov-group/caduceus

**What they do.** Extend the Mamba SSM block to **BiMamba** (bidirectional, with shared in/out projections) and then to **MambaDNA** (additionally reverse-complement equivariant), yielding the first RC-equivariant bidirectional long-range DNA LM. Two variants: **Caduceus-PH** (RC data augmentation) and **Caduceus-PS** (RC equivariant by construction, no augmentation needed).

**Data structure consumed.** **Raw sequence only** (human reference genome, hg38).

**Structure in pretraining or downstream.** Neither. No Hi-C anywhere in the paper.

**Pretraining objective.** Masked language modeling, `mlm_probability = 0.15`, on hg38, at sequence lengths up to 131k.

**Downstream benchmarks.** GenomicBenchmarks (8 classification tasks); Nucleotide Transformer task suite; eQTL SNP variant effect prediction from the Long Range Benchmark (frozen embeddings + SVM). Reported to beat models ~10× larger on the long-range VEP task.

**How this project differs.** Caduceus is our **architectural parent and primary baseline arm**. Our novel model should be Caduceus-class — same MLM objective, same RC handling, matched parameter count within 5% — differing *only* in that chromatin structure enters the pretraining computation. It is the correct baseline precisely because it is sequence-only by design and small enough to actually train on 2× L40S.

> `[UNVERIFIED]` Parameter counts. The repo describes both variants as d_model=256, 16 layers, 131k sequence length, which yields a figure in the ~8–11M range, but **no parameter count was read from a primary source**. Do not put a number in the paper until it is read off the paper's tables or computed from our own instantiated model. Since the fairness of our whole comparison rests on "matched parameter count within 5%," compute this from our own code, not from a citation.

---

## D. Structure-conditioned regulatory models (the actual competition)

### D1. CHROME — "A chromatin-structure-guided framework for predictive and interpretable regulatory genomics" `[V-FULL]`
*Briefings in Bioinformatics* 27(4):bbag360, published 17 July 2026. DOI: 10.1093/bib/bbag360. Preprint: bioRxiv 10.1101/2025.11.03.686435. Ye, Du, Chen, Dai, Ma, Liang (UIC; SJTU).

**This is the paper the project is defined against. Read it line by line; the summary below is necessary but not sufficient.**

**Core method.** Three parts.
1. **Contact filtering (CHROMATIX null model).** Simulate 5×10⁵ self-avoiding polymer chains under confined nuclear volume to get expected contact probabilities; keep only measured Hi-C contacts exceeding that physical null at FDR 0.05. This retains ~1.8% (GM12878), ~6.0% (K562), ~5.0% (IMR-90) of raw interactions — these are the "physically specific, non-random" contacts.
2. **Graph construction.** Each predicted ChIP-seq peak defines a signal-centered subgraph: center node = the 5 kb locus containing the peak; neighbor nodes = loci joined to it by surviving non-random contacts; receptive field up to 4 Mb (800 loci at 5 kb).
3. **Graph attention.** Two stacked GAT layers propagate neighbor information into the center-node embedding with learned attention weights.

**Data structure consumed.** Both sequence and Hi-C — but through *separate, sequentially trained* pathways. Three interchangeable encoder modalities: (a) one-hot sequence through an EPCOT-like 1D CNN; (b) sequence concatenated with cell-line DNase accessibility through the same CNN; (c) **frozen precomputed Evo 2 (7B) per-locus embeddings** through a lightweight MLP projection.

**Structure in pretraining or downstream — the decisive point.** CHROME trains in **two stages**:
- *Stage 1*: encoders are trained on **center nodes only** — no graph, no neighbors, **no 3D structure at all** — supervised on ChIP-seq labels until validation saturates.
- *Stage 2*: early encoder layers are frozen, later layers unfrozen, and the GAT + classifier are optimized end-to-end. **This is the first moment structure exists in the computation.**

So: structure is strictly post-hoc. It never shapes the representation the encoder learns from raw sequence; it only reweights and mixes representations that were already fixed by a structure-blind stage. And in the Evo 2 variant, the sequence representation is entirely frozen and externally computed — structure cannot influence it even in principle.

**Training objective.** ChIP-seq profile prediction as multi-label classification with per-center masked binary cross-entropy, over 751 ENCODE assays (GM12878: 539, K562: 168, IMR-90: 44). **Note: this is supervised, not self-supervised.** CHROME has no masked-token or generative pretraining stage anywhere.

**Data.** Hi-C from 4D Nucleome, bulk, 5 kb resolution; cell lines GM12878, K562, IMR-90 for training; HepG2 (chr9 only) held out as an unseen cell line. Chromosome split: chr9 test, chr8 validation, remainder train. Epigenomic targets hg38-aligned.

**Downstream results.** ChIP-seq prediction: best absolute performance with seq+DNase; smaller but significant gains for seq-only; gains with Evo 2 embeddings but weaker absolute performance. On unseen HepG2, gains hold for DNase and Evo 2 but **seq-only falls slightly below baseline**. eQTL: CHROME embeddings beat baseline embeddings (GM12878→blood, IMR-90→lung, GTEx), with sign-agreement strongest at high effect sizes. ClinVar: improved benign/pathogenic discrimination across all input types, with pathogenic enrichment in **distal** non-random neighbors as well as center loci. Bootstrap 95% CIs positive for macro AUPR/F1/AUROC; ablation shows real non-random contacts beat degree- and distance-matched random controls.

**Stated limitations (theirs).** Bulk population-averaged Hi-C obscures cell-state heterogeneity; 4 Mb receptive field misses very long-range structure; no dynamic or allele-specific interactions; averaging DNase across cell lines for ClinVar loses cell-type context; raw-sequence features generalize poorly to unseen cell lines without accessibility.

**How this project differs — five specific ways.**
1. **Stage of entry.** CHROME's encoder is trained to convergence *before* structure exists. Ours has structure in the loss/recurrence from step 0, so structure can shape what sequence features are learned at all — not merely which already-learned features get mixed.
2. **Objective class.** CHROME is supervised on ChIP-seq labels; there is no self-supervised pretraining. Ours is a self-supervised (MLM) genomic FM. This means CHROME's representation is only ever as general as its 751 assays; ours is meant to be general-purpose and is evaluated by transfer.
3. **Where structure lives at inference.** CHROME needs the contact graph **at inference time** — it is an input to the GAT. Our target is a model where structure is a *training-time* signal distilled into weights, so inference needs sequence only. That is a materially different deployment story: CHROME cannot score a locus in a cell type with no Hi-C; we should be able to. **This is arguably the strongest differentiator and should be tested explicitly.**
4. **Architecture family.** CHROME is CNN/MLP + GAT. Ours is a Mamba/SSM sequence backbone with structure entering the recurrence, the loss, or the state-reset policy — a mechanism that has no analogue in a graph-attention layer over pooled 5 kb bins.
5. **Resolution regime.** CHROME operates on 5 kb bins as graph nodes. A nucleotide-resolution SSM conditioned on structure can, in principle, use CTCF motif orientation — the Rao et al. mechanism — which a 5 kb-binned graph cannot represent.

**Honest counter-argument we must be ready for.** CHROME's stage 2 *does* unfreeze later encoder layers. A reviewer can say "that is already joint training, so your contribution is a matter of degree." The answer must be empirical, not rhetorical: show that a model with structure in self-supervised pretraining beats CHROME-style late partial unfreezing **at matched compute and matched parameters**, and show the inference-time-structure-free result from point 3, which CHROME architecturally cannot produce.

---

### D2. GraphReg (adjacent, precursor) — "Chromatin interaction–aware gene regulatory modeling with graph attention networks" `[V-ABS]`
*Genome Research* 32(5):930, 2022. Karbalayghareh et al. Code: github.com/karbalayghareh/GraphReg

**What they do.** Graph attention networks over chromatin-interaction graphs (HiChIP/Hi-C) to predict gene expression from regulatory input, with attention providing interpretable enhancer→gene attribution.

**Data structure consumed.** Sequence/epigenomic features as node features + a chromatin interaction graph.

**Structure in pretraining or downstream.** Downstream/supervised — structure is an input to a supervised expression model, never a pretraining signal.

**How this project differs.** Same post-hoc-graph critique as CHROME, one generation earlier and without CHROME's polymer null-model filtering. Its relevance is that it establishes "GAT over contact graph" as an *established* pattern — which weakens any novelty claim framed merely as "we use structure," and strengthens the framing that our novelty is specifically about *pretraining-stage* entry.

---

### D3. Akita and Orca (adjacent — the closest genuine novelty risk) `[V-ABS]`
- **Akita**: Fudenberg et al., "Predicting 3D genome folding from DNA sequence," *Nature Methods* 2020; preprint bioRxiv 10.1101/800060.
- **Orca**: Zhou, "Sequence-based modeling of three-dimensional genome architecture from kilobase to chromosome scale," *Nature Genetics* 54:725–734 (2022), s41588-022-01065-4.

**What they do.** Train sequence models whose **prediction target is the contact map**. Akita: CNN, ~1 Mb input → contact frequency for all pairs of ~2 kb bins, trained on five high-quality Hi-C/Micro-C datasets against log(obs/exp) maps. Orca: hierarchical sequence encoder + multilevel cascading decoder giving "zooming" predictions from kilobase to whole-chromosome scale, including effects of structural variants.

**Data structure consumed.** Sequence in, contact map out.

**Structure in pretraining or downstream.** **Structure is the supervised training target.** This is the one place in the literature where a sequence model's weights are genuinely shaped by Hi-C during training.

**How this project differs — and why this needs care.** Phase 2 mechanism (b), "auxiliary contact-prediction loss alongside masked-token prediction," is close to *Akita's objective used as an auxiliary head on a genomic FM*. The novelty there is not the objective; it is the multi-task combination with self-supervised MLM and the claim that it improves **general-purpose sequence representations** for downstream regulatory and variant tasks — something Akita and Orca never claim or evaluate, since they are single-purpose folding predictors and are assessed on contact-map accuracy, not on transfer.

**Consequence for Phase 2 (act on this before choosing a mechanism).** If mechanism (b) is chosen, the paper must (i) cite Akita/Orca as the origin of the objective, (ii) frame the contribution as the *transfer* result rather than the loss, and (iii) include an arm where the auxiliary head is trained but its representation is evaluated on non-folding tasks. Mechanisms (a) and (c) — structural bias inside the SSM recurrence, and TAD-conditioned state resets — have no equivalent in this literature and are, on current evidence, the more defensible novelty claims.

---

## E. The gap, in one table

| Work | Sequence in? | Structure in? | Structure enters at | Self-supervised pretraining? | Structure needed at inference? |
|---|---|---|---|---|---|
| Lieberman-Aiden '09 / Dixon '12 / Rao '14 | — | — | *(assay/descriptive)* | — | — |
| HiCFoundation | No | Yes | pretraining | Yes | Yes |
| Hi-Cformer | No | Yes | pretraining | `[UNVERIFIED]` | Yes |
| Evo 2 | Yes | **No** | — | Yes (autoregressive) | No |
| Caduceus | Yes | **No** | — | Yes (MLM) | No |
| GraphReg | Yes (features) | Yes | supervised downstream | No | Yes |
| CHROME | Yes | Yes | **stage 2, post-hoc** | **No** (supervised ChIP-seq) | **Yes** |
| Akita / Orca | Yes | Yes (as target) | supervised training target | No | No |
| **This project** | **Yes** | **Yes** | **self-supervised pretraining** | **Yes (MLM + structure)** | **Target: No** |

The empty cell in the literature is the bottom row: **no existing model uses 3D chromatin structure to shape a self-supervised sequence pretraining objective and then evaluates the resulting general-purpose representation on regulatory and variant tasks.** Akita/Orca get closest on mechanism but are supervised single-task folding predictors. CHROME gets closest on evaluation but its sequence encoder never sees structure while it is learning.

---

## F. Phase 0 gate — say these out loud, without notes

The gate in `CLAUDE.md` is: *one sentence each on why CHROME doesn't already solve the problem.*

1. CHROME's encoder finishes training on center nodes with no graph attached — structure has provably zero influence on what sequence features get learned.
2. CHROME has no self-supervised pretraining at all; it is supervised on 751 ChIP-seq assays, so its representation is bounded by those labels rather than being a general-purpose foundation model.
3. CHROME requires the contact graph as an inference-time input, so it cannot say anything about a cell type or species with no Hi-C — the case a pretraining-time structural prior is specifically meant to cover.
4. CHROME's nodes are 5 kb bins, so CTCF motif orientation — the actual mechanism of loop formation from Rao et al. — is not representable in its graph.
5. In CHROME's own results, the seq-only variant *underperforms* its baseline on an unseen cell line, which is precisely the generalization failure a structure-shaped sequence representation is supposed to fix.

---

## G. Open items to resolve before Phase 2

- [ ] Read CHROME's full text line by line (esp. §Methods stage-1/stage-2 details and the ablation design). The summary above came from the PMC full text via automated extraction and must be human-verified before anything here is cited.
- [ ] Resolve Hi-Cformer's pretraining objective from the bioRxiv PDF (blocked to automated fetch; download manually).
- [ ] Get Caduceus's exact parameter counts and per-benchmark numbers from the ICML paper tables — needed for the matched-parameter constraint in Phase 4.
- [ ] Decide whether the cross-species arm (DNA Zoo) is in scope. If yes, HiCFoundation's 316-species validation is the relevant prior art and must be added here.
- [ ] Search specifically for 2026 preprints combining "SSM/Mamba genomic LM" with "Hi-C" — this literature is moving fast and the gap identified in §E has a shelf life. Re-run before the paper draft.

---

*Compiled Phase 0. Sources verified via web search and publisher full text on 2026-08-07. Any claim above still tagged `[UNVERIFIED]` is barred from the paper draft.*

---

# Appendix — Follow-up verification, 2026-08-07

Two targeted follow-ups requested before Phase 0 closes. Entries below are **appended, not merged**; §A–§G above are untouched and remain subject to your own independent verification.

A new `[V-PARTIAL]` tag is used here: read from primary source, but the source was incompletely accessible (paywall/truncation) and the claim rests on the portion I could actually see.

---

## H1. Have Akita or Orca representations ever been probed for general-purpose transfer? `[V-FULL]` / `[V-PARTIAL]`

**Short answer: no — on the evidence I could reach, contact-map prediction and contact-map-derived variant *disruption scores* are the only things either model's outputs have ever been evaluated on. Neither paper transfers learned representations to a non-folding task, and I found no later paper that does.** One important distinction governs this whole finding, and it should be stated explicitly in the paper:

> **Using model PREDICTIONS ≠ transferring learned REPRESENTATIONS.** Both Akita and Orca have downstream variant applications, but every one works by running the model twice — once on reference sequence, once on variant sequence — and comparing the two *predicted contact maps*. The internal embeddings are never extracted, never frozen-and-probed, never fine-tuned into another head.

### Akita (Fudenberg et al.) — preprint full text read `[V-FULL]`
bioRxiv 10.1101/800060; published as *Nature Methods* 17:1111–1117 (2020), s41592-020-0958-x.

The preprint's complete section list is: Training Data · Model architecture · Training Approach · **Comparison with 1D features** · **In silico motif mutagenesis** · **In silico CTCF motif inversions** · **Predictions for mouse DNA sequences** · **5C data processing** · **In silico deletions**, plus two supplemental notes on prior architectures and on differences from DeepC.

Every evaluation is folding-internal. There is no variant pathogenicity task, no ClinVar, no regulatory element classification, no gene expression task, and no probing of the intermediate 1D representations on anything other than the 2D folding head they feed.

The word "representation" appears in the paper only in an *architectural* sense — the Basenji-derived trunk produces "1D representations of genomic sequence" which the head averages pairwise into 2D. It never means "a transferable embedding."

Two direct quotes worth having on hand:
- The authors' own framing of transfer as **future work, not done**: *"we envision that end-to-end sequence-to-genome-folding approaches will advance our ability to design functional screens, model enhancer-promoter interactions, prioritize causal variants in association studies, and predict the impacts of rare and de novo variants."*
- An argument **against** transfer learning that cuts in our favor: discussing DeepC (which pretrains on epigenomic profiles then transfers weights), Akita's authors write that *"strict transfer learning could limit the richness of representations that a deep CNN can learn for 3D genome folding; for example, a CTCF profile may not contain information about the directionality of motifs under its peaks, which is important for predicting genome folding."*

> That second quote is a gift for our Method section. The originators of sequence→Hi-C modeling state on the record that a pretraining signal lacking CTCF **motif directionality** produces impoverished representations. That is precisely the argument for putting orientation-bearing structural signal into pretraining rather than layering it on afterwards.

> `[NOTE]` Read from the **bioRxiv preprint**, not the *Nature Methods* version of record. Section structure is unlikely to have changed materially, but confirm against the published paper before citing the quotes.

### Orca (Zhou 2022) — abstract, extended data captions, section headings, references `[V-PARTIAL]`
*Nature Genetics* 54:725–734, s41588-022-01065-4.

**The Nature Genetics full text is paywalled; I could not read Results or Methods.** What I could read: the abstract, the complete list of Extended Data figure captions, all section headings, and the reference list.

All ten Extended Data figures are folding-internal — model performance on HFF, cross-cell-type interaction differences, Polycomb-mediated interactions, promoter–enhancer interactions, transposon boundary-element insertion effects, comparison against Capture Hi-C for structural variants, multiplexed vs. single in silico mutagenesis, compartment-alteration virtual screens, random sequence permutation effects on compartment activity, and predicted effects of permuting genomic regions.

Supplementary Data 2 and 3 are, respectively, "predicted multiscale structural variant effects for all transposon insertion sites tested" and "…for all structural variants tested" — again, predicted *structural effects*, not pathogenicity labels.

The abstract's applications claim is: *"Orca enables various applications including predicting structural variant effects on multiscale genome organization and it recapitulated effects of experimentally studied variants at varying sizes (300 bp to 90 Mb)."* Recapitulating *measured 3D effects* of known variants is a folding-accuracy result, not a pathogenicity-classification result.

> `[V-PARTIAL]` **I cannot fully rule out** an unreported-in-captions evaluation buried in the Orca main text. Given the paper is central to the novelty argument, get institutional access and check the Results section directly before the paper draft asserts "Orca was never evaluated on transfer."

### Later literature — no transfer probing found `[V-ABS]`
Searches for downstream reuse of Akita/Orca representations returned only:
- Papers using Akita/Orca **predictions** as features or evidence (e.g. structural-variant interpretation, a 2025 preprint on genome 3D folding and human height variation). Predictions, not embeddings.
- Genomic-FM benchmark papers (BEND; "Benchmarking DNA Foundation Models for Genomic and Genetic Tasks"; "Tokenization to Transfer") that probe **DNA language models** — DNABERT-2, Nucleotide Transformer v2, HyenaDNA, Caduceus-Ph, GROVER — and do **not** include Akita or Orca. Sequence→Hi-C models are simply not in the benchmark population.

Notably, one such benchmark includes a **TAD region recognition** task using zero-shot embeddings. That is a directly relevant eval for Phase 5 and worth pulling in as an additional structure-sensitive probe.

> `[UNVERIFIED]` Search coverage is not proof of absence. This is a negative claim from ~6 targeted searches, several of which were polluted by unrelated ML models named "Orca" and by Akita Prefecture. Before the draft asserts novelty here, run a forward-citation sweep on both papers (Semantic Scholar / Google Scholar "cited by") and check for a transfer evaluation.

### What this means for us
The gap in §E holds and is now sharper: **sequence→Hi-C models are single-purpose folding predictors whose representations nobody — including their authors — has ever asked to do anything else.** Our contribution is therefore not "use Hi-C to train a sequence model" (Akita/Orca did that) but "**show that a Hi-C-shaped sequence representation transfers**." That transfer claim is genuinely unclaimed territory. Frame the contribution that way and the Akita/Orca prior art strengthens rather than threatens it — but only if we actually run the transfer evaluation, which is exactly what Phase 5 is for.

---

## H2. Hi-Cformer's pretraining objective — resolved `[V-FULL]`

bioRxiv full text retrieved via browser (WebFetch is blocked by bioRxiv's bot protection; the browser loads it fine — use that route for all future bioRxiv reads). Authors: **Xiaoqing Wu, Xiaoyang Chen, Rui Jiang**. Posted 5 Aug 2025.

**This closes the `[UNVERIFIED]` flag on the Hi-Cformer entry in §B2 and the corresponding cell in the §E table. The §B2 entry itself is left unedited per your instruction; treat this as the authoritative correction.**

**Objective: a masked language modeling task over contact-map block embeddings, trained as multi-scale reconstruction.** Verbatim from Methods:

> *"Inspired by these masked language modeling tasks, this work also designs a masked language modeling task to train the model. Specifically, given a masking ratio, after obtaining the embedding sequence E, all embeddings (including both chromosomal map and block embeddings) belonging to the same chromosome are randomly replaced with a learnable special embedding [MASK] at the specified ratio."*

Default masking ratio **20%**, raised for sparser contact maps and lowered for denser ones.

And from Results:

> *"Hi-Cformer is trained in a self-supervised manner to reconstruct input signals using a tailored masked language modeling (MLM) task, which encourages the model to capture long-range dependencies."*

**Loss.** Three reconstruction losses computed separately at cell level, chromosome level, and block level, combined as a weighted sum with tunable α, β, γ. Chromosome-level maps are interpolated to a fixed size before the reconstruction loss is computed. The supervised cell-type-annotation extension adds a cross-entropy term on top.

**Training schedule.** A **"preheating"** phase precedes main training: the transformer module is removed while all other modules and losses stay unchanged, then the transformer is introduced. Ablations show removing the transformer and removing MLM hurt embedding quality about equally — *"the models without the MLM task and those without the transformer module perform similarly, underscoring the critical role of the MLM task in effectively training the transformer module."* Removing preheating costs less; it mainly speeds up training.

**Answer to your structure-before-or-after-sequence question: neither — there is no sequence stage at all.** Input is *"the intra-chromosomal contact maps of all chromosomes from a single cell at a given resolution."* DNA sequence appears nowhere in the model, at any stage. What is masked is **contact-map block embeddings**, not nucleotides. The "language model" analogy is structural only: contact-map blocks are treated as token-like embeddings ordered by chromosome index, block size, and position.

**Also worth logging:** the authors explicitly claim foundation-model framing — *"Hi-Cformer, as a transformer-based model trained with a self-supervised objective using MLM, aligns with recent trends in foundation models, offering a scalable framework for learning general-purpose representations of 3D genome organization."* Their "general-purpose" scope is cells, not loci: clustering, cell-type annotation, imputation. Not variants.

**Datasets** (all 1 Mb resolution unless noted): Ramani2017, Lee2019, Tan2021A, Tan2021B, Wu2024; a 50 kb focused analysis at the ABL1 locus, chr9:132.95–138.0 Mb. **Baselines**: Higashi, scDEC-Hi-C, HiCRep/MDS, scHiCluster, PCA, LDA; supervised arm vs. scHiClassifier, logistic regression, random forest.

**Correction to the §E table.** The `[UNVERIFIED]` cell should read: Hi-Cformer **does** use self-supervised pretraining — MLM over contact-map block embeddings, 20% masking, three-scale reconstruction loss.

---

## H3. `[UNPLANNED FIND — HIGH PRIORITY]` Evo2HiC — the closest work to this project that exists `[V-FULL]`

**"Evo2HiC: a multimodal foundation model for integrative analysis of genome sequence and architecture."** Tangqi Fang, Xiao Wang, Zhiping Xiao, Shengqi Hang, Ghulam Murtaza, Junwei Yang, Hanwen Xu, Anupama Jha, **William Noble**, Sheng Wang. bioRxiv 10.1101/2025.11.18.689171, posted Nov 2025.

This surfaced while searching for Akita/Orca transfer evaluations. It was not on the crash-course reading list and it is **closer to your thesis than CHROME is**. Same lab as HiCFoundation (Noble Lab, UW).

**What they do.** Distill Evo 2 (7B, frozen) into a compact **3.6M-parameter, 7-layer CNN** sequence encoder trained on 1.2M 2 kb human genomic bins, and **guide that distillation with Hi-C**. Two SigLIP-style contrastive objectives, applied in two stages:
1. **Sequence distillation** — align CNN sequence embeddings with frozen Evo 2 embeddings, via linear projections into a shared 512-d space, with learnable temperature and bias.
2. **Structure distillation** — align the CNN's 2D DNA embeddings with **Hi-C patch embeddings** from a CNN Hi-C encoder. Positive pairs are Hi-C and 2D DNA embeddings from the *same pixel*; negatives are different pixels. Stated purpose: *"transfer structural knowledge, such as chromatin loops and topologically associating domains (TADs), from Hi-C contact maps into the DNA encoder."*

They deliberately avoid aligning Evo 2 directly to the Hi-C encoder, for compute reasons. Result: ~500× cheaper inference than Evo 2 at 70 kb.

**Two encoder modes:** a DNA-only encoder, and a joint encoder that additionally consumes Hi-C.

**Evaluations.** (1) Hi-C contact matrix prediction from sequence — **+10.9% Spearman over Orca**, and beats Evo 2 with a matched U-Net decoder; also best on insulation score. (2) Epigenomic signal prediction from joint Hi-C+sequence embeddings, five assays (DNase, CTCF, H3K27ac, H3K27me3, H3K4me3) — +34.7% over Hi-C-only, +26.2% over Evo 2 cross-chromosome; wins 3/5 cross-cell-line. (3) Cell-type-specific motif interpretation via conditioned attribution scores, recovering CTCF as a shared motif. (4) Hi-C resolution enhancement across **177 DNA Zoo species**.

**Why this is a real threat to the framing in §D1 and §E.** Structure genuinely shapes a *sequence* encoder's representation during its representation-learning stage — not post-hoc, not as a graph layer on frozen embeddings. Concretely, it partially preempts **differentiator #3** in §D1: their DNA-only encoder needs no Hi-C at inference, so "structure at training time, sequence-only at inference" is no longer uniquely ours.

**Why the gap in §E nonetheless survives — three specifics, verified by keyword count over the full text.**
1. **No variant-level evaluation whatsoever.** Occurrences across the entire paper: `ClinVar` **0**, `pathogenic` **0**, `eQTL` **0**, `variant` **0**, `splice` **0**, `GUE` **0**. Their entire eval surface is contact maps, epigenomic signal tracks, motifs, and resolution enhancement. The transfer claim in H1 remains unclaimed.
2. **The objective is contrastive distillation from a frozen teacher, not self-supervised pretraining.** Occurrences: `masked` **0**, `MLM` **0**. Nothing here learns from raw sequence de novo — the CNN's knowledge ceiling is Evo 2's frozen embeddings plus Hi-C alignment. A model whose structure-aware signal is *inside* a generative/masked objective over nucleotides is still unbuilt.
3. **No SSM/state-space work.** Occurrences: `Mamba` **0**, `state space` **0**. Architecture is CNN throughout. Phase 2 mechanisms (a) structural bias in the SSM recurrence and (c) TAD-conditioned state resets have no counterpart here.

**Their stated limitations — the third one is the project, named as open.** Verbatim: *"in the current work we extracted embeddings from Evo 2 by freezing its parameters, as the model's size makes fine-tuning computationally challenging. We plan to develop efficient fine-tuning strategies that enable adapting Evo 2 with Hi-C data, potentially unlocking its full capacity for chromatin structure analysis."*

> Read that carefully. The Noble Lab has publicly flagged "actually adapt the sequence foundation model itself with Hi-C" as future work **they have not done**, and named compute as the reason. That is simultaneously the strongest possible validation of the research question and a clear signal that a well-resourced group intends to pursue it. Our 2× L40S constraint is an advantage in exactly one respect: a small-model, matched-compute study is the version of this experiment they are not positioned to run quickly.

**Actions this forces.**
- [ ] Evo2HiC must be a **first-class related-work entry and a comparison arm**, not a footnote. It, not CHROME, is now the nearest neighbour on the "structure shapes the sequence encoder" axis.
- [ ] Re-frame the contribution around the two things Evo2HiC does not do: **(a)** structure inside a self-supervised objective over nucleotides rather than contrastive alignment to a frozen teacher, and **(b)** evaluation of the resulting representation on **variant and regulatory transfer tasks**. Both survive contact with this paper; "structure before inference" does not.
- [ ] Reconsider differentiator #3 in §D1. It still holds against CHROME. It does **not** hold against Evo2HiC. Do not let it into the draft unqualified.
- [ ] Watch for a v2 or a journal version adding variant evaluation. If they add ClinVar/eQTL, the transfer gap closes and the contribution must be re-scoped. **Re-check before Phase 4 commits compute.**

---

## Status of §G open items after this pass

- ~~Resolve Hi-Cformer's pretraining objective~~ — **done**, §H2.
- Akita/Orca transfer audit — **done for Akita `[V-FULL]`, partial for Orca `[V-PARTIAL]`**; Orca main text still needs institutional access, and a forward-citation sweep on both is still outstanding.
- CHROME line-by-line read — **still open**, yours.
- Caduceus parameter counts — **still open**.
- Cross-species/DNA Zoo scope decision — **now more urgent**: Evo2HiC uses 177 DNA Zoo species and HiCFoundation uses 316. If we go cross-species we are entering a crowded lane against two well-resourced groups.
- ~~Search for 2026 preprints combining SSM/Mamba genomic LMs with Hi-C~~ — partially done and it immediately produced §H3. **The `Mamba`=0 count in Evo2HiC means the SSM+Hi-C combination is still unoccupied.** Re-run this sweep before the draft regardless.

*Appendix compiled 2026-08-07. Hi-Cformer, Akita, and Evo2HiC read from primary full text via browser; Orca `[V-PARTIAL]` (paywalled). Keyword counts in §H3 computed over the retrieved full text of the preprint, not the abstract.*

---

# Appendix II — Forward-citation sweep, 2026-08-07

Closes the outstanding sweep from §H1. Ran forward citations via Semantic Scholar: **118** papers citing Orca, **200** citing Akita.

## I1. Akita/Orca transfer — negative result confirmed `[V-ABS]`

No citing paper probes Akita's or Orca's **learned representations** on a non-folding task. Citing work falls into three buckets, none of which is representation transfer:

1. **Other models citing them as prior art** (AlphaGenome, HiCFoundation, Evo2HiC, EpiGePT).
2. **Reviews** of non-coding variant effect prediction that discuss their *predictions*.
3. **Benchmarks that include Akita as a task baseline** — not as an encoder whose embeddings get transferred.

This upgrades §H1's claim from "6 searches found nothing" to "a forward-citation sweep over 318 citing papers found nothing." The transfer gap in §E stands.

## I2. `[IMPORTANT]` DNALongBench — a ready-made Phase 5 evaluation suite `[V-ABS]`

**"DNALONGBENCH: a benchmark suite for long-range DNA prediction tasks."** Cheng, Song, Zhang, Wang, Wang, Yang, Ma. *Nature Communications* 16(1):10108, 2025. Preprint bioRxiv 10.1101/2025.01.06.631595. PMC12627797.

Five long-range tasks, dependencies up to 1 Mb: **enhancer–target gene interaction**, **eQTL**, **3D genome organization (contact map)**, regulatory sequence activity, transcription initiation. Baselines include **HyenaDNA, Caduceus-Ph, and Akita** (as the "expert model").

**Why this matters a lot for us.** Three of its five tasks are our task list, it already benchmarks **Caduceus-Ph** — our chosen baseline architecture — and it includes contact-map prediction as a task. Phase 5 should adopt DNALongBench rather than assembling an evaluation from scratch: it gives published comparison numbers, a standard split, and removes the "you picked flattering tasks" objection entirely.

**Calibration to note:** contact-map prediction is by far the hardest task in the suite — best reported stratum-adjusted correlation is **0.233**, by Akita. Sequence-only models do worse. Set expectations accordingly if mechanism (b)'s auxiliary head is ever evaluated on it.

## I3. `[MAJOR — SUPPORTS THE PROJECT]` Evo2 fails at higher-order 3D structure `[V-FULL]`

**"Probing 3D Chromatin Structure Awareness in Evo2 DNA Language Model."** UkJin Lee (Weill Cornell Graduate School of Medical Sciences). arXiv:2604.07196v1, 8 April 2026. Code: github.com/ukjinlee101/evo2-3d-chromatin

> **Correction to an earlier reading in this project.** An automated summary of this paper reported that Evo2 "captures substantial information about 3D chromatin structure." **That is the opposite of what the paper finds.** The full text was read directly; the summary was wrong. Everything below comes from the paper itself.

**What they did.** Probed **Evo2-7B** on 1 Mb hg38 windows centred on features from H1-ESC Micro-C (4DN 4DNES21D8SP8), CTCF ChIP-seq (ENCODE), and FIMO motif scans. Two complementary tests:
- *Perturbation sensitivity* — does Evo2's likelihood penalise functional disruptions (5 kb TAD boundary deletions; 19 bp CTCF motif inversions and deletions) more than GC- and size-matched random controls?
- *Sequence generation* — do Evo2-generated segments produce plausible 3D structure when evaluated through **Orca**?

Cohorts: 231 TAD boundary regions (strong w/ CTCF, strong w/o CTCF, weak, and matched boundary controls) and 120 convergent CTCF loop regions with Micro-C validation.

**What they found — Evo2 fails on both tests.**

| Test | Result |
|---|---|
| TAD boundary deletion | **Weaker** likelihood penalty than matched random controls (mean paired difference +1.42×10⁻⁴); deletions exceeded controls in only 15/36 regions; paired Wilcoxon **p = 0.405**, no category significant |
| CTCF motif inversion | **Less** penalised than matched controls (+9.5×10⁻⁶, p = 0.021) |
| CTCF motif deletion | **Less** penalised than matched controls (+1.6×10⁻⁵, p = 0.006) |
| TAD boundary generation | median generated insulation **0.407** vs reference **0.587** (median delta −0.134); CTCF motifs recovered in 4/10 |
| Convergent loop generation | only **5/10** produced convergent motif pairs; median loop enrichment **0.054** vs reference **0.388** (median delta −0.280) |

Their conclusion, verbatim: *"Evo2 has learned local CTCF grammar but misses higher-order 3D organization."* And: *"revealing fundamental limitations of current DNA language models for encoding higher-order genome organization."*

**Why this is the most favourable finding in the whole literature review — three ways.**

1. **It substantially defuses failure mode F3.** F3 in `architecture_spec.md` §4.1.4 is "structure is predictable from sequence, so structural input is redundant." This is direct evidence that a **7B model with 1 M context trained on 9.3T tokens** does *not* infer higher-order 3D organisation from sequence. If Evo2 cannot, a 7.7 M-parameter model certainly cannot, and the structural input is carrying information the sequence pathway does not already have. F3 remains possible for *local* structure — they explicitly find local CTCF grammar **is** learned — but not for the TAD/loop scale our mechanism targets.
2. **It independently prescribes our architecture.** From the Discussion, verbatim: *"3D-aware DNA language models will require bidirectional architectures, cell type conditioning, and explicit 3D contact inputs rather than longer contexts alone."* They name **Caduceus** by citation as the bidirectional, RC-equivariant answer to Evo2's autoregressive asymmetry. Our stack is a Caduceus-class bidirectional backbone with explicit 3D contact input. This is an independent group arriving at our design from the opposite direction.
3. **It explains *why* Evo2 fails, in a way that favours the SSM route.** They attribute the failure partly to Evo2's **autoregressive left-to-right scoring**, which "breaks the symmetry of double-stranded DNA, so penalties propagate only downstream and orientation-dependent features are intrinsically hard to score." Per-position rescoring showed penalties concentrated ~100–200 bp 3′ of the edit. **Orientation-dependent features are exactly CTCF convergence** — the Rao et al. mechanism (§A3), and the thing §D1 differentiator #5 argued a 5 kb-binned graph cannot represent.

**Their stated limitations, recorded honestly.** Compute restricted them to subsampled regions and precluded testing **Evo2-40B**; they evaluated only CTCF/cohesin-mediated structures, leaving Polycomb domains, promoter–enhancer hubs and tissue-specific super-enhancer contacts to future work. So this is a 7B result on one structural class, not a universal claim. It is also a **preprint, not peer reviewed**.

**Actions.**
- [ ] Cite in the Method section as the motivation for explicit structural conditioning, and in Limitations as the reason F3 is considered unlikely at TAD scale.
- [ ] Adopt their perturbation-sensitivity protocol as a **Phase 5 evaluation**: does *our* structurally-conditioned model penalise TAD boundary deletions and CTCF inversions more than matched controls, where Evo2 does not? That is a direct, quantitative, head-to-head claim against a published negative result — and it is a far stronger contribution than another benchmark table.
- [ ] Their cohorts are hg38/GRCh38 and derived from public 4DN and ENCODE data, so they are directly reusable with our pilot.

---

*Appendix II compiled 2026-08-07. §I3 read from the primary PDF in full after an automated summary was found to have inverted the paper's conclusion.*
