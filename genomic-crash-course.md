# Genomics Crash Course: Zero → Moderate-Expert (for a 3D-Chromatin-Aware Genomic FM)

This is built specifically to get you fluent enough to design and defend the research problem: *baking chromatin 3D structure into genomic foundation model pretraining.* Every level ends with what you should be able to explain out loud before moving on, and what it's for in your project.

---

## Level 0 — The Absolute Basics

**What DNA actually is**
DNA is a double-stranded molecule made of 4 bases: A, T, G, C. A pairs with T, G pairs with C — this is why the two strands are "complementary." The human genome is ~3.1 billion base pairs, split across 23 chromosome pairs (46 total).

**Central dogma**
DNA → RNA (transcription) → Protein (translation). A **gene** is a stretch of DNA that gets transcribed into RNA. Only ~1-2% of the genome directly codes for protein — the rest is regulatory, structural, or currently poorly understood ("non-coding DNA," not "junk DNA" — that term is outdated).

**Gene structure**
A gene has **exons** (coding, kept) and **introns** (non-coding, spliced out of the RNA before translation). Upstream of a gene sits its **promoter** — where transcription machinery binds to start reading. Nearby or far away sit **enhancers** — sequences that boost transcription of a gene, sometimes from hundreds of thousands of base pairs away. This last fact is the seed of your whole research problem: *how does a distant enhancer "reach" its gene?* Answer, coming in Level 3: physical 3D folding.

**Species and variation**
All humans share ~99.9% of their DNA. Differences between species are driven by both sequence divergence and differences in *regulation* — often the same genes exist across species, but when/where/how much they're expressed differs. This is why cross-species genomic modeling is hard: it's not just "different letters," it's different regulatory logic layered on similar genes.

✅ **Checkpoint:** Explain the difference between a coding and non-coding region, and why an enhancer 200kb away from a gene is a real biological phenomenon and not a modeling error.

---

## Level 1 — Genomics as a Sequence-Level System

**Regulatory elements you'll keep seeing:**
- **Promoter** — transcription start site, often has a "TATA box" motif (this is literally one of your GUE benchmark tasks)
- **Enhancer** — boosts transcription, position-independent, can be upstream, downstream, or inside introns
- **Silencer** — represses transcription
- **Splice site** — exon/intron boundary; alternative splicing lets one gene produce multiple protein variants
- **Transcription factor binding site (TFBS)** — short motifs (~6-20bp) where regulatory proteins dock

**Epigenetics (without the sequence changing):**
DNA is wrapped around proteins called **histones**. Chemical tags on DNA (methylation) or histones (acetylation, methylation) change how tightly packed and how "readable" a region is — without altering the underlying sequence. This is the layer that makes the same genome produce a neuron in your brain and a skin cell, and it's also central to disease and mutation research.

**How researchers currently measure any of this (you'll see these dataset names constantly):**
- **RNA-seq** — what's being transcribed, and how much
- **ChIP-seq** — where a specific protein (e.g. a transcription factor, or a histone mark) binds across the genome
- **ATAC-seq** — which regions of the genome are "open" (accessible) vs tightly packed
- **Hi-C** — which regions of the genome are physically close in 3D space (this is your Level 3 tool)

✅ **Checkpoint:** Given a new dataset, you should be able to say "this measures accessibility" vs "this measures expression" vs "this measures 3D proximity" just from knowing which assay produced it.

---

## Level 2 — Genomics as Data (the ML-relevant layer)

**Reference genomes and coordinates**
A "reference genome" (e.g. GRCh38 for human) is a coordinate system: chromosome + position. Every dataset (ChIP-seq peaks, Hi-C contacts, gene annotations) is expressed relative to it. Cross-species work means every species has its *own* reference genome and coordinate system — a big part of why phylogeny-aware modeling is nontrivial.

**Common file formats you'll touch**
- **FASTA** — raw sequence
- **BED** — genomic intervals ("chr3:1000-2000 is an enhancer")
- **GTF/GFF** — gene annotation (where genes, exons, introns are)
- **.hic / .cool** — Hi-C contact matrices (this is the one that matters most for you)

**Benchmarks you already know, contextualized**
GUE (Genome Understanding Evaluation) — the benchmark you used for HopField-Mamba — is built almost entirely from Level 1 tasks: promoter detection, splice site prediction, TF binding. None of GUE's current tasks are 3D-structure-aware. That's part of why your proposed contribution is novel — you'd need to either extend GUE-style eval with a regulatory/3D task, or adopt tasks from the Hi-C/regulatory genomics literature (enhancer-promoter interaction prediction, ClinVar variant pathogenicity — both used by CHROME, which we discussed).

✅ **Checkpoint:** You should be able to look at a Hi-C `.cool` file and a ChIP-seq BED file and know immediately which one tells you "close in 3D space" vs "this protein binds here."

---

## Level 3 — The 3D Genome (your core research territory)

**Packing hierarchy** (memorize this order — it's the backbone of everything else here):
1. **DNA double helix** (2nm wide)
2. Wrapped around **histones** → **nucleosomes** ("beads on a string")
3. Further coiled into **chromatin fiber**
4. Folded into large-scale domains: **TADs**
5. TADs cluster into **A/B compartments**
6. Compartments occupy **chromosome territories** within the nucleus

**TADs (Topologically Associating Domains)**
Self-interacting genomic neighborhoods — DNA inside a TAD contacts itself far more than it contacts DNA outside the TAD, even though both are "close" in raw sequence terms. TAD boundaries are often marked by a protein called **CTCF**, working with a ring-shaped protein complex called **cohesin**. Enhancers overwhelmingly act only within their own TAD — this is *the* mechanistic answer to "how does a distant enhancer reach its gene": the DNA physically loops so the enhancer and promoter touch, and the TAD boundary constrains which enhancers can reach which genes.

**Loop extrusion model**
The current leading mechanistic model: cohesin rings actively extrude a loop of DNA through themselves until they hit a pair of CTCF binding sites (in convergent orientation), which stall the ring and stabilize a loop. This is why CTCF motif orientation, not just presence, matters — a detail that's almost never encoded in current sequence-only genomic FMs, and a candidate architectural signal for you.

**A/B compartments**
At a coarser scale than TADs, the genome separates into "A" (open, active, gene-rich) and "B" (closed, inactive) compartments — visible as a checkerboard pattern in a Hi-C contact matrix.

**Hi-C, concretely**
Hi-C works by chemically freezing the genome's 3D shape, cutting DNA, and re-ligating pieces that are physically near each other — then sequencing to find out which distant sequence pairs got stitched together. The output is a **contact matrix**: rows and columns are genome positions, and the value is "how often were these two positions found close together." This is the data structure CHROME and HiCFoundation build on, and it's what you'd need to source (public datasets exist — 4D Nucleome, ENCODE, DNA Zoo for cross-species) to condition your pretraining on.

✅ **Checkpoint:** Given a Hi-C contact matrix, you should be able to point to what a TAD looks like visually (a bright square block along the diagonal), what a loop looks like (an off-diagonal bright dot), and explain in one sentence why CTCF orientation matters for loop formation.

---

## Level 4 — Where Genomics Meets Your Architecture Work

At this point you have enough vocabulary to read the actual papers, not just abstracts. Read in this order:

1. **Lieberman-Aiden et al. 2009** (original Hi-C paper) — for the assay itself
2. **Dixon et al. 2012** — TADs, first described
3. **Rao et al. 2014** — high-resolution Hi-C, loops, CTCF/cohesin loop extrusion
4. **CHROME paper** (the one we found) — closest existing work to your idea; read this one *closely*, line by line, because your contribution is defined relative to it
5. **HiCFoundation** and **Hi-Cformer** — so you understand what a Hi-C-*only* foundation model looks like, and why yours is different (sequence + structure, jointly, from pretraining)
6. **Evo2 / StripedHyena2 paper** and **Caduceus paper** — your architectural baselines

**The specific question you're now equipped to answer:** how do you turn a Hi-C contact matrix into a signal a sequence model's pretraining can use? Candidate entry points once you're here: contact-map-derived positional biases injected into attention/state-space recurrence, an auxiliary contact-prediction pretraining loss alongside masked-token prediction, or graph-structured input where TAD/loop membership modulates how far information can propagate along the sequence.

---

## Suggested pace

Given you're doing this alongside your Aynstyn work and BAH submission — 2-3 weeks is realistic if you spend ~1-2 hrs/day:

- **Days 1-3:** Levels 0-1 (get comfortable saying the vocabulary out loud, no papers yet)
- **Days 4-7:** Level 2, plus actually download and open a `.cool` Hi-C file and a GTF file so the formats aren't abstract
- **Days 8-14:** Level 3 — this is the core investment, re-read until loop extrusion and TAD/compartment structure feel intuitive, not memorized
- **Days 15-21:** Level 4 — read the 6 papers in order, take notes specifically on "what does this paper's architecture do with the DNA sequence vs. what does it do with 3D structure"

By the end, you won't be a genomics PhD, but you'll know enough to design the mechanism, defend it against someone who does have that background, and read new papers in this space without getting lost.