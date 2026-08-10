# Data Card — Pilot Dataset (Phase 1)

**Scope: a single chromosome of a single cell line of a single species.** Every number and conclusion derived from this dataset inherits that limitation. This document exists so the paper's Limitations section can be written from record rather than from memory.

**Status:** Phase 1 **step 1 complete** — pilot acquired 2026-08-07, provenance and measured properties recorded below, all md5s verified. Steps 2 (φ features) and 3 (visual validation) are outstanding; the Phase 1 gate in §6 has **not** been passed.

Acquisition script: [`scripts/phase1_acquire.py`](../scripts/phase1_acquire.py). Machine-readable provenance: `data/pilot_manifest.json` (tracked in git; the data files themselves are not).

---

## 1. Sources

### 1.1 Hi-C — 4D Nucleome

| Field | Value |
|---|---|
| Experiment set | **4DNES3JX38V5** |
| Assay | in situ Hi-C, MboI digestion, bio-dATP fill-in |
| Biosample | GM12878 (lymphoblastoid cell line) |
| Lab | Erez Lieberman Aiden, BCM |
| Publication | Rao SS et al. (2014), PMID:25497547 |
| Assembly | GRCh38 |
| Status at retrieval | `released` |

Selected because it is the primary in situ Hi-C experiment from Rao et al. 2014 — the densest human Hi-C map available (`related_work.md` §A3) — and because GM12878 is one of CHROME's three training cell lines, keeping this pilot comparable to the arm we benchmark against.

**Files used:**

| Accession | Role | Handling |
|---|---|---|
| `4DNFIXP4QG5B` | multires contact matrix (`.mcool`, 27.4 GB) | **never downloaded** — read remotely via HTTP range requests |
| `4DNFIVK5JOFU` | TAD boundary calls (`.bed.gz`) | mirrored locally |
| `4DNFIBMOGOZC` | insulation score, diamond method (`.bw`) | mirrored locally |
| `4DNFILYQ1PAY` | A/B compartment eigenvector (`.bw`) | mirrored locally |

The last three are 4DN's own derived tracks. **They are not inputs to the model.** They are held as an independent validation target for the φ features we compute ourselves in step 2 — if our insulation score disagrees with 4DN's, our pipeline is wrong, and we want to find that out before training rather than after.

**Access note.** 4DN's `@@download` endpoint returns HTTP 403 to unauthenticated clients. The script resolves each accession through the 4DN API and reads the `open_data_url` field, which points at the public S3 bucket and supports range requests. No credentials are involved and no authentication wall is bypassed — this is the documented public path for released files.

### 1.2 Reference sequence — Ensembl

GRCh38, release 113, chromosome 9 primary assembly:
`https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.9.fa.gz`

### 1.3 Annotation — GENCODE

Release 47, comprehensive annotation, filtered to chr9:
`https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz`

> UCSC's `hgdownload` mirror was unreachable from the development machine (connection timeout, not a 4xx). Ensembl and the EBI GENCODE mirror were used instead. Same assembly, same coordinates.

---

## 2. Coordinate system and the one naming hazard

All three sources are **GRCh38**, so coordinates are directly comparable with no liftOver.

**Chromosome naming differs and this is the most likely source of a silent off-by-everything error:**

| Source | Chromosome name |
|---|---|
| 4DN Hi-C (cooler bins) | `chr9` |
| GENCODE GTF | `chr9` |
| **Ensembl FASTA** | **`9`** |

The acquisition script rewrites the Ensembl FASTA header to `chr9` on write. **Only the label changes; no coordinates are touched.** Both are GRCh38 primary assembly and are 1:1 positionally.

This is recorded here rather than left in code comments because a naming mismatch would not crash anything — it would silently produce an empty join, and a model trained on empty structural features would look exactly like failure mode F1 in `architecture_spec.md` §4.1.4. If F1 fires during Phase 4, **re-check this first.**

---

## 3. Processing

### 3.1 What was extracted

| Product | Content |
|---|---|
| `data/interim/hic_band_chr9_5000bp.npz` | near-diagonal band, ±2 Mb, 5 kb bins, ICE-balanced, upper triangle, COO |
| `data/interim/hic_coarse_chr9_250000bp.npz` | dense chr9 matrix at 250 kb, balanced — for compartment calling |
| `data/interim/chr9.fa` | chr9 sequence, header normalised |
| `data/interim/gencode.v47.chr9.gtf` | annotation rows on chr9 |

### 3.2 Why a band rather than the full matrix

The full chr9 matrix at 5 kb is 27,679 × 27,679. We do not need it. Per `architecture_spec.md` §4.1, the φ features are per-bin: insulation (diamond windows of a few hundred kb), directionality index (~2 Mb window), and local contact mass. All are computable from a ±2 Mb band. Compartments are the exception — they are a whole-chromosome property — and are therefore taken from a separate coarse 250 kb matrix where the full chromosome is cheap.

Extraction tiles the diagonal in 10 Mb windows overlapping by 2 Mb, so no band entry is lost at a tile seam. Overlap-duplicated entries are deduplicated on `(row, col)`.

### 3.3 Normalization

Contacts are read with `balance=True` — the ICE-balanced weights stored in the cooler by 4DN's processing pipeline, not a normalization we applied. Non-finite entries (bins with no balancing weight, typically low-mappability regions) are dropped at extraction.

**Not yet applied, and required before φ is built:** distance-decay correction (observed/expected). Raw balanced contact frequency is dominated by the P(s) decay curve (`related_work.md` §A1), and feeding it in uncorrected would make the structural signal largely a re-encoding of 1D genomic distance — precisely what control **S3** in `architecture_spec.md` §4.1.3 exists to detect. Step 2 must apply O/E before deriving φ.

### 3.4 Resolution choice

5 kb, matching CHROME's bulk Hi-C resolution (`related_work.md` §D1) for comparability.

**Open risk — failure mode F4. MEASURED 2026-08-07, and the news is not good.** At Mamba's standard initialization for the planned config, the slowest channel's memory horizon is **994 tokens ≈ 1 kb**, and **zero** (channel, state) pairs reach the 5,000 tokens spanned by one bin. Full numbers and consequences in `architecture_spec.md` §4.1.4 F4; script: [`scripts/f4_memory_horizon.py`](../scripts/f4_memory_horizon.py).

Two things follow for this data card:

1. **φ must be interpolated to token resolution**, linearly between bin centres, rather than assigned as a step function per bin. This is now a requirement of the dataset builder, not an option.
2. **Resolution is an ablation axis.** The pilot `.mcool` carries 1 kb and 2 kb levels alongside 5 kb, so switching costs no new download — only a re-run of the band extraction with `RESOLUTION` changed. 5 kb is retained as the default for CHROME comparability; 1 kb (τ_max/bin = 0.994) is the alternative if F4 bites, at the cost of noisier contacts.

---

## 4. Measured properties

Measured from the acquired files on 2026-08-07. All md5s verified against the 4DN API records at download.

| Property | Value |
|---|---|
| chr9 length (GRCh38), per cooler chromsizes | 138,394,717 bp |
| chr9 length, per Ensembl FASTA | **138,394,717 bp — exact match** |
| Bins at 5 kb | 27,679 |
| Band half-width | ±2,000,000 bp = 400 bins |
| Band entries retained (upper triangle, finite) | 7,108,483 |
| Band cells in range (upper triangle) | 11,019,079 |
| **Band occupancy** | **0.6451** |
| Non-finite values in retained band | 0 |
| Balanced contact values | min 4.25×10⁻⁶, median 3.10×10⁻⁴, max 0.942 |
| Bins with no balancing weight | 5,345 (**19.31%**) |
| Coarse matrix @ 250 kb | 554 × 554, finite fraction 0.6452 |
| chr9 N (assembly gap) bases | 16,604,167 (**12.00%**) |
| GENCODE rows on chr9 | 169,668 (3,190 genes, 16,528 transcripts, 93,738 exons) |
| Total on disk | 348 MB |

### 4.1 Coordinate alignment — first check passed

The cooler's chr9 length and the Ensembl FASTA length agree **exactly** at 138,394,717 bp. Given §2's naming hazard, this is the cheapest possible confirmation that the two sources describe the same coordinate system. It is necessary, not sufficient — §6 still requires the positional check — but a mismatch here would have been fatal and silent.

### 4.2 The 17 Mb dead zone — expected, and it doubles as a validation

The banded fetch returned almost nothing for chr9:40–66 Mb, and **exactly zero** for chr9:48–58 Mb. This is not a pipeline fault. Two independent lines of evidence agree:

- **Hi-C:** contiguous runs of bins with no balancing weight at **chr9:43,270,000–60,520,000 (17.25 Mb)** and chr9:61,985,000–62,530,000 (0.55 Mb).
- **Sequence:** the FASTA is **100% N from ~46 to ~60 Mb**, rising from 1.4% N at 42–44 Mb through 24% at 44–46 Mb and falling back through 41% at 60–62 Mb.

That is the chr9 centromere plus the 9q12 pericentromeric heterochromatin block, which GRCh38 represents as an assembly gap. Unmappable sequence produces no reads, which produces no contacts, which produces no balancing weight.

**This is a second, stronger coordinate-alignment check**: two entirely independent files — a contact matrix from 4DN and a FASTA from Ensembl — place the same feature at the same coordinates. Had the `9`→`chr9` normalisation gone wrong, or had an off-by-one crept into the bin↔position mapping, these intervals would not coincide.

**Consequences that must be handled in step 2 and in Phase 4 window sampling:**

1. **~19% of chr9 bins carry no structural signal.** Training windows overlapping them will have undefined φ. They need an explicit mask, not silent NaN propagation — a NaN reaching `W_Δs s_t` would poison the whole recurrence.
2. **A 17 Mb contiguous hole sits in the middle of the pilot chromosome.** Window sampling must not stride blindly across it. The usable chromosome is effectively two arms of roughly 43 Mb and 78 Mb.
3. **12% of chr9 sequence is N.** The tokenizer needs a stated policy for N (own token vs. mask vs. exclude window), and that policy must be identical for baseline and structural arms or the comparison is confounded.
4. **Band occupancy is 0.645, not 1.0.** Even inside mappable regions the band is not fully populated. Whether a missing band cell means "no contact observed" or "not measurable" is a distinction step 2 has to make explicitly when building φ, because they should not be encoded identically.

---

## 4A. Step 2 — the φ feature vector

Built by [`scripts/phase1_features.py`](../scripts/phase1_features.py) → `data/processed/phi_chr9_5000bp.npz`. **This file owns the definition of φ**; `architecture_spec.md` §4.1 gave a working sketch and explicitly deferred the choice to Phase 1.

| idx | feature | window | RC symmetry |
|---|---|---|---|
| 0 | insulation | 100 kb diamond | symmetric |
| 1 | insulation | 250 kb diamond | symmetric |
| 2 | insulation | 500 kb diamond | symmetric |
| 3 | directionality index (Dixon 2012) | 2 Mb | **antisymmetric** |
| 4 | log contact density | 2 Mb | symmetric |
| 5 | upstream mass fraction | 2 Mb | **antisymmetric** |
| 6 | short/long range ratio | 100 kb / 2 Mb | symmetric |
| 7 | A/B compartment eigenvector | 250 kb, GC-oriented | symmetric |

**The symmetry column is load-bearing, not decoration.** `architecture_spec.md` §4.1.0 needs it for the reverse-complement sign-tying decision, and failure mode **F7** is exactly the case where features 3 and 5 cancel between the forward and reverse Mamba passes. Both the symmetry vector and the standardisation constants are stored in the `.npz`.

**Coverage:** 21,519 / 27,679 bins (77.74%) have a complete φ. Standardisation is chromosome-wide (zero mean, unit variance) over those bins only, per §4.1.1's F1 mitigation. Raw values are retained alongside the standardised ones.

### 4A.1 Validation against 4DN's independently derived tracks

| Our feature | vs 4DN track | Pearson r | n |
|---|---|---|---|
| insulation 100 kb | 4DNFIBMOGOZC insulation | **+0.9969** | 21,740 |
| insulation 250 kb | same | +0.7802 | 21,613 |
| insulation 500 kb | same | +0.4895 | 21,515 |
| compartment PC1 | 4DNFILYQ1PAY compartments | **+0.9759** | 22,250 |

The 100 kb window reproduces 4DN's track almost exactly, which identifies their diamond window as ~100 kb and confirms our implementation. The declining correlation at 250 kb and 500 kb is the expected consequence of comparing *different window sizes* against a fixed reference, not a defect. Compartments were derived independently — O/E → correlation matrix → PC1, oriented by GC content from the FASTA — and land at r = +0.976.

### 4A.2 Boundary calling — a weaker number, and why it does not matter here

Our naive caller (deepest local minima of insulation) recovers ~47–50% of 4DN's 300 chr9 boundary calls to the exact 5 kb bin, with the remainder scattered far away rather than near-missing. Taking the top 174 of ours against 4DN's 174 "Strong" calls gives 49.4% exact agreement and a median distance of 2 bins.

**This reflects the crudeness of our thresholding, not an error in the underlying signal** — the insulation score itself matches 4DN at r = 0.997. 4DN uses a calibrated caller with prominence criteria that we did not replicate.

**It is also off the critical path.** φ consumes the *continuous* insulation score; mechanism (a) takes a continuous s_t. Discrete boundary calls would matter for mechanism (c), which is deferred to `architecture_spec.md` §6. If (c) is ever revisited, a real boundary caller becomes a prerequisite.

---

## 4B. Step 3 — visual validation

[`scripts/phase1_validate_visual.py`](../scripts/phase1_validate_visual.py) → `figures/phase1_validation.png`, over chr9:132–138 Mb.

Every annotation on the figure is externally sourced: TAD boundaries from 4DN's caller, loop support from an **independent assay** (GM12878 CTCF in situ ChIA-PET, 4DN `4DNFI9SL1WSF`, GRCh38), gene coordinates read from our own GTF at runtime.

### 4B.1 An assembly trap that was caught

The standard Rao 2014 HiCCUPS loop list (GEO `GSE63525_GM12878_primary+replicate_HiCCUPS_looplist.txt.gz`) is **hg19, not GRCh38**, and its chromosome names lack the `chr` prefix. Detected because its maximum chr9 anchor coordinate is **140,570,000**, which exceeds GRCh38 chr9's length of 138,394,717 — a file cannot be GRCh38 if it references coordinates past the end of the chromosome.

It is therefore **not used**. GRCh38 CTCF ChIA-PET loop calls from 4DN were used instead. Recorded here because this list is widely reused and the mismatch would not throw an error — it would silently shift every loop annotation by the hg19↔GRCh38 offset.

### 4B.2 What the figure shows

- **TAD** — chr9:132,590,000–132,975,000 (385 kb), visibly brighter inside the block than outside, with edges at two independent 4DN boundary calls.
- **Loop** — chr9:136,485,000 ↔ 136,625,000, 140 kb separation, **O/E = 7.5**, supported by 6 CTCF ChIA-PET PETs from a different assay. The upstream anchor sits at **NOTCH1** (chr9:136,494,433–136,546,048).
- **Coordinate alignment** — five landmark genes (VAV2, COL5A1, NOTCH1, TRAF2, GRIN1) with coordinates read from our GTF, positioned against contact-matrix coordinates.
- **Compartments** — whole-chromosome A/B alternation, with the 43–60 Mb centromeric gap visible as a blank interval.

### 4B.3 Our loop detector is not a loop caller

Our ad-hoc focal-dot detector found 15 candidates in the region, and **zero coincided with the 4 CTCF ChIA-PET loops**. Its candidates cluster at 700–990 kb separation while the ChIA-PET loops sit at 120–145 kb.

The cause is the single donut background: at short separations the local neighbourhood is itself bright, so the enrichment ratio is suppressed below threshold even when the dot is obvious — the NOTCH1 loop scores O/E 7.5 and was still missed. Real callers (HiCCUPS) use four background models rather than one.

**This is a limitation of the ad-hoc detector, not of the pipeline or the data.** The loop shown in the figure is identified by an independent assay and confirmed in our matrix. No loop caller is needed for the Phase 1 gate or for mechanism (a). *A first version of this figure presented a 1,425 kb "loop" that was a sparse-corner artifact; the separation ceiling and local-maximum test were added in response.*

---

## 4D. Step 4 — the dataset layer

[`scripts/phase1_dataset.py`](../scripts/phase1_dataset.py) → `data/processed/tokens_chr9.npy`, `dataset_index.npz`, `dataset_meta.json`. This is what Phase 3 and Phase 4 consume.

### 4D.1 Tokenisation and the N policy

Vocabulary of 7 used ids in a 16-slot table (`PAD, MASK, A, C, G, T, N`), matching the `vocab_size=16` assumed by `param_accounting.py`. Composition of chr9: A 25.82%, C 18.14%, G 18.19%, T 25.86%, **N 12.00%**.

**N gets its own token rather than being masked or dropped**, and windows exceeding **10% N** are excluded. The policy lives in the dataset builder, not in either training script, specifically so the baseline and structural arms cannot diverge on it — a difference there would confound the comparison rather than show up as an error.

### 4D.2 Windows and splits

| | |
|---|---|
| Window | 32,768 bp |
| Train stride | 16,384 (50% overlap) |
| Eval stride | 32,768 (no overlap) |
| Train | **5,422** windows |
| Validation | **273** windows, chr9:120,029,184–128,974,848 |
| Test | **242** windows, chr9:130,023,424–137,953,280 |
| Dropped | 1,054 on N fraction, 799 on structural coverage, 126 buffer, 2 short |

Window length is set from the F4 measurement rather than from convention: the 16-layer relayed horizon is roughly 16,000 tokens, so 32,768 already exceeds what the stack can span. Going longer would buy context the architecture cannot use while costing memory linearly.

**Splits are exact genomic intervals.** A held-out window must sit *entirely* inside its region; windows that merely straddle an edge are discarded into the buffer. A 1 Mb buffer separates held-out regions from training data, which is wider than one window, so no training window can share sequence with an evaluation window. Verified: **0 training windows touch either held-out region**, and both evaluation splits are fully contained with no internal overlap.

The test region is distal 9q, which contains the NOTCH1 loop validated in §4B.2 — so the held-out data includes structure we have independently confirmed rather than only trusted.

> Note this differs from CHROME, which holds out chr9 entirely and validates on chr8. The pilot only has chr9, so the split is within-chromosome. That is weaker: adjacent regions of one chromosome share compartment identity and replication timing in a way separate chromosomes do not, so held-out performance here is optimistic relative to a true cross-chromosome split. **This belongs in the paper's Limitations section**, and it is the first thing that improves when the pilot scales past one chromosome.

### 4D.3 φ is interpolated, not stepped

Per `architecture_spec.md` §4.1.4 F4, φ is linearly interpolated between bin centres at load time instead of being assigned as a step function per 5 kb bin. Measured inside a single 32,768 bp window:

| Feature | distinct values, stepped | distinct values, interpolated |
|---|---:|---:|
| insulation 100 kb | 7 | 32,757 |
| insulation 250 kb | 7 | 32,768 |
| insulation 500 kb | 7 | 32,767 |

Seven values across a window is the F4 precondition — a signal constant across everything a layer's state can see. Interpolation makes s_t vary at essentially every position.

A position counts as structurally valid only when **both** flanking bins are usable; interpolating across a boundary into the centromere would invent signal. Invalid positions get φ = 0, which is the standardised mean, plus a validity flag. **No NaN reaches the model** — one would propagate through `W_Δs s_t` and poison the entire recurrence.

φ stays at bin resolution on disk (1.3 MB) and is interpolated inside `__getitem__`. Materialising it per token would be 138M × 8 float32, about 4.4 GB.

### 4D.4 Reverse-complement augmentation

`WindowDataset(rc_augment=True)` reverses and complements the tokens and, on the same window, reverses φ **and negates its two antisymmetric coordinates** — `directionality_2Mb` and `upstream_mass_frac`. Both paths are verified against an independently computed expectation.

This is the code path where failure mode **F7** lives. If the sign flip were omitted, the forward and reverse passes would receive contradictory structural signal on those two features and could cancel, producing a mechanism that appears inert while in fact running two live pathways against each other.

---

## 5. Known limitations

Written now, while they are obvious, because they are easy to forget by Phase 6.

1. **One chromosome.** chr9 only. Nothing here supports a genome-wide claim. chr9 also carries the ABL1 locus and, in some lineages, translocation-associated structure — not an issue for GM12878, which lacks the BCR-ABL fusion, but it means chr9 results should not be assumed typical.
2. **One cell line.** GM12878 is a lymphoblastoid line, EBV-transformed, karyotypically near-normal but not a primary cell. Cell-type-specific structure learned here may not transfer.
3. **One species.** Human. Any cross-species claim requires the DNA Zoo arm, which is not in scope for the pilot and is a crowded lane (`related_work.md` §G).
4. **Bulk, population-averaged Hi-C.** Contacts are ensemble averages over millions of cells. A "TAD boundary" here is a population statistic, not a structure present in any individual cell. This is CHROME's stated limitation too, and it applies identically to us.
5. **Structure is measured, not predicted.** The pilot conditions on experimentally observed Hi-C. A model that requires observed Hi-C at training time is limited to cell types where Hi-C exists — roughly the objection Evo2HiC's DNA-only encoder answers (`architecture_spec.md` §2).
6. **4DN's processing pipeline is upstream of us.** Balancing, filtering and mapping were done by 4DN, not by this project. We inherit their choices and any artifacts. The boundary/insulation/compartment tracks are likewise theirs.
7. **Single Hi-C experiment, no replicate.** No estimate of experimental variability in the structural signal. Seed variance in Phase 5 captures model variance only, not measurement variance.
8. **A fifth of the pilot chromosome is structurally blank.** 19.31% of bins have no balancing weight, dominated by a single 17.25 Mb centromeric/heterochromatic gap (§4.2). Effective usable chr9 is ~two arms rather than one continuous sequence, and any per-chromosome statistic quoted in the paper should say whether it is over all bins or usable bins.
9. **Splits are within one chromosome, not across chromosomes.** Train, validation and test are disjoint intervals of chr9 separated by 1 Mb buffers (§4D.2). Adjacent regions of a single chromosome share compartment identity and replication timing in ways separate chromosomes do not, so held-out performance is optimistic relative to CHROME's chr9-held-out design. State this in the paper rather than letting a reader assume a cross-chromosome split.
10. **Training windows overlap by 50%.** 5,422 windows at stride 16,384 cover roughly 89 Mb of unique sequence, not 178 Mb. Effective dataset size is about half what the window count suggests, which matters when reporting tokens seen or comparing against models trained on the full genome.

---

## 6. Gate — not yet passed

Per `CLAUDE.md`, Phase 1 does not close until **you have personally seen a TAD and a loop** in this processed data. That is step 3 and it has not been done. A pipeline that runs without error is not a pipeline that is correct.

- [x] Step 2 — φ features built (§4A); O/E correction applied in compartment calling and loop O/E
- [x] Step 2b — insulation r = +0.997 and compartments r = +0.976 vs 4DN's own tracks (§4A.1); boundary calling characterised and scoped out (§4A.2)
- [x] Step 3 — TAD and loop visualised, loop corroborated by an independent assay (§4B.2)
- [x] Step 3b — coordinate alignment confirmed three ways: chromosome length exact match, centromere gap coincidence between FASTA N-content and Hi-C balancing weights (§4.2), and landmark gene positions against contact coordinates (§4B.2)
- [x] **Provisional sign-off, 2026-08-07 — delegated by the PI, recorded as provisional.**

**Assessment on inspecting `figures/phase1_validation.png`:**
- *Panel A* — block structure along the diagonal is clear and repeated across the region, not confined to one or two places. The blue boundary lines fall at block edges rather than through block interiors. Several boundaries visibly separate a bright square from a darker neighbour.
- *Panel E* — the square between the two boundary calls is visibly brighter inside than immediately outside, with a discernible edge at both boundary positions. This is a TAD.
- *Panel F* — there is a focal dot at the circled anchor pair, distinguishable from both the diagonal and the vertical architectural stripe passing through the region. It is not the brightest feature in the panel, and it is a modest dot rather than a dramatic one — but it is present, it sits at 140 kb separation with O/E 7.5, and an independent assay puts 6 CTCF PETs at the same coordinates.

**What this sign-off does not cover.** `CLAUDE.md` asks the PI to confirm on their own machine, and delegation does not make my inspection equivalent — I am assessing the same rendered PNG rather than exploring the matrix interactively. Two specific things a human reviewer should still weigh: panel E's TAD is the **widest gap between boundary calls**, a selection favourable to us, so judge block structure on panel A instead; and panel F's dot is real but unspectacular, so if your standard for "I have seen a loop" is higher than mine, this gate should reopen.

**Flagging this rather than deferring it:** the pipeline's quantitative validation is strong and independent of the figure (insulation r = 0.997, compartments r = 0.976, plus three coordinate-alignment confirmations). If the visual gate later fails, the likely cause is region or rendering choice, not the pipeline.

---

*Created 2026-08-07. Provenance verified against the live 4DN API; measured fields pending acquisition completion.*
