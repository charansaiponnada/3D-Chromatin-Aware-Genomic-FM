// The paper's text. build_master.js turns this into master_copy.docx.
//
// Inline markup inside strings:
//   [@key1,key2]   numbered citation, resolved against bibliography.json in order
//                  of first appearance; an unverified or unknown key fails the build
//   *text*         italic        **text**  bold
//   $...$          inline maths: _x / _{xy} subscript, ^x / ^{xy} superscript
//   {fig:name}     "Figure n"    {tab:name} "Table n"    {eq:name} "Eq. (n)"
//
// Claim discipline (paper-writing skill, claim-evidence alignment): every
// number in Sections 4-6 is copied from a file on disk, named in the comment
// beside it. The pre-registered decision is results/decision_claims_val.json
// (scripts/decide.py claims --recipe v1); its outcome is "no positive claim",
// and the wording below is the one docs/preregistration.md fixes for it.
//
// Rewritten 2026-10-16 against the code. In the Hi-C-free setting the local
// edges still carry measured strength (chromgraph/graph.py:_local_edges,
// chromgraph/model.py:_edges); the text now says so, and the sequence-only
// re-evaluation (scripts/eval_neutral_local.py) fills the slots marked
// [PENDING NEUTRAL].
//
// The method described is the recipe the pre-registered rule selected (v1,
// configs/base.yaml, commit 660d20f): three objectives, a soft-strength
// cross-entropy contact target, and structure dropout. Structure distillation
// and log(O/E) regression belong to the screened v2 recipe and appear only
// as such (Section 3.8).

const C = {};

C.title =
  "ChromGraphFM: Contact-Biased Attention Lets a DNA Encoder Exploit Measured 3D Chromatin Structure, and a Controlled Test of What Transfers Without It";

C.authors = "P. Charan Sai, G. Karthik, P. Sanath, K. Divya Kothapali";
C.affiliation =
  "Department of Computer Science and Engineering, VR Siddhartha Engineering College, Vijayawada, India";

C.draftNote = [
  "Working master copy, rewritten against the code on 2026-10-16. The pre-registered decision has been made on the validation chromosomes (results/decision_claims_val.json): no positive claim. Sections 4–6 report it in the wording the pre-registration fixes.",
  "Still open: (i) the single test-split evaluation (Section 4.4); (ii) the sequence-only re-evaluation with neutral local-edge strengths ([PENDING NEUTRAL], Section 4.5); (iii) the B2 secondary arm. No other slot is open.",
];

// Abstract: Version 2 of the skill (task, challenge, insight, contribution,
// advantage, experiment summary).
C.abstract = [
  "DNA sequence models contextualise a genomic window by what lies near it on the linear chromosome, yet much of gene regulation acts across hundreds of kilobases through the three-dimensional folding of chromatin. Chromosome conformation capture (Hi-C) measures that folding directly, but existing models use it as a prediction target, as an input that must be present at inference, or as an alignment target applied outside the encoder. We ask whether measured contacts can instead act as a training-time routing signal: if they decide which distal windows exchange information during self-supervised pretraining and are then removed, is the resulting representation better? We implement this in ChromGraphFM, a 7.3 M-parameter DNA encoder in which a sparse graph of distance-detrended Hi-C contacts restricts and biases attention between 5 kb windows across a 640 kb context, with predicted contacts withheld from the graph and whole-graph structure dropout so the same weights also run without distal contacts. Against six matched controls that share its architecture, data and compute, the encoder exploits measured structure strongly and specifically: on held-out chromosomes, its long-range (≥150 kb) contact correlation reaches r = 0.644, against 0.597 when contact strengths are shuffled and 0.175 with a random graph, a 3.7-fold gain from real contacts. We then test, under a protocol fixed before any confirmatory result and on three fresh training seeds, whether this knowledge persists once distal contacts are removed. Here no pre-registered comparison passes: the full model leads a DNA-only control by +0.015 (95% CI −0.004 to +0.035) and a random-graph control by +0.010 (−0.008 to +0.026), and does not differ from a control with shuffled contact strengths (+0.001). Hi-C conditioning thus did not produce a better Hi-C-free representation at this scale, and we trace this to the Hi-C-free receptive field (about ±60 kb), which never connects the windows of a long-range pair. The result locates where structure-aware DNA models must change to transfer what they learn from Hi-C, and we release the model, the seven-arm protocol and all decision files as a benchmark for them.",
];

C.keywords = "3D genome; Hi-C; chromatin contacts; DNA language models; self-supervised learning; graph attention; privileged information; pre-registration; benchmark";

// --------------------------------------------------------------------------- //
// 1 Introduction
// --------------------------------------------------------------------------- //

C.introduction = [
  // [opening] general task -> specific setting (skill: Part A, Version 3)
  "Learning general-purpose representations of DNA sequence has become a central route to predicting the regulatory consequences of the genome, from chromatin accessibility and transcription to the effects of non-coding variants [@dnabert,ntransformer,hyenadna,caduceus,evo,evo2]. A representation is useful to the extent that it captures what controls a locus, and for a large share of human gene regulation that control is distal: enhancers act on promoters tens to hundreds of kilobases away, brought into contact by the folding of chromatin into loops and topologically associating domains [@dixon2012tads,rao2014]. This paper focuses on learning a representation of a genomic window from DNA alone that nevertheless reflects the window's three-dimensional context.",

  // [challenge 1] sequence models: limitation + technical reason
  "This setting is challenging because three-dimensional proximity is not a property of linear proximity. Sequence models enlarge their receptive field with convolutions and dilations [@basenji], attention [@enformer,alphagenome], or long-range state-space and implicit-convolution layers [@hyenadna,caduceus,evo]; however, all of them weigh context by position along the chromosome, and must infer from sequence alone which of thousands of distal windows are physically close. Folding is partly encoded in sequence, through features such as CTCF motifs and their orientation [@rao2014,fudenberg2016], but it is cell-type-specific and shaped by factors a sequence model never observes, so the relevant distal context is diluted among the irrelevant.",

  // [challenge 2] Hi-C methods: limitation + root technical reason
  "Chromosome conformation capture measures this folding directly. Hi-C reports, for every pair of genomic bins, how often the two were found in contact [@liebermanaiden2009,rao2014], and a growing body of work learns from it. Sequence-to-structure models predict contact maps from DNA [@akita,orca,corigami,chimaera]; foundation models are pretrained on contact maps themselves [@hicfoundation,mixhic]; graph and attention models feed contact frequencies into supervised predictors of gene expression [@graphreg,chromoformer,chrome,puget]; and Evo2HiC aligns a compact sequence encoder to a Hi-C encoder with a contrastive objective [@evo2hic]. In each design, however, measured contacts never decide how a sequence encoder contextualises its input during self-supervised pretraining: Hi-C is the output, an input that a supervised model needs at prediction time, or an alignment target applied from outside the encoder. Two technical obstacles explain this gap. First, an encoder whose attention is driven by contacts can satisfy any contact objective by reading the contacts back from its input, so the objective measures nothing. Second, a representation that depends on Hi-C at inference cannot be used for the great majority of sequences and cell types, for which no contact map exists.",

  // [method] contribution 1 (skill: Pipeline Version 2)
  "In this paper, we propose ChromGraphFM, a DNA encoder in which measured chromatin contacts shape sequence contextualisation during pretraining and are not required afterwards. The basic idea is illustrated in {fig:overview}. Our first contribution is contact-biased sparse attention: 5 kb sequence windows become the nodes of a graph whose edges link each window to its strongest distance-detrended Hi-C partners, attention between windows runs over these edges only, and each edge's measured strength and genomic separation enter its attention logit through a learned bias. A fraction of the strongest contacts is withheld from this graph and used only as prediction targets, so the encoder cannot meet its contact objective by copying its input and must instead learn which sequence windows interact.",

  // [method] remaining challenge -> contribution 2
  "Contact-biased attention by itself yields a model that expects Hi-C at inference. Our second contribution removes this dependence by treating Hi-C as privileged, training-time-only information [@vapnik2009lupi]. Structure dropout removes the entire distal graph from a random 30% of training samples, so that one set of weights is trained both with and without contacts, and the Hi-C-free mode, in which every model is evaluated, is never out of distribution. Because the attention blocks, heads and objectives are shared by the two modes, whatever the contacts teach the encoder is learned in weights that the Hi-C-free mode also uses.",

  // [evidence design] contribution 3, no outcome claims
  "Whether such a model has learned anything about structure is easy to assert and hard to test, because a richer input, more parameters, knowledge of genomic distance, or a favourable choice among many runs can each masquerade as structural learning. Our third contribution is therefore an evaluation protocol designed to rule these out. Every model is evaluated with Hi-C removed at inference and compared against controls that share its architecture, data, parameter count and compute while removing the distal graph, replacing it with random edges, keeping its edges but flattening or shuffling their strengths, fusing Hi-C after the encoder, or using Hi-C only as a contrastive alignment target. Data are split by chromosome, a frozen-representation probe gives every arm the same read-out, and the claims, the statistic and the decision rule were fixed in writing before any confirmatory run was evaluated, with every possible outcome mapped in advance to the claim it supports.",

  // [evidence] outcome paragraph
  "Contact-biased attention works as designed. Measured contacts are used, and used specifically: with the graph present, the full model's long-range contact correlation on held-out chromosomes is 0.644, against 0.597 when strengths are shuffled among the measured edges and 0.175 when the edges are random. Yet with the distal graph removed, every arm falls into the same narrow band, including a DNA-only control that never saw a contact graph, and none of the pre-registered comparisons passes, so under matched budgets Hi-C conditioning during pretraining did not produce a better Hi-C-free representation. An analysis of the model explains why: without distal edges a window gathers information from at most about ±60 kb, so the two windows of a long-range pair never exchange information, and Hi-C can reach the Hi-C-free representation only through per-window features.",

  "In summary, our contributions are as follows:",
];

C.contributions = [
  "Contact-biased sparse attention, in which distance-detrended Hi-C contacts determine which 5 kb sequence windows attend to one another across a 640 kb context and bias how strongly, with held-out target edges that keep the contact objective from being solved by copying.",
  "A training-time-only use of Hi-C through whole-graph structure dropout, which trains one encoder to operate with and without contacts and yields DNA representations computed from sequence alone at inference.",
  "A pre-registered, controlled evaluation protocol with six matched controls, chromosome-level splits, a Hi-C-free headline setting and a shared frozen-representation probe, which separates the contribution of measured contacts from those of graph regularisation, genomic distance, contact-strength marginals, post-hoc fusion and run selection.",
  "Evidence, across 19 trained models and seven arms, that contact-conditioned attention exploits measured contacts strongly (r = 0.644 against 0.175 for a random graph), together with a pre-registered test showing where this knowledge stops transferring and an architectural explanation for it: a Hi-C-free receptive field far shorter than the contacts being predicted.",
];

// --------------------------------------------------------------------------- //
// 2 Related work
// --------------------------------------------------------------------------- //

C.related = [
  {
    heading: "Predicting chromatin structure from sequence",
    paras: [
      "A first line of work treats three-dimensional structure as the output of a sequence model. Akita predicts Hi-C maps over megabase windows from DNA sequence with a convolutional network [@akita], Orca extends sequence-to-structure prediction across multiple scales up to whole chromosomes [@orca], and C.Origami adds CTCF binding and chromatin accessibility to predict cell-type-specific maps [@corigami]; EPCOT predicts chromatin contacts alongside epigenomic and transcriptional tracks in a pre-training and fine-tuning framework [@epcot]. These models establish that folding is substantially predictable from sequence. However, Hi-C is the training target rather than a signal that shapes how the encoder contextualises its input, and the learned features are specialised to the map they reproduce rather than offered as a general sequence representation. ChromGraphFM inverts this relation: measured contacts are an input to the encoder during pretraining, and the product is a sequence representation rather than a contact map.",
    ],
  },
  {
    heading: "DNA language and sequence foundation models",
    paras: [
      "Self-supervised DNA language models learn general sequence representations from unlabelled genomes. DNABERT and DNABERT-2 adapt masked language modelling to k-mer and byte-pair tokens [@dnabert,dnabert2], and the Nucleotide Transformer scales this recipe across thousands of genomes [@ntransformer]. To reach longer contexts, HyenaDNA replaces attention with implicit long convolutions [@hyenadna], Caduceus builds bidirectional, reverse-complement-equivariant state-space models [@caduceus] on structured state-space layers [@s4,s4d,mamba], and Evo and Evo 2 train very large autoregressive models over hundreds of kilobases to a megabase [@evo,evo2]. Supervised sequence-to-function models such as Basenji, Enformer and AlphaGenome reach long contexts through dilated convolutions and attention [@basenji,enformer,alphagenome]. Across this line, context is weighted by linear position, and distal windows must be identified from sequence alone. Our encoder uses an ordinary bidirectional state-space model within each window and leaves long-range context to a graph defined by measured contacts.",
    ],
  },
  {
    heading: "Learning from Hi-C contact maps",
    paras: [
      "A third line learns representations of Hi-C itself. HiCFoundation pretrains a foundation model on a large corpus of contact maps [@hicfoundation], and MIX-HIC jointly pretrains on contact maps and epigenomic tracks [@mixhic]; in both, the reusable representation is of contacts rather than of DNA sequence. Chimaera links the latent space of a contact-map autoencoder to a DNA encoder to predict Hi-C from sequence across species [@chimaera]. GraphReg uses Hi-C and related 3D assays as the graph of a graph-attention network that predicts gene expression from epigenomic and sequence features [@graphreg], and CHROME and Puget likewise combine sequence representations with contact graphs or a Hi-C encoder to predict expression [@chrome,puget]. These works show that contact graphs carry information that linear context lacks, but in supervised settings where Hi-C is required at prediction time. Closest to our aim, and concurrent with it, Evo2HiC distils a frozen Evo 2 model into a compact sequence encoder that is aligned to a Hi-C encoder with a contrastive objective, and offers a sequence-only encoder for use without Hi-C [@evo2hic]. It shares our view of Hi-C as a training-time signal, but contacts act there as an alignment target applied from outside the encoder rather than inside it. ChromGraphFM differs in where Hi-C acts: contacts select and weight the information each window receives at every graph layer during pretraining, and structure dropout removes the dependence on Hi-C at inference. Our controls include both a late-fusion and an alignment-only variant (B4 and B5), so that this difference is tested rather than asserted.",
    ],
  },
  {
    heading: "Structural bias in attention and training-time-only signals",
    paras: [
      "Our attention mechanism belongs to a family that restricts or biases attention with a known structure. Graph attention networks compute attention over graph neighbourhoods [@gat], and Graphormer adds learned structural biases, such as shortest-path distances, directly to the attention logits of a Transformer [@graphormer,transformer]. In genomics, Chromoformer adds normalised Hi-C interaction frequencies to the self-attention scores of a supervised model that predicts expression from histone modifications around promoters [@chromoformer], a direct precedent for biasing attention with contact frequency. We follow the additive-bias form, but apply it during self-supervised pretraining of a DNA sequence encoder, with the bias computed from measured contact strength and genomic separation by separate learned functions, a separation that makes a distance-only control constructible. Removing structure during training to obtain robustness has precedents in DropEdge, which drops random edges as a regulariser [@dropedge], and in modality dropout, which drops whole input modalities so a model remains usable when one is missing [@moddrop]; our structure dropout removes the whole distal graph per sample for the second reason. Learning using privileged information formalises a signal available at training time only [@vapnik2009lupi]. A natural extension, distilling the Hi-C-conditioned pass into the Hi-C-free one [@hinton2015distill], was screened as an alternative recipe and not selected by our pre-registered rule (Section 3.8).",
    ],
  },
];

// --------------------------------------------------------------------------- //
// 3 Methodology
// --------------------------------------------------------------------------- //

C.method = [
  {
    heading: "Overview",
    paras: [
      "ChromGraphFM takes $N = 128$ consecutive 5 kb windows of DNA, spanning 640 kb, and produces one representation $z_i$ per window ({fig:overview}). It is organised as two input streams that meet in a contact-conditioned attention stage.",
      "*Sequence stream.* Each 5 kb window is encoded independently, by a convolutional tower followed by a bidirectional state-space encoder, into one vector $h_i$ (Section 3.4). No information crosses windows at this stage.",
      "*Structure stream (training only).* The same windows become the nodes of a graph. Each window is connected to its neighbours within ±3 windows and to its 16 strongest distal contact partners at least 25 kb away, where contact strength is first corrected for genomic distance (Sections 3.2 and 3.3). Of the distal edges, 15% are withheld from the graph and used only as prediction targets. This stream is not a second encoder whose output is merged with the sequence stream; Hi-C acts only by steering attention.",
      "*Contact-conditioned attention.* The model then decides how much each window should use information from each other window. In a standard Transformer this depends only on learned sequence features. Here, attention runs only along graph edges, and each edge's measured contact strength and genomic separation add a learned bias to its attention logit, so Hi-C decides both which windows exchange information and how strongly (Section 3.5).",
      "*Hi-C-free use.* Structure dropout trains the same weights to operate with the distal graph removed (Section 3.6), so distal contacts are needed only during training. As implemented, the local edges of this Hi-C-free mode still carry the measured strength of the adjacent-window pixels (Section 3.3); Section 4.5 reports a re-evaluation in which these are replaced by the neutral 0.5, so that the model sees DNA alone. This setting matters for two reasons: contact maps exist for few cell types and for no user-supplied sequence, and comparing models with Hi-C removed is the only setting in which a gain can be attributed to what training stored in the weights rather than to information supplied at inference. Three objectives train the model end to end (Section 3.7); Section 3.8 defines the controls, the recipe selection and the pre-registered evaluation protocol, and Section 3.9 gives implementation details.",
    ],
    figure: "overview",
  },
  {
    heading: "Data and contact processing",
    paras: [
      "We use the GM12878 lymphoblastoid in situ Hi-C map (MboI; 72 combined experiments) [@rao2014], obtained from the 4D Nucleome Data Portal [@dekker4dn,4dn] (experiment set 4DNES3JX38V5, file 4DNFIXP4QG5B) as an iteratively corrected [@imakaev2012ice] multi-resolution cooler file and used at 5 kb resolution, together with the hg38 reference genome. Each 5 kb bin is one window: its sequence is stored as a 5,000-symbol string over {A, C, G, T, N}, and it is marked usable when at most half of its bases are N and its balanced coverage is non-zero. Contacts involving an unusable bin are discarded.",
      "Raw contact frequency is dominated by genomic distance ({fig:detrend}a): two bins 10 kb apart touch far more often than two bins 500 kb apart, irrespective of any specific interaction. Selecting contacts on raw frequency would therefore return nearest neighbours and teach the model distance rather than structure. We detrend each balanced contact $M_{ij}$ by the expected contact at its separation $s = |i - j|$, and map the observed-over-expected ratio to a bounded strength:",
    ],
    eqs: ["expected"],
    parasAfter: [
      "where $V_s$ is the set of all bin pairs at separation $s$ whose two bins are both usable. Counting unobserved but valid pairs as zero contacts is essential: averaging only the pixels that happen to be observed overestimates $E(s)$ at long range, where the map is sparse, and biases the strength downward precisely at the separations that matter most. With this denominator the strength is centred on $c_{ij} = 0.5$, which corresponds to $\\mathrm{O/E} = 1$, at every separation ({fig:detrend}b), so $c_{ij} > 0.5$ marks contact enriched beyond what distance alone predicts.",
    ],
    figure: "detrend",
  },
  {
    heading: "Sparse contact graph with held-out targets",
    paras: [
      "A training sample is a run of $N = 128$ consecutive windows with at least 60% usable windows, and its graph has two kinds of edge ({fig:sample}a, b). *Local edges* connect each window to its neighbours within ±3 windows and are always present; they carry the measured strength where a pixel exists and a neutral 0.5 otherwise, so the sequential backbone is never broken by a missing measurement. *Distal edges* connect windows at least 5 windows (25 kb) apart: for each window we keep its $k = 16$ partners of highest detrended strength, and an edge survives if either endpoint selects it. Selecting on detrended rather than raw strength is what makes the graph structural rather than a proxy for distance.",
      "From the distal edges of each sample, 15% are withheld as *target edges*. Targets never enter the graph the encoder attends over; they are used only by the contact and contrastive objectives (Section 3.7). Without this separation, the contact objective could be solved by reading an edge's strength from the attention input and echoing it back, and its value would measure nothing. The split is drawn from a fixed control seed, independent of the training seed, so that every model and every training seed is asked about the identical set of targets.",
    ],
    figure: "sample",
  },
  {
    heading: "Local sequence encoder",
    paras: [
      "Each window is encoded independently into $h_i \\in \\mathbb{R}^{d}$, with $d = 256$, so that no information crosses windows before the graph layers. The one-hot sequence first passes through a convolutional tower: a width-15 stem followed by five residual blocks, each consisting of batch normalisation, GELU, a width-5 convolution, batch normalisation, GELU and a pointwise convolution, and each followed by max-pooling by two. The tower detects local sequence motifs and reduces the 5,000 positions to 157, which is where most of the computational saving of the model comes from. Six bidirectional state-space layers then mix information along the window. Each direction is a diagonal state-space model [@s4,s4d] evaluated exactly as a long convolution in the frequency domain:",
    ],
    eqs: ["ssm"],
    parasAfter: [
      "where $u$ is the input sequence of one channel, $\\mathcal{F}$ the real FFT with zero padding to twice the length, and $\\rho_n$ a learned log-rate that keeps every decay $a_n$ strictly inside $(0, 1)$, so the kernel cannot diverge. Each layer applies two independent state-space models, one to the sequence and one to its reversal (whose output is reversed back), concatenates the two, projects back to $d$ channels, and gates the result with a sigmoid of the layer input inside a pre-normalised residual connection; the bidirectional form reflects that a double-stranded sequence has no reading direction. The window embedding $h_i$ is the layer-normalised mean over positions. Evaluating the state-space model by FFT is exact, stable for any value the optimiser reaches, and requires no custom kernel.",
    ],
  },
  {
    heading: "Contact-biased attention",
    paras: [
      "The core of the model is a stack of $L = 4$ attention blocks in which a window $i$ attends only to its graph neighbourhood $\\mathcal{N}(i)$, the union of its local and distal edges, with edges used in both directions. Measured structure enters through an additive per-head bias computed from each edge's strength $c_{ij}$ and separation $d_{ij}$:",
    ],
    eqs: ["bias", "attention", "block"],
    parasAfter: [
      "where $q$, $k$ and $v$ are linear projections of the current window representations split into $H = 8$ heads of size $d_h = 32$, $\\mathrm{LN}$ is layer normalisation, the feed-forward network has width 1,024, and the blocks start from $z_i = h_i$. The bias has two consequences that motivate its form. First, contacts decide *which* distal windows exchange information, because attention exists only on graph edges, and *how strongly*, because a stronger contact raises the logit of its edge relative to the window's other neighbours. Second, keeping the contact and separation terms as separate functions means that fixing the contact input to a constant leaves a working distance-only model, which is the basis of the distance-only control in Section 3.8. Attention is computed edge-wise with a scatter softmax, so its cost is linear in the number of edges and no dense $N \\times N$ matrix is formed; this is what makes a 640 kb context of 5 kb windows affordable.",
    ],
  },
  {
    heading: "Hi-C-free representations through structure dropout",
    paras: [
      "Hi-C is unavailable for most cell types and for any sequence a downstream user supplies, so a representation that requires it at inference has limited use. We therefore treat Hi-C as privileged information [@vapnik2009lupi] and train the model to produce its representation without it as well. *Structure dropout* removes the entire distal graph for each training sample independently with probability $p = 0.3$, leaving only local edges:",
    ],
    eqs: ["dropout"],
    parasAfter: [
      "Removing the whole graph rather than individual edges [@dropedge] reproduces the inference condition exactly, where every distal edge is absent at once [@moddrop]; edge-level dropout would only ever show the model a thinned graph and leave the fully Hi-C-free mode out of distribution. Local edges are never dropped, so the sequential backbone, and with it the model's knowledge of linear neighbourhood, is identical in both modes. Because one set of encoder weights, attention blocks and heads serves both modes, nothing is added at inference: the model is simply run in the mode that dropout trains. In that mode the local edges keep the measured strength of adjacent windows (up to 15 kb apart), so it is free of distal Hi-C but not of all Hi-C; Section 4.5 removes this remaining input at evaluation time.",
    ],
  },
  {
    heading: "Pretraining objectives",
    flow: [
      { p: "Three objectives train the model. The masked-window term is self-supervised with respect to the sequence; the contrastive and contact terms use the held-out Hi-C target edges as training-time-only supervision. No task labels are used." },
      { p: "*Masked-window reconstruction.* For 15% of windows, chosen at random, $h_i$ is replaced by a learned mask token before the attention blocks, and a two-layer network $g$ must reconstruct the window's original embedding from its output $z_i$. A whole window, not individual bases, is masked: a pooled 256-dimensional vector cannot carry 5,000 bases, but it can be inferred from the window's graph neighbours, which is the capacity we want to train. With $\\mathcal{M}$ the set of masked usable windows and $\\mathrm{sg}$ the stop-gradient," },
      { eq: "l_mask" },
      { p: "*Distance-matched contrastive alignment.* Each held-out target edge $(i, j) \\in \\mathcal{T}$ is a positive pair, contrasted against up to 16 negative pairs $\\mathcal{D}_{ij}$ drawn from the same sample at exactly the same separation, with an InfoNCE loss [@cpc,simclr] on projected, $\\ell_2$-normalised representations $\\tilde z$ and temperature $\\tau = 0.1$. Matching the separation of negatives to that of the positive removes the shortcut of judging contact by distance, which the model can already read from its input. Every held-out positive contributes, rather than one sampled positive per sample:" },
      { eq: "l_con" },
      { p: "*Contact prediction.* A symmetric head $f$ predicts each target edge's strength from the two window representations and their separation, and is trained with binary cross-entropy against the measured strength $c_{ij} \\in (0, 1)$ as a soft target. Because the targets are soft, the loss is bounded below by their entropy rather than by zero, so its absolute value is not a measure of quality; we therefore evaluate contact prediction by correlation (Section 3.8), not by loss:" },
      { eq: "l_contact" },
      { p: "The total objective is" },
      { eq: "l_total" },
      { p: "with $\\lambda_{\\mathrm{mask}} = 1$, $\\lambda_{\\mathrm{con}} = 0.5$ and $\\lambda_{\\mathrm{contact}} = 1$. Control arms that have no structural signal train only the terms their signal defines ({tab:arms})." },
    ],
  },
  {
    heading: "Controlled experimental design and pre-registered protocol",
    paras: [
      "The claim that measured contacts improve a sequence representation is only meaningful against controls that remove everything else a graph-conditioned model has. We therefore train six control arms ({tab:arms}) through the same code path as the full model, with the same architecture and parameter count, the same training windows, the same number of optimisation steps, the same pretraining recipe and the same held-out targets; only the structural signal differs. Each corruption is drawn from the fixed control seed, so every training seed sees the identical corrupted graph and seed variance cannot be confused with corruption variance. The arms are designed to be read in pairs: B0 against the full model asks whether structure helps at all; B1 asks whether any graph regularisation would do; B2 isolates genomic distance; B3, which keeps the measured topology but permutes strengths among edges, isolates the pairing of strength to edge; B4 asks whether conditioning inside the encoder is better than adding Hi-C features after it; and B5 asks whether Hi-C as a contrastive alignment target alone suffices.",
      "*Splits.* Hi-C is autocorrelated over megabases, so random window splits would leak. We split by chromosome: chr3, chr13 and chr17 form the test set, chr10 and chr11 the validation set, and the remaining 17 autosomes the training set (6,284 training samples at a stride of half a sample). All model and recipe decisions are made on the validation chromosomes; the test chromosomes are scored once, for the final table. One exception is disclosed: an evaluation bug, since fixed, read the test split for the full model and B3 at exploratory seed 0, so those two seed-0 test numbers are reported only as exploratory.",
      "*Headline setting and metric.* Every arm is evaluated with the distal graph removed at inference (the Hi-C-free setting; local edges as in Section 3.3), so that the only difference between arms is what shaped their weights during pretraining. Results with the graph present are reported separately as an upper bound with privileged input and are never compared with a sequence-only baseline. The headline metric is the Pearson correlation between predicted and measured strength on held-out target edges, stratified by separation into short (<50 kb), medium (50–150 kb) and long (≥150 kb) bands, because a single correlation is dominated by short-range contacts; the long-range band is the primary endpoint.",
      "*Two read-outs.* The *head* read-out uses each model's own contact head, and is valid only for arms that train one. The *probe* read-out fits a fresh contact head of the same architecture on frozen Hi-C-free representations, with the same training pairs, epochs and probe seed for every arm. The probe is the only valid read-out for B0 and B5, whose pretraining trains no contact head, and it compares all arms on representation quality alone, independently of how well each trained its own head.",
      "*Recipe selection (exploratory).* Two pretraining recipes were screened on training seed 0 before confirmation: the recipe described above, and a variant that regressed log(O/E) with a squared error instead of the soft-target cross-entropy and added a structure-distillation term pulling the Hi-C-free pass toward the Hi-C-conditioned one [@hinton2015distill]. A rule fixed in advance selected the variant only if, on the validation chromosomes, it raised the full model's score and gave a margin over B3 whose 95% interval excluded zero and exceeded the original recipe's margin; otherwise the recipe above was kept. The rule selected the recipe above, which was then frozen for every arm. Seed 0 is therefore exploratory for every arm and is never used as confirmatory evidence.",
      "*Pre-registered decision rule.* The claims, the statistic and the decision rule were fixed in writing before any confirmatory run was evaluated. Confirmation uses fresh seeds 1–3 for the full model, B3, B0 and B1, at 20,000 steps each, on the validation chromosomes. For a comparison X − Y, 10,000 paired bootstrap resamples of validation samples give a 95% interval for the seed-averaged difference, and X is said to beat Y only if the interval lies above zero and X − Y > 0 in at least two of the three seeds. Claim A, that measured strengths matter, requires the full model to beat B3 on the head read-out. Claim B, that measured topology matters, requires both the full model and B3 to beat both B0 and B1 on the probe read-out; because all four comparisons must pass, no multiple-comparison correction is needed. Every combination of outcomes maps to a pre-specified statement, including a controlled negative result. B2, B4 and B5 and the distance strata other than long range are reported as secondary comparisons, whatever the outcome.",
    ],
    table: "arms",
  },
  {
    heading: "Implementation details",
    paras: [
      "The model has 7.3 M parameters: 3.3 M in the local encoder, 3.2 M in the attention blocks and the remainder in the heads. Models are trained for 20,000 steps with AdamW (learning rate $3 \\times 10^{-4}$, weight decay 0.01), 1,000 warm-up steps followed by cosine decay, gradient clipping at 1.0 and bfloat16 mixed precision. Each step accumulates gradients over four micro-batches of two samples, so one step sees 8 samples, or 1,024 windows (5.1 Mb of sequence), and a run sees each training sample about 25 times. Training uses one NVIDIA L40S GPU per run, with a peak memory of 14.3 GiB. An update whose gradient norm is not finite is skipped rather than applied, and no checkpoint with non-finite weights is saved; skipped updates are logged and reported per run. Data are built reproducibly from a manifest that pins the source accessions and checksums, and every run records its resolved configuration and code commit before its first step. {tab:hparams} lists the hyperparameters.",
    ],
    table: "hparams",
  },
];

// --------------------------------------------------------------------------- //
// 4-6 Results, Discussion, Conclusion
// --------------------------------------------------------------------------- //

// Numbers: results/decision_claims_val.json (decision, per-seed r, CIs);
// results/<arm>_seed<N>/metrics.json (head, with_hic) and probe_metrics.json
// (probe). Means over seeds 1-3 are arithmetic means of those per-seed values.
C.results = [
  {
    heading: "Experimental setup",
    paras: [
      "All confirmatory runs completed: the full model, B3, B0 and B1, each at training seeds 1, 2 and 3, for 20,000 steps from the frozen recipe of Section 3.8. Every confirmatory run used the same model, loss and configuration code. No run logged a skipped update. Runs interrupted by server restarts were resumed from their last checkpoint and never replaced by a new seed. The secondary arms B4 and B5 were trained at seed 0. All numbers below are on the validation chromosomes (chr10, chr11; 406 samples), in the Hi-C-free setting unless stated otherwise, and use the long-range band (≥150 kb) as the primary endpoint.",
    ],
  },
  {
    heading: "Claim A: do measured contact strengths matter?",
    paras: [
      "On its own contact head, the full model scored r = 0.188, 0.206 and 0.204 on seeds 1–3, against 0.188, 0.196 and 0.211 for B3, which keeps the measured topology but shuffles strengths among edges. The seed-averaged difference is +0.0008 (95% CI −0.0060 to +0.0078), and the full model was ahead on one seed of three. Claim A is not supported ({tab:decision}).",
    ],
  },
  {
    heading: "Claim B: does measured contact topology matter?",
    paras: [
      "On the frozen probe, which gives every arm the same read-out, the four comparisons all point in the predicted direction but none passes the rule ({tab:decision}, {fig:results}a). The full model leads the DNA-only control B0 on all three seeds, by +0.0154 on average, but the interval includes zero (−0.0040 to +0.0348). Its lead over the random-graph control B1 is +0.0098 (−0.0075 to +0.0264; two seeds of three). B3 leads B0 by +0.0080 (−0.0092 to +0.0255; two of three) and B1 by +0.0024 (−0.0154 to +0.0195; one of three). Claim B is not supported.",
      "The seed-averaged long-range probe correlations are 0.175 (full), 0.168 (B3), 0.166 (B1) and 0.160 (B0). The spread of one arm across seeds is larger than the gaps between arms: B1 ranges from 0.145 to 0.203, and B3 from 0.140 to 0.197. Across all separations the four arms are indistinguishable (seed-averaged overall probe r of 0.311, 0.306, 0.310 and 0.309), and at short range (<50 kb) the full model ranks last (0.290 against 0.298–0.307). With both claims failing, the pre-registered outcome is: no positive claim. Under matched budgets, Hi-C conditioning during pretraining did not produce a better Hi-C-free representation.",
    ],
    table: "decision",
    figure: "results",
  },
  {
    heading: "Test-split confirmation",
    paras: [
      "[PENDING TEST: the single evaluation of seeds 1–3 on chr3, chr13 and chr17, with the same statistic, reported as measured. The seed-0 test numbers read by the earlier evaluation bug (Section 3.8) remain exploratory.]",
    ],
  },
  {
    heading: "Secondary analyses",
    paras: [
      "*With measured contacts present.* This comparison was not pre-registered as a claim and is reported as exploratory, without an interval. Run with its distal graph, the same full-model weights reach a long-range r of 0.670, 0.647 and 0.615 on seeds 1–3 (mean 0.644), against 0.597 for B3 and 0.175 for B1 ({fig:results}b). The model therefore exploits measured contacts strongly, and their specific strengths add to their topology when they are supplied; what fails is the transfer of this ability to the Hi-C-free mode.",
      "{tab:allruns} lists every trained model and every read-out, so that no number is selected.",
      "*Secondary arms (seed 0, exploratory).* B4, which adds per-window Hi-C summaries after the encoder instead of steering attention, reaches 0.826 with Hi-C and 0.192 on the Hi-C-free probe; B5, which uses Hi-C only as a contrastive target, reaches 0.167 on the probe. Both lie inside the band of the confirmatory arms without Hi-C. B2 (distance-only): [PENDING B2].",
      "*Sequence-only re-evaluation.* In the Hi-C-free setting as run, local edges carry the measured strength of adjacent-window pixels (Section 3.3). Every checkpoint was therefore re-evaluated, with no change to its weights, after replacing each local strength by the neutral 0.5 (scripts/eval_neutral_local.py). [PENDING NEUTRAL: head and probe long-range r per arm, and whether the ranking or the decision would change.]",
    ],
    table: "allruns",
  },
];

C.discussion = [
  "The study set out to test whether measured 3D contacts, used to route information inside a DNA encoder during pretraining and then removed, leave a better sequence representation behind. Under the pre-registered rule they did not: no comparison between the full model and its controls passed, and the full model did not differ from a control whose contact strengths were shuffled. The estimated gains over the DNA-only and random-graph controls, about +0.01 to +0.015 in long-range correlation, are smaller than the variation of a single arm across training seeds, so three seeds cannot separate them from zero. If an effect of that size is real, it is small.",
  "The negative result is not a failure of the mechanism. With its distal graph present, the same network predicts held-out long-range contacts at r = 0.644, far above a random graph (0.175), and above shuffled strengths (0.597). Contact-conditioned attention therefore learns to use measured structure, and to use its strengths. What does not happen is the transfer that structure dropout was meant to produce. The architecture suggests why. Without distal edges, each window reaches only ±3 neighbours per block, about ±60 kb over four blocks, while the primary endpoint concerns pairs at least 150 kb apart. In the Hi-C-free mode the two windows of such a pair never exchange information, and the contact head can only compare two independently computed window summaries. Hi-C can then affect the Hi-C-free representation only by changing what each window encodes about itself, and the DNA-only control, which reaches nearly the same score, shows how little that adds at this scale. The late-fusion arm points the same way: adding Hi-C after the encoder gives the best result with Hi-C (0.826) and no advantage without it.",
  "Two practical lessons follow. First, \"train with Hi-C, infer without it\" is not obtained for free by structure dropout: a model that routes information along measured contacts learns to rely on them. Transfer probably needs a Hi-C-free pathway that can itself connect distal windows, for example a sequence-predicted contact graph or a longer-range sequence model, so that what the conditioned pass learns has somewhere to go. Second, small gains in this setting are easily mistaken for structural learning. Matched controls, a shared probe and a rule fixed in advance were what separated a consistent +0.015 trend from a supported claim.",
  "*Scope and limitations.* The study is confined to one cell line (GM12878), one Hi-C resolution (5 kb), one context length (640 kb), one model scale (7.3 M parameters) and three confirmatory seeds, and its read-outs are contact-prediction tasks on held-out chromosomes. The negative result holds for this configuration; a larger model, base-pair resolution or more seeds could detect an effect too small to resolve here. The model is compared with matched controls of itself, not with independently trained sequence or structure models, and downstream regulatory tasks are not evaluated. As implemented, the Hi-C-free setting retained measured local-edge strengths; Section 4.5 reports the fully sequence-only re-evaluation.",
];

C.conclusion = [
  "We asked whether measured chromatin contacts, used to steer a self-supervised DNA encoder's attention during pretraining and then removed, improve its sequence-only representation. ChromGraphFM uses measured contacts strongly when they are present, but under a pre-registered protocol with matched controls, the Hi-C-free representation was not measurably better than that of a DNA-only or a random-graph control. The likely reason is architectural: without the contact graph, the encoder cannot connect the distal windows whose contacts it is asked to predict. A Hi-C-free pathway with its own long-range reach, such as a contact graph predicted from sequence, is the natural next test. The code, the controls and the pre-registered decision files are released so that future structure-aware DNA models can be held to the same protocol.",
];

C.figures = {
  overview: {
    file: "figures/fig1_overview.png",
    width: 6.3,
    caption:
      "**Overview of ChromGraphFM.** Top: each 5 kb DNA window is encoded independently by a convolutional tower and a bidirectional state-space model into $h_i$. Bottom: distance-detrended Hi-C contacts define a sparse graph (local and top-k distal edges), from which a fraction of distal edges is withheld as targets. Middle: contact-biased attention propagates information over graph edges only, with the measured strength and separation entering the attention logit through a learned edge bias. Structure dropout removes the whole distal graph for a fraction of training samples, so the same blocks also run on local edges alone (dashed), which is the Hi-C-free mode used at inference. Hi-C is used only during training.",
  },
  detrend: {
    file: "figures/fig2_detrending.png",
    width: 6.3,
    caption:
      "**Distance detrending of GM12878 Hi-C at 5 kb.** (a) The expected balanced contact $P(s)$ decays by roughly two orders of magnitude between 5 kb and 500 kb on every chromosome, so raw frequency is dominated by separation. (b) After detrending, the strength $c_{ij}$ of both near (<25 kb) and distal (≥25 kb) pixels is centred on 0.5 ($\\mathrm{O/E} = 1$; dashed), and the top-k distal edges selected for the graph form its enriched upper tail. Panel b shows chr20 and the edges of the sample in {fig:sample}.",
  },
  sample: {
    file: "figures/fig3_graph_sample.png",
    width: 6.3,
    caption:
      "**The contact graph of one real training sample** (GM12878 chr20, 128 windows, 640 kb), produced by the same code used in training. (a) Above the diagonal, the measured detrended strength; below it, what the encoder conditions on: local edges (blue band), distal edges coloured by strength, and held-out target edges (orange), which are never attended over. (b) The same distal edges (top) and targets (bottom) drawn as arcs. (c) The distal edges each arm conditions on, with the fraction of edges shared with the full model's topology and the correlation between the strengths the arm receives and the measured strengths. B3 retains the measured topology and destroys the pairing of strengths to edges; B2 retains the topology with constant strength; B1 replaces the topology itself.",
  },
  results: {
    file: "figures/fig4_results.png",
    width: 6.3,
    caption:
      "**Confirmatory results on the validation chromosomes (long range, ≥150 kb; seeds 1–3).** (a) Hi-C-free frozen-probe correlation for each arm; dots are seeds, bars are seed means. The arms differ by less than one arm varies across seeds, and no pre-registered comparison passes. (b) The same weights evaluated with their own contact head without (open) and with (filled) the measured distal graph at inference. Measured contacts are exploited strongly when present (full model, mean 0.644) but the gain does not carry into the Hi-C-free mode. Values from results/<arm>_seed<N>/metrics.json and probe_metrics.json.",
  },
};

C.tables = {
  arms: {
    caption:
      "**Matched experimental arms.** All arms share architecture, parameter count, training data, steps and held-out targets, and are evaluated with Hi-C removed at inference. Full, B3, B0 and B1 are confirmatory; B2, B4 and B5 are secondary.",
    widths: [1300, 2500, 5226],
    header: ["Arm", "Structural signal during pretraining", "Question it answers"],
    rows: [
      ["Full", "Measured detrended contacts bias attention over the top-k graph", "—"],
      ["B0", "No distal graph and no structural objective: local edges and masked-window reconstruction only", "Does structural conditioning help at all?"],
      ["B1", "Same number of distal edges placed uniformly at random, with borrowed strengths", "Is any graph regularisation enough?"],
      ["B2", "Measured topology, every strength set to 0.5", "Is the gain explained by genomic distance?"],
      ["B3", "Measured topology, strengths permuted among edges", "Does the pairing of measured strength to edge matter?"],
      ["B4", "No distal graph; per-window Hi-C summaries concatenated after the encoder", "Is conditioning inside the encoder better than late fusion?"],
      ["B5", "No distal graph; Hi-C used only as the contrastive alignment target", "Is Hi-C as an alignment target alone sufficient?"],
    ],
  },
  decision: {
    caption:
      "**Pre-registered decision (validation, long range, seeds 1–3).** Difference of seed-averaged Pearson r, 95% paired-bootstrap interval over 406 validation samples (10,000 resamples), and the number of seeds in which X > Y. A comparison passes only if the interval lies above zero and at least two seeds favour X. Source: results/decision_claims_val.json.",
    widths: [2600, 1100, 1300, 2226, 900, 900],
    header: ["Comparison (X > Y)", "Read-out", "Difference", "95% CI", "Seeds", "Pass"],
    rows: [
      ["A: Full > B3", "head", "+0.0008", "−0.0060 to +0.0078", "1/3", "no"],
      ["B: Full > B0", "probe", "+0.0154", "−0.0040 to +0.0348", "3/3", "no"],
      ["B: Full > B1", "probe", "+0.0098", "−0.0075 to +0.0264", "2/3", "no"],
      ["B: B3 > B0", "probe", "+0.0080", "−0.0092 to +0.0255", "2/3", "no"],
      ["B: B3 > B1", "probe", "+0.0024", "−0.0154 to +0.0195", "1/3", "no"],
    ],
  },
  allruns: {
    caption:
      "**Every trained model, every read-out (validation, Pearson r).** Head: the model's own contact head, Hi-C-free, long range (n/a for arms that train no head). With Hi-C: the same head with the measured distal graph supplied (exploratory upper bound; — for arms without a graph). Probe: the shared frozen probe, Hi-C-free, by separation band (short <50 kb, medium 50–150 kb, long ≥150 kb) and overall. Seed 0 is exploratory for every arm. Source: results/<arm>_seed<N>/metrics.json and probe_metrics.json.",
    widths: [800, 1100, 1000, 1100, 1000, 1100, 1000, 1000],
    header: ["Arm", "Seed", "Head (long)", "With Hi-C (long)", "Probe short", "Probe medium", "Probe long", "Probe overall"],
    rows: [
      ["Full", "0 (expl.)", "0.207", "0.669", "0.289", "0.258", "0.161", "0.301"],
      ["Full", "1", "0.188", "0.670", "0.300", "0.305", "0.197", "0.335"],
      ["Full", "2", "0.206", "0.647", "0.266", "0.252", "0.171", "0.302"],
      ["Full", "3", "0.204", "0.615", "0.303", "0.236", "0.159", "0.296"],
      ["B3", "0 (expl.)", "0.185", "0.533", "0.334", "0.317", "0.193", "0.336"],
      ["B3", "1", "0.188", "0.557", "0.274", "0.227", "0.140", "0.278"],
      ["B3", "2", "0.196", "0.653", "0.321", "0.302", "0.197", "0.333"],
      ["B3", "3", "0.211", "0.581", "0.327", "0.279", "0.167", "0.307"],
      ["B1", "0 (expl.)", "0.183", "0.252", "0.323", "0.275", "0.192", "0.325"],
      ["B1", "1", "0.187", "0.223", "0.270", "0.243", "0.145", "0.287"],
      ["B1", "2", "0.128", "0.131", "0.326", "0.309", "0.203", "0.345"],
      ["B1", "3", "0.144", "0.172", "0.315", "0.248", "0.149", "0.297"],
      ["B0", "0 (expl.)", "n/a", "—", "0.320", "0.310", "0.187", "0.338"],
      ["B0", "1", "n/a", "—", "0.287", "0.267", "0.164", "0.309"],
      ["B0", "2", "n/a", "—", "0.325", "0.295", "0.168", "0.326"],
      ["B0", "3", "n/a", "—", "0.281", "0.241", "0.148", "0.292"],
      ["B4", "0 (expl.)", "0.195", "0.826", "0.347", "0.303", "0.192", "0.330"],
      ["B5", "0 (expl.)", "n/a", "—", "0.331", "0.262", "0.167", "0.301"],
      ["B2", "0 (expl.)", "pending", "pending", "pending", "pending", "pending", "pending"],
    ],
  },
  hparams: {
    caption: "**Hyperparameters.** Values are those of the configuration file used for every arm (configs/base.yaml).",
    widths: [3000, 2013, 2000, 2013],
    header: ["Component", "Value", "Component", "Value"],
    rows: [
      ["Window / context", "5 kb / 128 windows", "Model width $d$", "256"],
      ["Local / distal edges", "±3 / top-16, ≥25 kb", "Held-out target edges", "15%"],
      ["Conv tower depth", "5 (÷32)", "State-space layers / state", "6 / 16"],
      ["Attention blocks / heads", "4 / 8", "Feed-forward width", "1,024"],
      ["Structure dropout $p$", "0.3", "Dropout", "0.1"],
      ["Masked windows", "15%", "Contrastive $\\tau$ / negatives", "0.1 / 16"],
      ["$\\lambda_{\\mathrm{mask}}$ / $\\lambda_{\\mathrm{con}}$", "1 / 0.5", "$\\lambda_{\\mathrm{contact}}$ / contact target", "1 / soft-target BCE"],
      ["Steps / warm-up", "20,000 / 1,000", "Learning rate / decay", "3×10⁻⁴ / cosine"],
      ["Samples per step", "8 (2 × 4 accum.)", "Precision", "bfloat16"],
    ],
  },
};

module.exports = C;
