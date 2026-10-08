// The paper's text. build_master.js turns this into master_copy.docx.
//
// Inline markup inside strings:
//   [@key1,key2]   numbered citation, resolved against bibliography.json in order
//                  of first appearance; an unverified or unknown key fails the build
//   *text*         italic        **text**  bold
//   $...$          inline maths: _x / _{xy} subscript, ^x / ^{xy} superscript
//   {fig:name}     "Figure n"    {tab:name} "Table n"    {eq:name} "Eq. (n)"
//
// Claim discipline (paper-writing skill, claim-evidence alignment): nothing in
// these sections states an experimental outcome. Results enter only from
// measured, val-decided, test-confirmed numbers.

const C = {};

C.title =
  "ChromGraphFM: Measured 3D Chromatin Contacts as a Training-Time Signal for Hi-C-Free DNA Sequence Representations";

C.authors = "P. Charan Sai, G. Karthik, P. Sanath, K. Divya Kothapali";
C.affiliation =
  "Department of Computer Science and Engineering, VR Siddhartha Engineering College, Vijayawada, India";

C.draftNote = [
  "Working master copy. Sections 1–3 are drafted; Sections 4–6 (Results, Discussion, Conclusion) are written only once the multi-seed validation results are final and the sealed test split has been scored once.",
  "The abstract's results sentence and the Introduction's experiments paragraph are marked [PENDING RESULTS] and must be filled from measured numbers only.",
];

C.abstract = [
  "DNA sequence models contextualise a genomic window by what lies near it on the linear chromosome, yet much of gene regulation acts across hundreds of kilobases through the three-dimensional folding of chromatin. Chromosome conformation capture (Hi-C) measures that folding directly, but existing models use it as a prediction target, as an input that supervised predictors require at inference, or as an external alignment signal; none lets measured contacts decide which distal windows exchange information inside a self-supervised DNA encoder while it learns. We present ChromGraphFM, a DNA encoder pretrained without task labels, in which a sparse graph built from distance-detrended Hi-C contacts biases attention between 5 kb sequence windows across a 640 kb context. Contacts enter the attention logit through a learned edge bias, and the edges the model is asked to predict are withheld from the graph it attends over, so the contact objective cannot be solved by copying its input. Because Hi-C is rarely available where a sequence model is applied, we treat it as training-time-only information: structure dropout removes the distal graph for a fraction of samples, and a structure-distillation objective pulls the representation computed from sequence alone toward the Hi-C-conditioned one. We evaluate every comparison with Hi-C removed at inference, against matched controls that keep the parameter count, data and compute fixed while removing the graph, randomising it, keeping its edges but flattening their strengths, shuffling its strengths among edges, fusing Hi-C after the encoder, or using it only as an alignment target. [PENDING RESULTS: one sentence with the validation-decided, test-confirmed outcome.]",
];

C.keywords = "3D genome; Hi-C; chromatin contacts; DNA language models; self-supervised learning; graph attention; privileged information";

// --------------------------------------------------------------------------- //
// 1 Introduction
// --------------------------------------------------------------------------- //

C.introduction = [
  // task + application (general -> specific setting)
  "Learning general-purpose representations of DNA sequence has become a central route to predicting the regulatory consequences of the genome, from chromatin accessibility and transcription to the effects of non-coding variants [@dnabert,ntransformer,hyenadna,caduceus,evo,evo2]. A representation is useful to the extent that it captures what controls a locus, and for a large share of human gene regulation that control is distal: enhancers act on promoters tens to hundreds of kilobases away, brought into contact by the folding of chromatin into loops and topologically associating domains [@dixon2012tads,rao2014]. This paper focuses on the setting of learning a sequence representation of a genomic window from DNA alone, where that representation should nevertheless reflect the window's three-dimensional context.",

  // challenge part 1: sequence models
  "This setting is challenging because three-dimensional proximity is not a property of linear proximity. Sequence models enlarge their receptive field with convolutions and dilations [@basenji], attention [@enformer,alphagenome], or long-range state-space and implicit-convolution layers [@hyenadna,caduceus,evo]; however, all of them weigh context by position along the chromosome, and must infer from sequence alone which of thousands of distal windows are physically close. Folding is partly encoded in sequence through features such as CTCF motifs and their orientation [@rao2014,fudenberg2016], but it is cell-type-specific and shaped by factors that a sequence model never observes, so the relevant distal context is diluted among the irrelevant.",

  // challenge part 2: Hi-C methods and the root technical reason
  "Chromosome conformation capture measures this folding directly. Hi-C reports, for every pair of genomic bins, how often the two were found in contact [@liebermanaiden2009,rao2014], and a growing body of work learns from it. Sequence-to-structure models predict contact maps from DNA [@akita,orca,corigami,chimaera]; foundation models are now pretrained on contact maps themselves [@hicfoundation,mixhic]; graph and attention models feed contact frequencies into supervised predictors of gene expression [@graphreg,chromoformer,chrome,puget]; and Evo2HiC distils a frozen DNA language model into a compact encoder aligned to Hi-C with a contrastive objective [@evo2hic]. However, in each of these designs measured contacts never decide how a sequence encoder contextualises its input during self-supervised pretraining: Hi-C is the output, an input that a supervised model needs at prediction time, or an alignment target applied from outside the encoder. The root technical difficulty is twofold. First, injecting contacts into an encoder's attention invites a degenerate solution in which any contact objective is satisfied by reading the contacts back from the input. Second, a representation that depends on Hi-C at inference is unusable for the overwhelming majority of sequences and cell types for which no contact map exists.",

  // pipeline: two-part contribution
  "In this paper, we propose ChromGraphFM, a DNA encoder in which measured chromatin contacts shape sequence contextualisation during pretraining and are not required afterwards. The basic idea is illustrated in {fig:overview}. Our first contribution is contact-biased sparse attention: 5 kb sequence windows become the nodes of a graph whose edges are each window's strongest distance-detrended Hi-C partners, and attention between windows runs over these edges only, with the measured contact strength and the genomic separation entering the attention logit through a learned bias. Because a fraction of the strongest contacts is withheld from the graph and used only as prediction targets, the encoder cannot satisfy its contact objective by copying its input, and must instead learn which sequence windows interact.",

  "Contact-biased attention alone produces a representation that expects Hi-C at inference. Our second contribution resolves this by treating Hi-C as privileged, training-time-only information [@vapnik2009lupi]. Structure dropout removes the entire distal graph for a random subset of training samples, so that the encoder is never out of distribution without it, and a structure-distillation objective trains the representation computed from sequence alone to match the one computed with contacts, with the latter held fixed as the teacher [@hinton2015distill]. The structural knowledge is thereby transferred into the weights that a Hi-C-free model uses.",

  // evaluation design (no outcome claims)
  "Whether such a model has learned anything about structure is easy to assert and hard to test, because a richer input, more parameters, or knowledge of genomic distance can each masquerade as structural learning. We therefore evaluate every model with Hi-C removed at inference and compare it against controls that share its architecture, data, parameter count and compute while removing the distal graph, replacing it with random edges, keeping its edges but flattening their strengths, shuffling its strengths among edges, fusing Hi-C after the encoder, or using Hi-C only as a contrastive alignment target. Splits are by chromosome, model decisions are made on validation chromosomes only, and the test chromosomes are scored once. [PENDING RESULTS: the experiments paragraph, stating what the controlled comparison shows, is written from the final measured numbers.]",

  "In summary, our contributions are as follows:",
];

C.contributions = [
  "Contact-biased sparse attention, in which distance-detrended Hi-C contacts determine which 5 kb sequence windows attend to one another across a 640 kb context and bias how strongly, with held-out target edges that make the contact objective non-trivial.",
  "A training-time-only use of Hi-C, combining structure dropout with structure distillation, that yields DNA representations computed from sequence alone at inference.",
  "A controlled evaluation protocol with six matched controls, chromosome-level splits and a Hi-C-free headline setting, which separates the contribution of measured contacts from those of graph regularisation, genomic distance, contact-strength marginals and post-hoc fusion.",
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
      "A third line learns representations of Hi-C itself. HiCFoundation pretrains a foundation model on a large corpus of contact maps [@hicfoundation], and MIX-HIC jointly pretrains on contact maps and epigenomic tracks [@mixhic]; in both, the reusable representation is of contacts rather than of DNA sequence. Chimaera links the latent space of a contact-map autoencoder to a DNA encoder to predict Hi-C from sequence across species [@chimaera]. GraphReg uses Hi-C and related 3D assays as the graph of a graph-attention network that predicts gene expression from epigenomic and sequence features [@graphreg], and CHROME and Puget likewise combine sequence representations with contact graphs or a Hi-C encoder to predict expression [@chrome,puget]. These works show that contact graphs carry information that linear context lacks, but in supervised settings where Hi-C is required at prediction time. Closest to our aim, and concurrent with it, Evo2HiC distils a frozen Evo 2 model into a compact sequence encoder that is aligned to a Hi-C encoder with a contrastive objective, and offers a sequence-only encoder for use without Hi-C [@evo2hic]. It thus shares our view of Hi-C as a training-time signal, but contacts act there as an alignment target applied from outside the encoder, through a contrastive loss, rather than inside it. ChromGraphFM differs in where Hi-C acts: contacts select and weight the information each window receives at every graph layer during pretraining, and dropout and distillation remove the dependence on Hi-C at inference. Our controls include both a late-fusion and an alignment-only variant (B4 and B5) so that this difference is tested rather than asserted.",
    ],
  },
  {
    heading: "Structural bias in attention and training-time-only signals",
    paras: [
      "Our attention mechanism belongs to a family that restricts or biases attention with a known structure. Graph attention networks compute attention over graph neighbourhoods [@gat], and Graphormer adds learned structural biases, such as shortest-path distances, directly to the attention logits of a Transformer [@graphormer,transformer]. In genomics, Chromoformer adds normalised Hi-C interaction frequencies to the self-attention scores of a supervised model that predicts expression from histone modifications around promoters [@chromoformer], a direct precedent for biasing attention with contact frequency. We follow the additive-bias form, but apply it during self-supervised pretraining of a DNA sequence encoder, with the bias computed from measured contact strength and genomic separation by separate learned functions, a separation that makes a distance-only control constructible. Removing structure during training to obtain robustness has precedents in DropEdge, which drops random edges as a regulariser [@dropedge], and in modality dropout, which drops whole input modalities so a model remains usable when one is missing [@moddrop]. Our structure dropout removes the whole distal graph per sample for the second reason. Finally, learning using privileged information formalises a teacher signal available at training time only [@vapnik2009lupi], and knowledge distillation transfers a teacher's behaviour into a student [@hinton2015distill]; our structure distillation uses the model's own Hi-C-conditioned pass as the teacher of its Hi-C-free pass.",
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
      "*Hi-C-free use.* Structure dropout and structure distillation train the model to produce its representation from DNA alone (Section 3.6), so Hi-C is needed only during training. This matters for two reasons: contact maps exist for few cell types and for no user-supplied sequence, and comparing models with Hi-C removed is the only setting in which a gain can be attributed to what training stored in the weights rather than to information supplied at inference. Four objectives train the model end to end (Section 3.7); Section 3.8 defines the controls and the evaluation protocol, and Section 3.9 gives implementation details.",
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
    heading: "Hi-C-free representations: structure dropout and structure distillation",
    paras: [
      "Hi-C is unavailable for most cell types and for any sequence a downstream user supplies, so a representation that requires it at inference has limited use. We therefore treat Hi-C as privileged information [@vapnik2009lupi] and train the model to produce good representations without it. *Structure dropout* removes the entire distal graph for each training sample independently with probability $p = 0.3$, leaving only local edges:",
    ],
    eqs: ["dropout"],
    parasAfter: [
      "Removing the whole graph rather than individual edges [@dropedge] matches the condition at inference exactly, where every distal edge is absent at once [@moddrop]. Dropout alone, however, only ensures that the Hi-C-free mode is in distribution; it gives that mode no direct access to what the contacts revealed. *Structure distillation* supplies it. For every sample, the same encoder outputs $h_i$, including any masking, are passed through the same attention blocks a second time with the distal graph removed, giving $z^{\\mathrm{free}}_i$, and this Hi-C-free representation is trained to match the Hi-C-conditioned $z_i$ ({eq:l_distill}). The teacher is held fixed with a stop-gradient, so the conditioned pass cannot degrade toward the impoverished one to make the two agree. Because the local encoder runs once and only the graph blocks run twice, distillation adds little computation. At inference, the model is run in exactly the Hi-C-free mode that distillation trains.",
    ],
  },
  {
    heading: "Pretraining objectives",
    flow: [
      { p: "Four objectives train the model. The masked-window and contrastive terms are self-supervised with respect to the sequence; the contact and distillation terms use Hi-C, or the model's own Hi-C-conditioned pass, as training-time-only targets. No task labels are used." },
      { p: "*Masked-window reconstruction.* For 15% of windows, chosen at random, $h_i$ is replaced by a learned mask token before the attention blocks, and a two-layer network $g$ must reconstruct the window's original embedding from its output $z_i$. A whole window, not individual bases, is masked: a pooled 256-dimensional vector cannot carry 5,000 bases, but it can be inferred from the window's graph neighbours, which is the capacity we want to train. With $\\mathcal{M}$ the set of masked usable windows and $\\mathrm{sg}$ the stop-gradient," },
      { eq: "l_mask" },
      { p: "*Distance-matched contrastive alignment.* Each held-out target edge $(i, j) \\in \\mathcal{T}$ is a positive pair, contrasted against up to 16 negative pairs $\\mathcal{D}_{ij}$ drawn from the same sample at exactly the same separation, with an InfoNCE loss [@cpc,simclr] on projected, $\\ell_2$-normalised representations $\\tilde z$ and temperature $\\tau = 0.1$. Matching the separation of negatives to that of the positive removes the shortcut of judging contact by distance, which the model can already read from its input. Every held-out positive contributes, rather than one sampled positive per sample:" },
      { eq: "l_con" },
      { p: "*Contact regression.* A symmetric head $f$ predicts each target edge's log observed-over-expected contact from the two window representations and their separation. Because $c = \\mathrm{O/E}/(1+\\mathrm{O/E})$, the target $\\log \\mathrm{O/E}$ equals the logit of $c$, so the head's output is a logit and $\\sigma(f)$ is a predicted strength. Regressing the logit directly, rather than applying binary cross-entropy against the soft strength, keeps the same optimum while avoiding the shrinkage of the cross-entropy gradient by $\\sigma'(f)$, which is small for the strongly enriched contacts that make up the targets:" },
      { eq: "l_contact" },
      { p: "*Structure distillation* is defined over the set $V$ of usable windows as" },
      { eq: "l_distill" },
      { p: "and the total objective is" },
      { eq: "l_total" },
      { p: "with $\\lambda_{\\mathrm{mask}} = 1$, $\\lambda_{\\mathrm{con}} = 0.5$, $\\lambda_{\\mathrm{contact}} = 1$ and $\\lambda_{\\mathrm{distill}} = 1$." },
    ],
  },
  {
    heading: "Controlled experimental design",
    paras: [
      "The claim that measured contacts improve a sequence representation is only meaningful against controls that remove everything else a graph-conditioned model has. We therefore train six control arms ({tab:arms}) through the same code path as the full model, with the same architecture and parameter count, the same training windows, the same number of optimisation steps, the same pretraining recipe and the same held-out targets; only the structural signal differs. Each corruption is drawn from the fixed control seed, so every training seed sees the identical corrupted graph and seed variance cannot be confused with corruption variance. The arms are designed to be read in pairs: B0 against the full model asks whether structure helps at all; B1 asks whether any graph regularisation would do; B2 isolates genomic distance; B3, which keeps the measured topology but permutes strengths among edges, isolates the pairing of strength to edge; B4 asks whether conditioning inside the encoder is better than adding Hi-C features after it; and B5 asks whether Hi-C as a contrastive alignment target alone suffices.",
      "*Splits.* Hi-C is autocorrelated over megabases, so random window splits would leak. We split by chromosome: chr3, chr13 and chr17 form the test set, chr10 and chr11 the validation set, and the remaining 17 autosomes the training set (6,284 training samples at a stride of half a sample). All model and recipe decisions are made on the validation chromosomes; the test chromosomes are scored once, for the final table.",
      "*Headline setting and metric.* Every arm is evaluated with Hi-C removed at inference, so that the only difference between arms is what shaped their weights during pretraining. Results with the graph present are reported separately as an upper bound with privileged input and are never compared with a sequence-only baseline. The headline metric is the Pearson correlation between predicted and measured strength on held-out target edges, stratified by separation into short (<50 kb), medium (50–150 kb) and long (≥150 kb) bands, because a single correlation is dominated by short-range contacts; the long-range band is the primary endpoint. For arms whose pretraining does not train a contact head (B0 and B5), the same contact head is fitted on frozen representations, and all arms are additionally compared under this frozen-representation probe. Differences between arms are assessed with a paired bootstrap over evaluation samples, which is valid because every arm is asked about the identical target edges, and over multiple training seeds.",
    ],
    table: "arms",
  },
  {
    heading: "Implementation details",
    paras: [
      "The model has 7.3 M parameters: 3.3 M in the local encoder, 3.2 M in the attention blocks and the remainder in the heads. Models are trained for 20,000 steps with AdamW (learning rate $3 \\times 10^{-4}$, weight decay 0.01), 1,000 warm-up steps followed by cosine decay, gradient clipping at 1.0 and bfloat16 mixed precision. Each step accumulates gradients over four micro-batches of two samples, so one step sees 8 samples, or 1,024 windows (5.1 Mb of sequence). Training uses one NVIDIA L40S GPU per run, with a peak memory of 14.3 GiB. Data are built reproducibly from a manifest that pins the source accessions and checksums, and every run records its resolved configuration and code commit before its first step. {tab:hparams} lists the hyperparameters.",
    ],
    table: "hparams",
  },
];

C.figures = {
  overview: {
    file: "figures/fig1_overview.png",
    width: 6.3,
    caption:
      "**Overview of ChromGraphFM.** Top: each 5 kb DNA window is encoded independently by a convolutional tower and a bidirectional state-space model into $h_i$. Bottom: distance-detrended Hi-C contacts define a sparse graph (local and top-k distal edges), from which a fraction of distal edges is withheld as targets. Middle: contact-biased attention propagates information over graph edges only, with the measured strength and separation entering the attention logit through a learned edge bias. A second pass over the same encoder outputs with the distal graph removed gives the Hi-C-free representation $z^{\\mathrm{free}}_i$, which is distilled toward $z_i$. Hi-C is used only during training; at inference the model runs from sequence alone.",
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
};

C.tables = {
  arms: {
    caption:
      "**Matched experimental arms.** All arms share architecture, parameter count, training data, steps and held-out targets, and are evaluated with Hi-C removed at inference.",
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
  hparams: {
    caption: "**Hyperparameters.** Values are those of the configuration file used for every arm.",
    widths: [3000, 2013, 2000, 2013],
    header: ["Component", "Value", "Component", "Value"],
    rows: [
      ["Window / context", "5 kb / 128 windows", "Model width $d$", "256"],
      ["Local / distal edges", "±3 / top-16, ≥25 kb", "Held-out target edges", "15%"],
      ["Conv tower depth", "5 (÷32)", "State-space layers / state", "6 / 16"],
      ["Attention blocks / heads", "4 / 8", "Feed-forward width", "1,024"],
      ["Structure dropout $p$", "0.3", "Dropout", "0.1"],
      ["Masked windows", "15%", "Contrastive $\\tau$ / negatives", "0.1 / 16"],
      ["$\\lambda_{\\mathrm{mask}}$ / $\\lambda_{\\mathrm{con}}$", "1 / 0.5", "$\\lambda_{\\mathrm{contact}}$ / $\\lambda_{\\mathrm{distill}}$", "1 / 1"],
      ["Steps / warm-up", "20,000 / 1,000", "Learning rate / decay", "3×10⁻⁴ / cosine"],
      ["Samples per step", "8 (2 × 4 accum.)", "Precision", "bfloat16"],
    ],
  },
};

module.exports = C;
