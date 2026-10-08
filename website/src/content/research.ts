/**
 * Every claim, question and citation on the site.
 *
 * Mirrored by hand from scripts/build_decks.py so the site and the review decks
 * tell the same story. Prose drifting between the two is a proofreading problem;
 * numbers drifting would be a wrong claim, which is why no number lives in this
 * file — those come from configs/base.yaml via src/lib/config.ts.
 */

export interface ResearchQuestion {
  id: string;
  title: string;
  body: string;
}

export const QUESTIONS: ResearchQuestion[] = [
  {
    id: "RQ1",
    title: "Does structural conditioning help at all?",
    body: "Does biasing attention with measured Hi-C improve held-out chromosome contact prediction over a matched sequence-only encoder?",
  },
  {
    id: "RQ2",
    title: "Is the gain from real biology, or from any extra signal?",
    body: "Does it beat matched controls given shuffled Hi-C, distance-only edges, and a random contact graph?",
  },
  {
    id: "RQ3",
    title: "Is conditioning better than fusion?",
    body: "Does contact-biased attention inside the encoder beat late fusion applied after the sequence has already been encoded?",
  },
  {
    id: "RQ4",
    title: "Is the structure retained in the sequence representation?",
    body: "Does any improvement survive removing the contact graph at inference time?",
  },
  {
    id: "RQ5",
    title: "Does it transfer?",
    body: "Does the pretrained encoder beat a from-scratch model on chromatin loop detection and enhancer–promoter linking, including on a held-out cell line?",
  },
];

export interface Objective {
  id: string;
  title: string;
  body: string;
  phase: string;
}

export const OBJECTIVES: Objective[] = [
  {
    id: "O1",
    title: "Reproducible data foundation",
    body: "A chromosome-split hg38 and Hi-C dataset, rebuildable on any machine from a tracked manifest of accessions and checksums.",
    phase: "Phase 1",
  },
  {
    id: "O2",
    title: "Sparse contact graph construction",
    body: "Genomic regions as sparse graphs with local and distal edges, and prediction-target edges provably held out from the conditioning graph.",
    phase: "Phase 2",
  },
  {
    id: "O3",
    title: "Contact-biased sequence encoder",
    body: "A convolutional tower and bidirectional state-space encoder per window, contact-biased sparse attention across windows, and structure dropout, trained end to end from scratch.",
    phase: "Phase 4",
  },
  {
    id: "O4",
    title: "Structure-aware pretraining",
    body: "Masked-window reconstruction, distance-matched contrastive learning and held-out contact prediction combined into a single objective, with structure distillation in the v2 recipe.",
    phase: "Phase 4",
  },
  {
    id: "O5",
    title: "Controlled evaluation",
    body: "Six control arms at matched parameters, data and steps, scored on held-out chromosomes with Hi-C removed at inference, plus a frozen-representation probe; a held-out cell line follows.",
    phase: "Phase 5",
  },
  {
    id: "O6",
    title: "Biological interpretation",
    body: "Test whether high-attention contact anchors are enriched for CTCF motifs and overlap known loop anchors.",
    phase: "Phase 6",
  },
];

export interface Paper {
  name: string;
  venue: string;
  idea: string;
  gap: string;
  family: "sequence-to-structure" | "hi-c representation" | "sequence foundation" | "closest";
  href: string;
}

export const LITERATURE: Paper[] = [
  {
    name: "Akita",
    venue: "Nature Methods, 2020",
    idea: "A convolutional network predicting 3D genome folding directly from DNA sequence.",
    gap: "A task-specific sequence-to-contact predictor. It produces no reusable pretrained representation, and observed Hi-C never conditions its encoder.",
    family: "sequence-to-structure",
    href: "https://www.nature.com/articles/s41592-020-0958-x",
  },
  {
    name: "C.Origami",
    venue: "Nature Biotechnology, 2023",
    idea: "Cell-type-specific 3D chromatin organisation predicted from sequence, CTCF binding and accessibility.",
    gap: "Hi-C is the prediction target rather than a signal shaping a reusable sequence encoder.",
    family: "sequence-to-structure",
    href: "https://www.nature.com/articles/s41587-022-01612-8",
  },
  {
    name: "EPCOT",
    venue: "Nucleic Acids Research, 2023",
    idea: "A pretrain-and-fine-tune framework spanning epigenome, chromatin contacts, transcription and enhancer activity.",
    gap: "Its pretraining centres on epigenomic feature prediction, not on Hi-C-conditioned sequence modelling.",
    family: "sequence-to-structure",
    href: "https://academic.oup.com/nar/article/51/12/5931/7177889",
  },
  {
    name: "Caduceus",
    venue: "ICML 2024",
    idea: "A bidirectional, reverse-complement-equivariant long-range DNA sequence model.",
    gap: "A strong sequence-only backbone and the right baseline to beat — but it uses no Hi-C signal at all.",
    family: "sequence foundation",
    href: "https://proceedings.mlr.press/v235/schiff24a.html",
  },
  {
    name: "HiCFoundation",
    venue: "Nature Methods, 2026",
    idea: "A foundation model pretrained on a large corpus of Hi-C contact maps.",
    gap: "A genuine foundation model — but of contact maps. Its reusable representation is of Hi-C, not of DNA sequence.",
    family: "hi-c representation",
    href: "https://www.nature.com/articles/s41592-026-03097-8",
  },
  {
    name: "MIX-HIC",
    venue: "NeurIPS 2025",
    idea: "Multimodal pretraining over Hi-C contact maps and epigenomic tracks, on more than a million paired samples.",
    gap: "Raw DNA sequence is not an input. The authors name sequence integration explicitly as future work.",
    family: "hi-c representation",
    href: "https://arxiv.org/abs/2504.09060",
  },
  {
    name: "GraphReg",
    venue: "Genome Research, 2022",
    idea: "Graph attention over Hi-C, HiChIP or HiCAR graphs to predict gene expression from epigenomic tracks or DNA sequence.",
    gap: "Supervised and task-specific, and the graph is required at prediction time. Contacts carry information, but no reusable sequence representation is pretrained.",
    family: "hi-c representation",
    href: "https://doi.org/10.1101/gr.275870.121",
  },
  {
    name: "Chromoformer",
    venue: "Nature Communications, 2022",
    idea: "A transformer predicting gene expression from histone marks, with promoter-capture Hi-C interaction frequencies added to its attention scores.",
    gap: "The closest precedent for contact-biased attention logits. But it is supervised, its input is histone marks rather than DNA, and it needs interaction data at prediction time.",
    family: "closest",
    href: "https://doi.org/10.1038/s41467-022-34152-5",
  },
  {
    name: "CHROME",
    venue: "Briefings in Bioinformatics, 2026",
    idea: "Graph attention over polymer-null-filtered Hi-C contacts, with sequence and pretrained embeddings as node features.",
    gap: "Supervised regulatory prediction; the contact graph is needed at inference and does not shape a pretrained encoder.",
    family: "hi-c representation",
    href: "https://doi.org/10.1093/bib/bbag360",
  },
  {
    name: "Evo2HiC",
    venue: "bioRxiv, Nov 2025",
    idea: "Contrastive distillation of a frozen 7B-parameter Evo 2 into a compact CNN encoder, aligned against Hi-C patch embeddings.",
    gap: "The closest precedent for Hi-C as a training-time-only signal, and the reason this project cannot claim to be first. But the Evo 2 teacher stays frozen and contacts never enter the student encoder's own contextualisation; they supply an alignment target from outside it.",
    family: "closest",
    href: "https://www.biorxiv.org/content/10.1101/2025.11.18.689171v1",
  },
];

export const GAP = {
  established: [
    "Foundation models pretrained on Hi-C contact maps exist.",
    "Multimodal Hi-C and epigenomic pretraining exists.",
    "Hi-C-guided distillation into a DNA encoder exists.",
    "Interaction frequencies have biased attention before, in supervised models with Hi-C at inference.",
  ],
  open: [
    "No published model pretrains a DNA encoder, without task labels, while measured contact graphs control which distal windows exchange information at every layer, and then removes them.",
    "Post-hoc alignment and late fusion have not been separated from in-encoder conditioning under matched parameters, data and compute.",
    "Whether structural knowledge survives the removal of Hi-C at inference is asserted more often than it is tested against a shuffled-contact control.",
  ],
  statement:
    "It remains untested whether contact graphs conditioning sequence contextualisation throughout pretraining yield a more transferable DNA representation than the alternatives, measured against distance-matched and shuffled-contact controls.",
  notClaimed: [
    "The first genomic foundation model to use Hi-C",
    "The first Hi-C foundation model",
    "The first model to combine DNA sequence and Hi-C",
    "The first sequence encoder pretrained with Hi-C-guided contrastive learning",
    "The first model to add contact frequencies to attention scores",
  ],
};

export interface Arm {
  id: string;
  name: string;
  tests: string;
  structure: string;
  decisive?: boolean;
  ours?: boolean;
}

export const ARMS: Arm[] = [
  {
    id: "B0",
    name: "DNA-only",
    structure: "Local edges only: no distal graph and no structural objective.",
    tests: "The floor. Does structure buy anything over pure sequence at matched capacity?",
  },
  {
    id: "B1",
    name: "Random contact graph",
    structure: "The same number of distal edges, placed at random.",
    tests: "Whether any graph regularisation helps, independent of what the edges mean.",
  },
  {
    id: "B2",
    name: "Distance-only edges",
    structure: "The real edges, with every contact strength set to one constant.",
    tests: "Whether the gain is just linear distance — a fact the model can already read off the input.",
    decisive: true,
  },
  {
    id: "B3",
    name: "Shuffled Hi-C",
    structure: "The real edges, with contact strengths permuted among them.",
    tests: "Whether the measured strength of each contact matters. Identical edges and strength distribution, scrambled assignment.",
    decisive: true,
  },
  {
    id: "B4",
    name: "Late fusion",
    structure: "No distal graph; Hi-C summaries concatenated after the graph blocks.",
    tests: "Whether conditioning inside the encoder beats adding structure afterwards.",
  },
  {
    id: "B5",
    name: "Contrastive alignment only",
    structure: "No distal graph; Hi-C used only as a contrastive alignment target.",
    tests: "The simplified analogue of the closest published precedent.",
  },
  {
    id: "Ours",
    name: "Contact-biased attention",
    structure: "Measured contacts bias attention at every block.",
    tests: "The full proposed model.",
    ours: true,
  },
];

export interface Phase {
  id: string;
  weeks: string;
  deliverable: string;
  gate: string;
  gpu: boolean;
  status: "done" | "active" | "planned";
}

export const PHASES: Phase[] = [
  {
    id: "Phase 0",
    weeks: "Week 1",
    deliverable: "Scaffold, config system, split validator",
    gate: "The config refuses a chromosome leak between splits",
    gpu: false,
    status: "done",
  },
  {
    id: "Phase 1",
    weeks: "Weeks 2–5",
    deliverable: "hg38 windows, streamed Hi-C, binned contacts, QC report",
    gate: "Bins, windows and coordinates agree at five independently inspected loci",
    gpu: false,
    status: "done",
  },
  {
    id: "Phase 2",
    weeks: "Weeks 6–8",
    deliverable: "Sparse contact graphs, held-out edges, control corruptions",
    gate: "Target edges provably absent from the conditioning graph",
    gpu: false,
    status: "done",
  },
  {
    id: "Phase 3",
    weeks: "Weeks 9–11",
    deliverable: "State-space encoder, B0 baseline, evaluation harness",
    gate: "B0 trains, checkpoints and evaluates end to end",
    gpu: true,
    status: "done",
  },
  {
    id: "Phase 4",
    weeks: "Weeks 12–15",
    deliverable: "Contact-biased attention, pretraining losses, structure distillation (v2)",
    gate: "STOP GATE — real Hi-C beats shuffled Hi-C on validation, or the claim dies here",
    gpu: true,
    status: "active",
  },
  {
    id: "Phase 5",
    weeks: "Weeks 16–18",
    deliverable: "Matched B0–B5 sweep, held-out chromosome and cell line",
    gate: "The gain survives three seeds and a held-out cell line",
    gpu: true,
    status: "active",
  },
  {
    id: "Phase 6",
    weeks: "Weeks 19–20",
    deliverable: "CTCF interpretation, report, poster, public repository",
    gate: "Every number traces to a tracked file",
    gpu: true,
    status: "planned",
  },
];

export interface DataSource {
  data: string;
  source: string;
  format: string;
  use: string;
}

export const DATA_SOURCES: DataSource[] = [
  { data: "Reference genome", source: "UCSC hg38", format: "FASTA + .fai", use: "DNA window per node" },
  { data: "Bulk Hi-C", source: "4DN Data Portal", format: ".mcool, ICE-normalised", use: "Contact graph, pretraining signal" },
  { data: "Hi-C, second cell line", source: "4DN Data Portal (K562)", format: ".mcool", use: "Held-out cell-type test" },
  { data: "Chromatin loops", source: "CTCF ChIA-PET / HiChIP", format: "BEDPE", use: "Loop detection labels" },
  { data: "Gene expression", source: "ENCODE CAGE / RNA-seq", format: "bigWig / TSV", use: "Downstream transfer task" },
  { data: "CTCF motif", source: "JASPAR MA0139", format: "PWM", use: "Interpretation only" },
];

export const REFERENCES = [
  "Fudenberg, G., Kelley, D. R., & Pollard, K. S. (2020). Predicting 3D genome folding from DNA sequence with Akita. Nature Methods, 17, 1111–1117.",
  "Tan, J., et al. (2023). Cell-type-specific prediction of 3D chromatin organization enables high-throughput in silico genetic screening. Nature Biotechnology, 41(8), 1140–1150.",
  "Zhang, Z., et al. (2023). A generalizable framework to comprehensively predict epigenome, chromatin organization, and transcriptome. Nucleic Acids Research, 51(12), 5931–5947.",
  "Schiff, Y., et al. (2024). Caduceus: Bi-directional equivariant long-range DNA sequence modeling. Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235, 43632–43648.",
  "Wang, X., et al. (2026). A generalizable Hi-C foundation model for chromatin architecture, single-cell and multiomics analysis across species. Nature Methods, 23(7), 1334–1348.",
  "Yang, M., et al. (2025). Multimodal 3D genome pre-training (MIX-HIC). Advances in Neural Information Processing Systems (NeurIPS 2025).",
  "Fang, T., et al. (2025). Evo2HiC: a multimodal foundation model for integrative analysis of genome sequence and architecture. bioRxiv 2025.11.18.689171.",
  "Karbalayghareh, A., Sahin, M., & Leslie, C. S. (2022). Chromatin interaction-aware gene regulatory modeling with graph attention networks. Genome Research, 32(5), 930–944.",
  "Lee, D., Yang, J., & Kim, S. (2022). Learning the histone codes with large genomic windows and three-dimensional chromatin interactions using transformer. Nature Communications, 13, 6678.",
  "Ye, B., et al. (2026). A chromatin-structure-guided framework for predictive and interpretable regulatory genomics. Briefings in Bioinformatics, 27, bbag360.",
  "Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). On the parameterization and initialization of diagonal state space models. NeurIPS 2022.",
  "Imakaev, M., et al. (2012). Iterative correction of Hi-C data reveals hallmarks of chromosome organization. Nature Methods, 9(10), 999–1003.",
  "Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 2017.",
  "Rao, S. S. P., et al. (2014). A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. Cell, 159(7), 1665–1680.",
  "Abdennur, N., & Mirny, L. A. (2020). Cooler: scalable storage for Hi-C data and other genomically labeled arrays. Bioinformatics, 36(1), 311–316.",
];

export const SECTIONS = [
  { id: "problem", label: "The problem" },
  { id: "aim", label: "Aim" },
  { id: "questions", label: "Research questions" },
  { id: "objectives", label: "Objectives" },
  { id: "literature", label: "Literature" },
  { id: "gap", label: "The gap" },
  { id: "architecture", label: "Architecture" },
  { id: "objective-fn", label: "Objectives function" },
  { id: "data", label: "Data & splits" },
  { id: "experiments", label: "Experiments" },
  { id: "results", label: "Results" },
  { id: "timeline", label: "Timeline" },
  { id: "reproducibility", label: "Reproducibility" },
  { id: "references", label: "References" },
] as const;
