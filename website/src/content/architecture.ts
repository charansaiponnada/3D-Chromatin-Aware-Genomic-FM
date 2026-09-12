/**
 * The model architecture as data.
 *
 * The diagram is not a picture with hotspots painted on it — it is rendered from
 * this array, so a node cannot exist visually without an explanation attached,
 * and an explanation cannot drift away from the box it describes.
 *
 * Coordinates live in a 1000 x 640 virtual grid. The SVG edge layer and the HTML
 * node layer both use it, which is what makes the whole thing responsive without
 * measuring anything at runtime.
 *
 * `configRefs` are dotted paths into configs/base.yaml. scripts/sync-assets.mjs
 * fails the build if one of them stops resolving.
 */

export type NodeId =
  | "dna"
  | "tokenizer"
  | "mamba"
  | "window-emb"
  | "hic"
  | "graph-build"
  | "held-out"
  | "struct-dropout"
  | "bias"
  | "attn"
  | "an1"
  | "ffn"
  | "an2"
  | "z"
  | "head-dna"
  | "head-contrast"
  | "head-contact"
  | "loss";

export type Lane = "sequence" | "structure" | "fusion" | "heads";
export type NodeKind = "data" | "module" | "bias" | "head" | "novel" | "output";

export interface ArchNode {
  id: NodeId;
  label: string;
  sub?: string;
  lane: Lane;
  kind: NodeKind;
  x: number;
  y: number;
  w: number;
  h: number;
  /** One line. Shown on hover and focus. */
  short: string;
  /** What this component does, mechanically. */
  what: string;
  /** Why it is here — the design decision, not the description. */
  why: string;
  math?: string;
  configRefs?: string[];
  /** Marked true for the four components that constitute the novelty claim. */
  novel?: boolean;
}

export type EdgeKind = "tensor" | "structure" | "bias" | "loss" | "target";

export interface ArchEdge {
  from: NodeId;
  to: NodeId;
  kind: EdgeKind;
  label?: string;
  /** True when structure dropout removes this edge during pretraining. */
  drops?: boolean;
}

export const LANE_LABEL: Record<Lane, string> = {
  sequence: "Sequence stream",
  structure: "Structure stream",
  fusion: "Structure-conditioned encoder block",
  heads: "Self-supervised objectives",
};

export const GRID = { w: 1000, h: 640 } as const;

export const NODES: ArchNode[] = [
  // ---------------------------------------------------------------- sequence
  {
    id: "dna",
    label: "DNA sequence",
    sub: "one-hot A/C/G/T/N",
    lane: "sequence",
    kind: "data",
    x: 16,
    y: 44,
    w: 150,
    h: 70,
    short: "The raw genomic input: a stack of fixed-width DNA windows from hg38.",
    what:
      "Each sample is a contiguous run of genomic windows read from the reference assembly. " +
      "One window becomes one node in the contact graph, so the window width and the Hi-C bin " +
      "width are deliberately the same number.",
    why:
      "Keeping window width equal to bin width removes a pooling layer between the sequence " +
      "encoder and the graph, and with it an entire class of off-by-one coordinate bugs. " +
      "Windows whose sequence is mostly N are dropped rather than fed to the model as noise.",
    configRefs: ["data.assembly", "data.bin_size", "data.nodes_per_sample", "data.max_n_frac"],
  },
  {
    id: "tokenizer",
    label: "Embedding",
    sub: "nucleotide + position",
    lane: "sequence",
    kind: "module",
    x: 182,
    y: 44,
    w: 150,
    h: 70,
    short: "Nucleotides become vectors; position within the window is encoded.",
    what:
      "Each base is mapped to a learned vector and combined with its offset inside the window. " +
      "N is a real token rather than a dropped position, so assembly gaps stay visible to the model.",
    why:
      "Positional information has to survive into the encoder because regulatory grammar is " +
      "positional — a motif's spacing from a neighbouring motif carries meaning that a bag of " +
      "bases would destroy.",
    configRefs: ["model.d_model"],
  },
  {
    id: "mamba",
    label: "Bi-Mamba encoder",
    sub: "local, per window",
    lane: "sequence",
    kind: "module",
    x: 348,
    y: 44,
    w: 150,
    h: 70,
    short:
      "A bidirectional state-space encoder that reads each window and emits one vector for it.",
    what:
      "A stack of bidirectional Mamba blocks runs over the bases inside a single window and pools " +
      "to one embedding per window. It sees the window's own sequence and nothing else — all " +
      "cross-window communication is deferred to the graph.",
    why:
      "A state-space model is linear in sequence length where attention is quadratic, which is what " +
      "makes per-base modelling affordable inside every one of the windows. Bidirectional because " +
      "DNA has no reading direction: a motif is a motif on either strand, so a causal scan would " +
      "throw away half the context for no reason.",
    configRefs: ["model.encoder_layers", "model.d_state", "model.d_conv", "model.expand"],
  },
  {
    id: "window-emb",
    label: "Window embeddings",
    sub: "graph nodes",
    lane: "sequence",
    kind: "output",
    x: 514,
    y: 44,
    w: 150,
    h: 70,
    short: "One vector per window. These are the nodes the contact graph connects.",
    what:
      "The output of the local encoder: a matrix of one embedding per genomic window. From here " +
      "on, the model reasons about windows rather than bases.",
    why:
      "This is the level at which Hi-C is measured, so it is the level at which structure can be " +
      "applied honestly. Conditioning individual base positions on a contact map would be applying " +
      "a megabase-scale measurement to a one-base-scale decision.",
    configRefs: ["data.nodes_per_sample", "model.d_model"],
  },

  // --------------------------------------------------------------- structure
  {
    id: "hic",
    label: "Hi-C contact map",
    sub: "ICE-normalised",
    lane: "structure",
    kind: "data",
    x: 16,
    y: 500,
    w: 150,
    h: 76,
    short: "The experimental measurement of which genomic regions are physically close.",
    what:
      "Hi-C sequences pairs of genomic fragments that were in contact in the nucleus, producing a " +
      "matrix of contact counts between genomic bins. Maps are read at a fixed bin size and used " +
      "after ICE normalisation.",
    why:
      "This is the whole premise. An enhancer can regulate a gene a megabase away because folding " +
      "brings them together, and Hi-C is the direct readout of that folding. A sequence-only model " +
      "has to infer this relationship; here it is handed the measurement.",
    configRefs: ["data.bin_size", "data.cell_lines"],
  },
  {
    id: "graph-build",
    label: "Sparse graph builder",
    sub: "top-k + local",
    lane: "structure",
    kind: "novel",
    novel: true,
    x: 182,
    y: 500,
    w: 150,
    h: 76,
    short: "Turns the dense contact matrix into a sparse graph over windows.",
    what:
      "Contacts are first detrended against the expected contact at each genomic separation. For " +
      "every window the strongest distal partners are kept, together with its immediate sequential " +
      "neighbours. Each surviving edge carries contact strength, genomic distance, resolution and " +
      "cell type.",
    why:
      "Dense attention over every window pair would be quadratic and would spend most of its budget " +
      "on pairs that never touch. Detrending first matters more than the sparsity: raw Hi-C is " +
      "dominated by the fact that nearby loci contact more often, so a top-k over raw counts would " +
      "select nothing but near neighbours and the model would learn genomic distance, not structure.",
    configRefs: ["data.top_k_edges", "data.local_radius", "data.min_separation"],
  },
  {
    id: "held-out",
    label: "Held-out target edges",
    sub: "excluded from conditioning",
    lane: "structure",
    kind: "novel",
    novel: true,
    x: 348,
    y: 500,
    w: 150,
    h: 76,
    short: "A fraction of edges is withheld from the model and used only as prediction targets.",
    what:
      "A fixed fraction of edges is sampled out of the graph before it reaches the encoder. Those " +
      "edges never bias attention; they are only ever used as labels for the contact-prediction head.",
    why:
      "Without this the contact head could read an edge from its own input and echo it back, and the " +
      "contact metric would measure nothing at all. Sampling uses a control seed separate from the " +
      "training seed, so every training run sees the identical hold-out and seed variance never gets " +
      "confounded with hold-out variance.",
    configRefs: ["data.held_out_edge_frac", "control_seed"],
  },
  {
    id: "struct-dropout",
    label: "Structure dropout",
    sub: "whole graph, probability p",
    lane: "structure",
    kind: "novel",
    novel: true,
    x: 514,
    y: 500,
    w: 150,
    h: 76,
    short: "With probability p the entire contact graph is removed for a training sample.",
    what:
      "On each sample a coin is flipped. When it comes up, every distal edge is dropped and the " +
      "block runs on local edges alone — the model must produce its representation from sequence " +
      "with no structural help.",
    why:
      "This is what makes the central claim testable rather than hopeful. The project asks whether " +
      "structure gets internalised into the sequence representation, which is only meaningful if the " +
      "encoder still works when Hi-C is taken away. Without dropout, removing the graph at inference " +
      "is an out-of-distribution input and a poor score would say nothing about what was learned.",
    math: "p_{\\text{drop}} \\sim \\text{Bernoulli}(p), \\quad E_{\\text{in}} \\to E_{\\text{local}}",
    configRefs: ["model.structure_dropout"],
  },

  // ------------------------------------------------------------------ fusion
  {
    id: "attn",
    label: "Contact-biased attention",
    sub: "over graph edges only",
    lane: "fusion",
    kind: "novel",
    novel: true,
    x: 352,
    y: 374,
    w: 266,
    h: 56,
    short: "Multi-head attention restricted to graph edges, with measured contacts added to the score.",
    what:
      "For each window, attention is computed only over its graph neighbours. The usual scaled " +
      "dot-product score is shifted by a learned function of the edge's contact strength, its " +
      "genomic distance and the contact resolution before the softmax.",
    why:
      "This is the mechanism the whole project exists to test. The contact map does not get " +
      "concatenated to the output or aligned against it afterwards — it decides which distal windows " +
      "exchange information and how strongly, at every layer, while the representation is still being " +
      "formed. Restricting attention to edges is also what makes the context length affordable on one GPU.",
    math:
      "a_{ij} = \\operatorname*{softmax}_{j \\in \\mathcal{N}(i)}\\left(\\frac{\\mathbf{q}_i^{\\top}\\mathbf{k}_j}{\\sqrt{d}} + b_{\\text{HiC}}(c_{ij}) + b_{\\text{dist}}(d_{ij}) + b_{\\text{scale}}(r)\\right)",
    configRefs: ["model.n_heads", "model.d_model", "model.block_layers"],
  },
  {
    id: "bias",
    label: "Learned edge bias",
    sub: "b_HiC + b_dist + b_scale",
    lane: "fusion",
    kind: "novel",
    novel: true,
    x: 690,
    y: 374,
    w: 290,
    h: 56,
    short: "Three learned functions turn edge features into an additive attention bias.",
    what:
      "Contact strength, genomic separation and contact-map resolution are each passed through a " +
      "small learned function, and the three outputs are summed into a scalar added to the attention " +
      "logit for that edge.",
    why:
      "Learned rather than fixed, because how strongly a given contact should matter is exactly the " +
      "thing nobody knows. Separating distance from contact strength is deliberate: it lets the " +
      "distance-only control arm be built by keeping b_dist and deleting b_HiC, which is what " +
      "distinguishes 'the model learned 3D structure' from 'the model learned that near things touch'.",
    math:
      "\\text{bias}_{ij} = b_{\\text{HiC}}(c_{ij}) + b_{\\text{dist}}(d_{ij}) + b_{\\text{scale}}(r)",
    configRefs: ["data.bin_size"],
  },
  {
    id: "an1",
    label: "Add & Norm",
    lane: "fusion",
    kind: "module",
    x: 352,
    y: 328,
    w: 266,
    h: 30,
    short: "Residual connection and normalisation after attention.",
    what: "The attention output is added back to its input and normalised.",
    why:
      "Standard, and load-bearing here for a specific reason: the residual path means that when " +
      "structure dropout removes the graph, information still flows through the block unchanged " +
      "rather than collapsing.",
  },
  {
    id: "ffn",
    label: "Feed forward",
    lane: "fusion",
    kind: "module",
    x: 352,
    y: 268,
    w: 266,
    h: 44,
    short: "Position-wise transformation applied to each window independently.",
    what:
      "A two-layer network applied to every window separately, widening to an inner dimension and " +
      "projecting back.",
    why:
      "Attention moves information between windows; this is where each window integrates what it " +
      "received. Without it, stacking blocks would just re-average the same vectors.",
    configRefs: ["model.d_ff", "model.dropout"],
  },
  {
    id: "an2",
    label: "Add & Norm",
    lane: "fusion",
    kind: "module",
    x: 352,
    y: 222,
    w: 266,
    h: 30,
    short: "Residual connection and normalisation after the feed-forward layer.",
    what: "The feed-forward output is added back to its input and normalised, closing the block.",
    why: "Keeps the block's output on the same scale as its input so blocks can be stacked.",
  },

  // ------------------------------------------------------------------- heads
  {
    id: "z",
    label: "Structure-informed representations",
    sub: "the transferable artefact",
    lane: "heads",
    kind: "output",
    x: 690,
    y: 228,
    w: 290,
    h: 62,
    short: "The output of the encoder stack, and the thing this project is actually building.",
    what:
      "One vector per genomic window, produced after every block has run. These embeddings are what " +
      "downstream tasks consume, with or without a contact graph available.",
    why:
      "The deliverable is not a Hi-C predictor — it is a reusable sequence representation. The project " +
      "succeeds only if these vectors stay useful when the graph is removed, which is the difference " +
      "between having internalised structure and merely having been given it.",
    configRefs: ["model.d_model", "model.block_layers"],
  },
  {
    id: "head-dna",
    label: "Masked DNA reconstruction",
    lane: "heads",
    kind: "head",
    x: 690,
    y: 40,
    w: 290,
    h: 50,
    short: "Predict masked bases from the surrounding sequence and the contact-linked windows.",
    what:
      "A fraction of the input bases is masked and reconstructed from the structure-informed " +
      "representation.",
    why:
      "The standard self-supervised objective, kept as a regulariser rather than a headline metric. " +
      "Predicting a masked base from a window several hundred kilobases away is close to " +
      "information-free, so this loss is expected to move very little — reporting it as the main " +
      "result would be measuring the wrong thing.",
    math:
      "\\mathcal{L}_{\\text{DNA}} = -\\sum_{t \\in \\mathcal{M}} \\log p\\!\\left(x_t \\mid x_{\\setminus \\mathcal{M}}, H\\right)",
    configRefs: ["train.mask_frac", "train.lambda_dna"],
  },
  {
    id: "head-contrast",
    label: "Contact-aware contrastive",
    sub: "distance-matched negatives",
    lane: "heads",
    kind: "head",
    x: 690,
    y: 100,
    w: 290,
    h: 50,
    short: "Pull contacting windows together, push distance-matched non-contacting pairs apart.",
    what:
      "Window pairs joined by a high-confidence contact are treated as positives. Negatives are drawn " +
      "at the same genomic separation as the positive they are compared against.",
    why:
      "Distance matching is not an optimisation, it is the whole validity of this loss. Sampled " +
      "freely, negatives would sit further apart than positives and the model could minimise the " +
      "objective by learning genomic distance alone, which it can already read off the input.",
    math:
      "\\mathcal{L}_{\\text{contrast}} = -\\log \\frac{\\exp(\\mathrm{sim}(\\mathbf{z}_i,\\mathbf{z}_j)/\\tau)}{\\sum_{k \\in \\mathcal{N}_i}\\exp(\\mathrm{sim}(\\mathbf{z}_i,\\mathbf{z}_k)/\\tau)}",
    configRefs: ["train.temperature", "train.lambda_contrast"],
  },
  {
    id: "head-contact",
    label: "Contact prediction",
    sub: "on held-out edges only",
    lane: "heads",
    kind: "head",
    x: 690,
    y: 160,
    w: 290,
    h: 50,
    short: "Predict contact strength for the edges the encoder was never shown.",
    what:
      "A small head takes two window representations and their genomic separation and predicts the " +
      "normalised contact strength, scored only on the withheld edges.",
    why:
      "The primary training signal, and the one that forces structural information into the " +
      "representation. It is only meaningful because its targets were removed from the conditioning " +
      "graph first — otherwise the head would be copying, and a perfect score would mean nothing.",
    math: "\\hat{c}_{ij} = g(\\mathbf{z}_i, \\mathbf{z}_j, d_{ij})",
    configRefs: ["train.lambda_contact", "data.held_out_edge_frac"],
  },
  {
    id: "loss",
    label: "Total objective",
    lane: "heads",
    kind: "output",
    x: 690,
    y: 520,
    w: 290,
    h: 52,
    short: "The three losses, weighted, optimised jointly from scratch.",
    what:
      "A single weighted sum. Every control arm optimises the same objective with the same weights; " +
      "only the structural signal reaching the encoder differs.",
    why:
      "Matched objectives are what make the comparison a comparison. If the controls trained on a " +
      "different loss, a difference in the result would say nothing about whether measured contacts " +
      "help. Weights are tuned on the validation chromosomes only.",
    math:
      "\\mathcal{L} = \\lambda_{\\text{DNA}}\\mathcal{L}_{\\text{DNA}} + \\lambda_{\\text{contrast}}\\mathcal{L}_{\\text{contrast}} + \\lambda_{\\text{contact}}\\mathcal{L}_{\\text{contact}}",
    configRefs: ["train.lambda_dna", "train.lambda_contrast", "train.lambda_contact"],
  },
];

export const EDGES: ArchEdge[] = [
  { from: "dna", to: "tokenizer", kind: "tensor" },
  { from: "tokenizer", to: "mamba", kind: "tensor" },
  { from: "mamba", to: "window-emb", kind: "tensor" },
  { from: "window-emb", to: "attn", kind: "tensor", label: "Q, K, V" },

  { from: "hic", to: "graph-build", kind: "structure" },
  { from: "graph-build", to: "held-out", kind: "structure" },
  { from: "held-out", to: "struct-dropout", kind: "structure" },
  { from: "struct-dropout", to: "bias", kind: "structure", drops: true },
  { from: "bias", to: "attn", kind: "bias", drops: true },

  { from: "attn", to: "an1", kind: "tensor" },
  { from: "an1", to: "ffn", kind: "tensor" },
  { from: "ffn", to: "an2", kind: "tensor" },
  { from: "an2", to: "z", kind: "tensor" },

  { from: "z", to: "head-contact", kind: "loss" },
  { from: "head-contact", to: "head-contrast", kind: "loss" },
  { from: "head-contrast", to: "head-dna", kind: "loss" },
  { from: "z", to: "loss", kind: "loss" },

  { from: "held-out", to: "head-contact", kind: "target", label: "targets" },
];

/** The block that repeats, drawn as a dashed container. */
export const BLOCK_BOX = { x: 330, y: 200, w: 310, h: 250 } as const;

export const NODE_BY_ID = new Map(NODES.map((n) => [n.id, n]));

/** Tab and prev/next order follows the data flow, not the source order. */
export const TOUR_ORDER: NodeId[] = [
  "dna",
  "tokenizer",
  "mamba",
  "window-emb",
  "hic",
  "graph-build",
  "held-out",
  "struct-dropout",
  "bias",
  "attn",
  "an1",
  "ffn",
  "an2",
  "z",
  "head-dna",
  "head-contrast",
  "head-contact",
  "loss",
];
