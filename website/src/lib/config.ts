import "server-only";

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";

/**
 * The site's numbers come from the project's real training config.
 *
 * Not one numeral below is retyped in JSX. A site that disagrees with the
 * config it describes is worse than no site: it reads as authoritative and is
 * wrong, and the reader has no way to tell.
 *
 * This module is `server-only`. With `output: "export"` it runs once at build
 * time and the values are baked into the HTML, so importing it from a client
 * component is a build error rather than a hydration crash.
 */

export interface DataConfig {
  assembly: string;
  bin_size: number;
  nodes_per_sample: number;
  cell_lines: string[];
  chroms_train: string[];
  chroms_val: string[];
  chroms_test: string[];
  pilot_chroms: string[];
  top_k_edges: number;
  local_radius: number;
  min_separation: number;
  held_out_edge_frac: number;
  max_n_frac: number;
}

export interface ModelConfig {
  d_model: number;
  encoder_layers: number;
  d_state: number;
  d_conv: number;
  expand: number;
  block_layers: number;
  n_heads: number;
  d_ff: number;
  dropout: number;
  structure_dropout: number;
}

export interface TrainConfig {
  arm: string;
  steps: number;
  batch_size: number;
  lr: number;
  warmup_steps: number;
  weight_decay: number;
  grad_clip: number;
  precision: string;
  grad_checkpoint: boolean;
  mask_frac: number;
  temperature: number;
  lambda_dna: number;
  lambda_contrast: number;
  lambda_contact: number;
  ckpt_every: number;
  eval_every: number;
  log_every: number;
}

export interface BaseConfig {
  seed: number;
  control_seed: number;
  data: DataConfig;
  model: ModelConfig;
  train: TrainConfig;
}

/** `.generated/base.yaml` is staged by scripts/sync-assets.mjs; `../configs` is the dev fallback. */
function locate(): string {
  const staged = join(process.cwd(), ".generated", "base.yaml");
  if (existsSync(staged)) return staged;
  const source = join(process.cwd(), "..", "configs", "base.yaml");
  if (existsSync(source)) return source;
  throw new Error(
    "base.yaml not found. Run `node scripts/sync-assets.mjs` (npm run dev does this for you).",
  );
}

export const config = parse(readFileSync(locate(), "utf8")) as BaseConfig;

/** Read a dotted path, e.g. cfg("model.d_model"). Verified at build time by sync-assets.mjs. */
export function cfg(path: string): unknown {
  return path
    .split(".")
    .reduce<unknown>(
      (o, k) => (o == null ? undefined : (o as Record<string, unknown>)[k]),
      config as unknown,
    );
}

/** Values the client components need, serialised across the server/client boundary. */
export interface PublicConfig {
  binSize: number;
  nodesPerSample: number;
  contextBp: number;
  topKEdges: number;
  localRadius: number;
  minSeparation: number;
  heldOutEdgeFrac: number;
  structureDropout: number;
  dModel: number;
  encoderLayers: number;
  blockLayers: number;
  nHeads: number;
  dFf: number;
  controlSeed: number;
  maskFrac: number;
  temperature: number;
  lambdaDna: number;
  lambdaContrast: number;
  lambdaContact: number;
  assembly: string;
  cellLines: string[];
  chromsTrain: string[];
  chromsVal: string[];
  chromsTest: string[];
  pilotChroms: string[];
  /** Every key/value pair, flattened to dotted paths, for the config-reference badges. */
  flat: Record<string, string>;
}

function flatten(obj: unknown, prefix = "", out: Record<string, string> = {}) {
  if (obj === null || typeof obj !== "object") return out;
  for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) flatten(v, path, out);
    else out[path] = Array.isArray(v) ? v.join(", ") : String(v);
  }
  return out;
}

export function publicConfig(): PublicConfig {
  const { data, model, train } = config;
  return {
    binSize: data.bin_size,
    nodesPerSample: data.nodes_per_sample,
    contextBp: data.bin_size * data.nodes_per_sample,
    topKEdges: data.top_k_edges,
    localRadius: data.local_radius,
    minSeparation: data.min_separation,
    heldOutEdgeFrac: data.held_out_edge_frac,
    structureDropout: model.structure_dropout,
    dModel: model.d_model,
    encoderLayers: model.encoder_layers,
    blockLayers: model.block_layers,
    nHeads: model.n_heads,
    dFf: model.d_ff,
    controlSeed: config.control_seed,
    maskFrac: train.mask_frac,
    temperature: train.temperature,
    lambdaDna: train.lambda_dna,
    lambdaContrast: train.lambda_contrast,
    lambdaContact: train.lambda_contact,
    assembly: data.assembly,
    cellLines: data.cell_lines,
    chromsTrain: data.chroms_train,
    chromsVal: data.chroms_val,
    chromsTest: data.chroms_test,
    pilotChroms: data.pilot_chroms,
    flat: flatten(config),
  };
}

/** 640000 -> "640 kb", 5000 -> "5 kb". */
export function kb(bp: number): string {
  return bp >= 1_000_000 ? `${bp / 1_000_000} Mb` : `${bp / 1000} kb`;
}
