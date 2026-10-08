import "server-only";

import katex from "katex";

import { NODES, type NodeId } from "@/content/architecture";

/**
 * KaTeX runs at build time, never in the browser.
 *
 * The formulas are fixed strings in content/architecture.ts, so rendering them
 * per-visit would ship ~270 kB of maths engine to redraw the same output. The
 * server renders each one once and the client components receive HTML strings.
 */

export function renderMath(tex: string, displayMode = true): string {
  return katex.renderToString(tex, {
    displayMode,
    throwOnError: false,
    output: "html",
    strict: false,
    // \htmlStyle is gated behind trust. The input is our own literal strings in
    // content/architecture.ts, never anything a reader supplies, so enabling it
    // lets the bias terms in the attention formula carry the same colours as the
    // boxes they refer to in the diagram.
    trust: true,
  });
}

/** Pre-rendered maths for every architecture node that has some. */
export function nodeMathHtml(): Partial<Record<NodeId, string>> {
  const out: Partial<Record<NodeId, string>> = {};
  for (const node of NODES) {
    if (node.math) out[node.id] = renderMath(node.math);
  }
  return out;
}

/** Formulas used outside the architecture diagram. */
export function pageMathHtml() {
  return {
    attention: renderMath(
      "a_{ij} = \\operatorname*{softmax}_{j \\in \\mathcal{N}(i)}\\left(" +
        "\\frac{\\mathbf{q}_i^{\\top}\\mathbf{k}_j}{\\sqrt{d_h}}" +
        " + \\htmlStyle{color:var(--struct)}{b_{\\text{HiC}}(c_{ij})}" +
        " + \\htmlStyle{color:var(--seq)}{b_{\\text{dist}}(d_{ij})}\\right)",
    ),
    total: renderMath(
      "\\mathcal{L} = \\lambda_{\\text{mask}}\\mathcal{L}_{\\text{mask}}" +
        " + \\lambda_{\\text{con}}\\mathcal{L}_{\\text{con}}" +
        " + \\lambda_{\\text{contact}}\\mathcal{L}_{\\text{contact}}",
    ),
    dna: renderMath(
      "\\mathcal{L}_{\\text{mask}} = \\frac{1}{|\\mathcal{M}|}\\sum_{i \\in \\mathcal{M}} \\left\\| g(\\mathbf{z}_i) - \\mathrm{sg}(\\mathbf{h}_i) \\right\\|_2^2",
    ),
    contrast: renderMath(
      "\\mathcal{L}_{\\text{con}} = -\\log \\frac{\\exp(\\tilde{\\mathbf{z}}_i^{\\top}\\tilde{\\mathbf{z}}_j/\\tau)}{\\exp(\\tilde{\\mathbf{z}}_i^{\\top}\\tilde{\\mathbf{z}}_j/\\tau) + \\sum_{(k,l) \\in \\mathcal{D}_{ij}}\\exp(\\tilde{\\mathbf{z}}_k^{\\top}\\tilde{\\mathbf{z}}_l/\\tau)}",
    ),
    contact: renderMath(
      "\\hat{c}_{ij} = \\sigma\\!\\left(\\mathrm{MLP}\\left([\\,\\mathbf{z}_i \\odot \\mathbf{z}_j,\\ |\\mathbf{z}_i - \\mathbf{z}_j|,\\ \\log_2(1+d_{ij})\\,]\\right)\\right)",
    ),
  };
}

export type PageMath = ReturnType<typeof pageMathHtml>;
