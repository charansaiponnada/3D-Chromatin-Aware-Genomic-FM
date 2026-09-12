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
        "\\frac{\\mathbf{q}_i^{\\top}\\mathbf{k}_j}{\\sqrt{d}}" +
        " + \\htmlStyle{color:var(--struct)}{b_{\\text{HiC}}(c_{ij})}" +
        " + \\htmlStyle{color:var(--seq)}{b_{\\text{dist}}(d_{ij})}" +
        " + \\htmlStyle{color:var(--head)}{b_{\\text{scale}}(r)}\\right)",
    ),
    total: renderMath(
      "\\mathcal{L} = \\lambda_{\\text{DNA}}\\mathcal{L}_{\\text{DNA}}" +
        " + \\lambda_{\\text{contrast}}\\mathcal{L}_{\\text{contrast}}" +
        " + \\lambda_{\\text{contact}}\\mathcal{L}_{\\text{contact}}",
    ),
    dna: renderMath(
      "\\mathcal{L}_{\\text{DNA}} = -\\sum_{t \\in \\mathcal{M}} \\log p\\!\\left(x_t \\mid x_{\\setminus \\mathcal{M}}, H\\right)",
    ),
    contrast: renderMath(
      "\\mathcal{L}_{\\text{contrast}} = -\\log \\frac{\\exp(\\mathrm{sim}(\\mathbf{z}_i,\\mathbf{z}_j)/\\tau)}{\\sum_{k \\in \\mathcal{N}_i}\\exp(\\mathrm{sim}(\\mathbf{z}_i,\\mathbf{z}_k)/\\tau)}",
    ),
    contact: renderMath("\\hat{c}_{ij} = g(\\mathbf{z}_i, \\mathbf{z}_j, d_{ij})"),
  };
}

export type PageMath = ReturnType<typeof pageMathHtml>;
