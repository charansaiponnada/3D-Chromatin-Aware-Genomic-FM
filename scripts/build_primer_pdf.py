"""Assemble the print edition of the primer and render it to PDF.

Reuses the artifact primer's body verbatim, swaps the web masthead for a print
cover, and appends four reference appendices so the PDF is self-contained.
Renders via headless Chrome, then stamps page numbers with reportlab/pypdf
(Chromium does not support CSS @page margin boxes, so numbering is a post-pass).

Run:  python scripts/build_primer_pdf.py
"""

from __future__ import annotations

import io
import re
import subprocess
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OUT_PDF = DOCS / "3D-Chromatin-Aware-Genomic-FM-primer.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

PRINT_CSS = """
@page { size: A4; margin: 19mm 17mm 20mm; }

:root {
  --ground: #FFFFFF; --surface: #FFFFFF; --sunk: #F4F5F7;
  --ink: #16191F; --ink-soft: #545C68; --ink-faint: #7C8492;
  --rule: #D8DCE3; --rule-soft: #EAECF0;
  --structure: #A83426; --structure-wash: #FBF1EF;
  --sequence: #1F5891; --sequence-wash: #EEF3F9;
  --confirm: #24693B; --warn: #8A5710; --warn-wash: #FBF5E9;
  --serif: Constantia, "Iowan Old Style", Charter, Georgia, serif;
  --sans: "Segoe UI", system-ui, Arial, sans-serif;
  --mono: Consolas, "Cascadia Mono", ui-monospace, monospace;
  --col: 100%; --wide: 100%;
}

* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

html, body { background: #fff; }
body { font-family: var(--serif); font-size: 10.4pt; line-height: 1.52; color: var(--ink); }
.page { max-width: none; margin: 0; padding: 0; }
.col { max-width: none; }

p, ul, ol, .note, .callout, table, .formula { margin: 0 0 0.85em; }
h1, h2, h3, h4 { text-wrap: balance; margin: 0; font-weight: 600; break-after: avoid; }
h1 { font-size: 22pt; line-height: 1.1; letter-spacing: -0.02em; }
h2 { font-size: 16pt; line-height: 1.16; margin-bottom: 0.4em; letter-spacing: -0.012em; }
h3 { font-size: 11.6pt; margin: 1.5em 0 0.35em; }
h4 { font-family: var(--sans); font-size: 8pt; text-transform: uppercase;
     letter-spacing: 0.09em; color: var(--ink-soft); margin: 1.4em 0 0.45em; font-weight: 700; }
.eyebrow { font-family: var(--sans); font-size: 7.5pt; text-transform: uppercase;
           letter-spacing: 0.14em; color: var(--ink-faint); font-weight: 700; }

/* ---- cover ---- */
.cover { break-after: page; padding-top: 34mm; }
.cover h1 { font-size: 30pt; max-width: 20ch; }
.cover .sub { font-size: 12.5pt; color: var(--ink-soft); margin-top: 1.1rem; max-width: 46ch; }
.cover .rule { height: 3px; background: var(--ink); margin: 1.6rem 0; width: 100%; }
.cover .meta { font-family: var(--sans); font-size: 8.6pt; color: var(--ink-faint);
               margin-top: 30mm; display: grid; gap: 0.3rem; }
.cover .warnbox { margin-top: 12mm; border-left: 3px solid var(--structure);
                  background: var(--structure-wash); padding: 0.9rem 1.1rem; font-size: 9.6pt; }
.cover .strip { display: flex; gap: 4px; margin: 1.4rem 0 0; }
.cover .strip i { display: block; height: 10px; flex: 1; }

/* ---- toc ---- */
.toc-print { break-after: page; }
.toc-print ol { list-style: none; padding: 0; margin: 1.2rem 0 0; }
.toc-print li { display: flex; gap: 1rem; padding: 0.42rem 0;
                border-bottom: 1px solid var(--rule-soft); font-family: var(--sans); font-size: 9.6pt; }
.toc-print .n { font-family: var(--mono); color: var(--structure); width: 2rem; flex: none; }

/* ---- parts ---- */
.part { break-before: page; padding-top: 0; margin-top: 0; border-top: 0; }
.part-head { display: flex; gap: 1.1rem; align-items: baseline; margin-bottom: 1.1rem;
             border-bottom: 2px solid var(--ink); padding-bottom: 0.6rem; }
.part-num { font-family: var(--mono); font-size: 10pt; color: var(--structure); flex: none; padding-top: 0.3em; }

/* ---- figures ---- */
figure { margin: 1.3rem 0; break-inside: avoid; }
figure svg, figure img { display: block; width: 100%; height: auto; }
.fig-frame { background: #fff; border: 1px solid var(--rule); padding: 0.9rem; overflow: visible; }
figcaption { font-family: var(--sans); font-size: 8.4pt; line-height: 1.45; color: var(--ink-soft);
             margin-top: 0.6rem; padding-left: 0.7rem; border-left: 2px solid var(--rule); max-width: none; }

.d-line { stroke: currentColor; fill: none; }
.d-fill { fill: currentColor; }
.d-box { fill: none; stroke: currentColor; stroke-width: 1.4; }
.d-struct { stroke: var(--structure); fill: none; }
.d-struct-f { fill: var(--structure); }
.d-seq { stroke: var(--sequence); fill: none; }
.d-seq-f { fill: var(--sequence); }
.d-ok-f { fill: var(--confirm); }
.d-ok { stroke: var(--confirm); fill: none; }
.d-faint { stroke: var(--ink-faint); fill: none; }
.d-faint-f { fill: var(--ink-faint); }
.d-wash { fill: var(--sunk); stroke: none; }
text { font-family: var(--sans); fill: currentColor; }
.t-lbl { font-size: 12px; } .t-sm { font-size: 10.5px; fill: var(--ink-soft); }
.t-mono { font-family: var(--mono); font-size: 11px; } .t-bold { font-weight: 650; }
.t-struct { fill: var(--structure); } .t-seq { fill: var(--sequence); } .t-ok { fill: var(--confirm); }

/* ---- callouts / tables ---- */
.callout { border-left: 3px solid var(--sequence); background: var(--sequence-wash);
           padding: 0.75rem 0.95rem; font-size: 9.6pt; break-inside: avoid; }
.callout.warn { border-left-color: var(--warn); background: var(--warn-wash); }
.callout.struct { border-left-color: var(--structure); background: var(--structure-wash); }
.callout .lbl { font-family: var(--sans); font-size: 7.4pt; text-transform: uppercase;
                letter-spacing: 0.11em; font-weight: 700; display: block; margin-bottom: 0.3rem; color: var(--sequence); }
.callout.warn .lbl { color: var(--warn); } .callout.struct .lbl { color: var(--structure); }

.tw { overflow: visible; margin: 1rem 0; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-family: var(--sans); font-size: 8.6pt; }
th, td { text-align: left; padding: 0.36rem 0.7rem 0.36rem 0; border-bottom: 1px solid var(--rule-soft); vertical-align: top; }
th { font-size: 7.4pt; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-faint); border-bottom: 1px solid var(--rule); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); padding-right: 0; }
tbody tr:last-child td { border-bottom: 0; }
table.wide { font-size: 7.9pt; }

.formula { font-family: var(--mono); font-size: 8.6pt; background: var(--sunk);
           border: 1px solid var(--rule); padding: 0.75rem 0.9rem; line-height: 1.62;
           white-space: pre-wrap; break-inside: avoid; }
code { font-family: var(--mono); font-size: 0.9em; background: var(--sunk); padding: 0.05em 0.25em; }
ul, ol { padding-left: 1.15rem; } li { margin-bottom: 0.3em; }
a { color: var(--sequence); text-decoration: none; }
.kicker { color: var(--structure); font-weight: 650; } .quiet { color: var(--ink-soft); }
hr.soft { border: 0; border-top: 1px solid var(--rule); margin: 1.6rem 0; }
dfn { font-style: normal; font-weight: 650; }
.term { border-bottom: 1px dotted var(--structure); }

.status-grid { display: grid; gap: 0.5rem; margin: 1rem 0; }
.status-row { display: grid; grid-template-columns: 4.6rem 1fr; gap: 0.8rem;
              padding-bottom: 0.5rem; border-bottom: 1px solid var(--rule-soft); font-size: 9.6pt; break-inside: avoid; }
.status-row .st { font-family: var(--sans); font-size: 7.2pt; text-transform: uppercase;
                  letter-spacing: 0.08em; font-weight: 700; padding-top: 0.3em; }
.st.done { color: var(--confirm); } .st.now { color: var(--structure); } .st.next { color: var(--ink-faint); }
.masthead { display: none; }
"""

COVER = """
<section class="cover">
  <span class="eyebrow">Project primer &middot; written for someone starting from zero</span>
  <div class="rule"></div>
  <h1>Teaching a DNA model that the genome is folded</h1>
  <p class="sub">Everything needed to understand this research: what DNA and chromatin are,
  what a foundation model is, what the project proposes to build, and exactly where it stands.</p>
  <div class="strip">
    <i style="background:var(--structure)"></i><i style="background:var(--sequence)"></i>
    <i style="background:var(--confirm)"></i><i style="background:var(--sunk)"></i>
    <i style="background:var(--sunk)"></i><i style="background:var(--sunk)"></i>
  </div>
  <div class="warnbox">
    <strong>No results in this document.</strong> No model has been trained. The abstract in Part 5
    is a proposal abstract. Every number here comes from a data-processing or accounting script in
    the repository, never from an experiment, because no experiment has been run.
  </div>
  <div class="meta">
    <span>Repository &mdash; 3D-Chromatin-Aware-Genomic-FM</span>
    <span>Status &mdash; Phase 0, 1 and 2 complete; Phase 3 (baselines) next</span>
    <span>Compiled 7 August 2026</span>
  </div>
</section>

<section class="toc-print">
  <h2>Contents</h2>
  <ol>
    <li><span class="n">01</span><span>DNA, genes, and the problem that starts everything</span></li>
    <li><span class="n">02</span><span>The genome is folded, and the folding matters</span></li>
    <li><span class="n">03</span><span>How folding gets measured, and what the data looks like</span></li>
    <li><span class="n">04</span><span>Foundation models, and the two that matter here</span></li>
    <li><span class="n">05</span><span>The research: title, abstract, and the gap it fills</span></li>
    <li><span class="n">06</span><span>The mechanism, in equations you can read</span></li>
    <li><span class="n">07</span><span>Where the project actually stands</span></li>
    <li><span class="n">08</span><span>Glossary</span></li>
    <li><span class="n">A</span><span>The competitive landscape, paper by paper</span></li>
    <li><span class="n">B</span><span>Mechanism specification and its controls</span></li>
    <li><span class="n">C</span><span>Data card and the validation figure</span></li>
    <li><span class="n">D</span><span>Decisions, open questions, and the repository map</span></li>
  </ol>
</section>
"""

APPENDICES = """
<section class="part" id="apA">
  <div class="part-head"><span class="part-num">A</span><h2>The competitive landscape, paper by paper</h2></div>

  <p>Eleven papers define the space this project sits in. The column that decides everything is the
  fourth: when, during training, does 3D structure first influence the model.</p>

  <div class="tw"><table class="wide">
    <thead><tr><th>Work</th><th>Sequence in?</th><th>Structure in?</th><th>Structure enters at</th><th>Self-supervised?</th><th>Structure needed at inference?</th></tr></thead>
    <tbody>
      <tr><td>Lieberman-Aiden 2009 / Dixon 2012 / Rao 2014</td><td>&mdash;</td><td>&mdash;</td><td>assay and descriptive work</td><td>&mdash;</td><td>&mdash;</td></tr>
      <tr><td>HiCFoundation (Nat Methods 2026)</td><td>no</td><td>yes</td><td>pretraining</td><td>yes</td><td>yes</td></tr>
      <tr><td>Hi-Cformer (bioRxiv 2025)</td><td>no</td><td>yes</td><td>pretraining</td><td>yes, masked block reconstruction</td><td>yes</td></tr>
      <tr><td>Evo 2 (Nature 2026)</td><td>yes</td><td><strong>no</strong></td><td>&mdash;</td><td>yes, autoregressive</td><td>no</td></tr>
      <tr><td>Caduceus (ICML 2024)</td><td>yes</td><td><strong>no</strong></td><td>&mdash;</td><td>yes, masked LM</td><td>no</td></tr>
      <tr><td>GraphReg (Genome Res 2022)</td><td>features</td><td>yes</td><td>supervised, downstream</td><td>no</td><td>yes</td></tr>
      <tr><td>CHROME (Brief Bioinform 2026)</td><td>yes</td><td>yes</td><td><strong>stage 2, post-hoc</strong></td><td>no, supervised on ChIP-seq</td><td><strong>yes</strong></td></tr>
      <tr><td>Akita 2020 / Orca 2022</td><td>yes</td><td>yes, as target</td><td>supervised training target</td><td>no</td><td>no</td></tr>
      <tr><td>Evo2HiC (bioRxiv 2025)</td><td>yes</td><td>yes</td><td>contrastive distillation</td><td>no, frozen teacher</td><td>no, for the DNA-only encoder</td></tr>
      <tr><td><strong>This project</strong></td><td><strong>yes</strong></td><td><strong>yes</strong></td><td><strong>self-supervised pretraining</strong></td><td><strong>yes, masked LM plus structure</strong></td><td><strong>target: no</strong></td></tr>
    </tbody>
  </table></div>

  <h3>The two nearest competitors</h3>

  <p><span class="kicker">CHROME</span> &mdash; Ye, Du, Chen, Dai, Ma and Liang, <em>Briefings in Bioinformatics</em> 27(4):bbag360,
  17 July 2026. Filters Hi-C against a self-avoiding polymer null model simulated over 500,000 chains,
  keeping only 1.8&ndash;6% of contacts as physically specific. Builds a subgraph per ChIP-seq peak with
  a 4 Mb receptive field at 5 kb bins, then runs two graph-attention layers. Trained in two stages:
  the encoder is trained on centre nodes alone until validation saturates, and only afterwards are early
  layers frozen and the graph module added. Supervised on 751 ENCODE assays across GM12878, K562 and
  IMR-90, with chr9 held out. Structure never influences what the sequence encoder learns.</p>

  <p><span class="kicker">Evo2HiC</span> &mdash; Fang, Wang, Xiao, Hang, Murtaza, Yang, Xu, Jha, Noble and Wang,
  bioRxiv 10.1101/2025.11.18.689171, November 2025. Distils a frozen Evo 2 (7B) into a 3.6M-parameter
  seven-layer CNN using two SigLIP contrastive objectives: one aligns the student to Evo 2, the second
  aligns it to Hi-C patch embeddings. Beats Orca by 10.9% Spearman on contact-map prediction and
  generalises across 177 DNA Zoo species. Structure genuinely shapes a sequence encoder &mdash; but by
  aligning output embeddings, with no objective over nucleotides anywhere in the model. A keyword count
  over the full preprint returns zero occurrences of ClinVar, pathogenic, eQTL, variant, masked, MLM,
  Mamba and state space.</p>

  <p>Their third stated limitation reads: <em>"in the current work we extracted embeddings from Evo 2 by
  freezing its parameters, as the model's size makes fine-tuning computationally challenging. We plan to
  develop efficient fine-tuning strategies that enable adapting Evo 2 with Hi-C data."</em> A well-resourced
  group has named this project's question as work they intend to do and have not done.</p>

  <h3>Two findings that changed the plan</h3>

  <p><span class="kicker">Lee, arXiv:2604.07196, 8 April 2026.</span> Probed Evo 2 (7B) on 231 TAD boundary
  regions and 120 convergent CTCF loops from H1-ESC Micro-C. Boundary deletions were penalised
  <em>less</em> than GC- and size-matched random controls (paired Wilcoxon p = 0.405); CTCF motif
  inversions and deletions likewise (p = 0.021 and p = 0.006, in the wrong direction). Generated loops
  reached a median enrichment of 0.054 against a reference of 0.388. The paper concludes Evo 2
  <em>"has learned local CTCF grammar but misses higher-order 3D organization"</em> and prescribes
  bidirectional architectures with explicit 3D contact inputs, citing Caduceus. Preprint, not yet
  peer reviewed.</p>

  <p><span class="kicker">DNALongBench, <em>Nature Communications</em> 16(1):10108, 2025.</span> Five
  long-range tasks with dependencies to 1 Mb, including enhancer&ndash;target gene interaction, eQTL and
  contact-map prediction, already benchmarking Caduceus-Ph and Akita. Adopting it for Phase 5 removes
  the objection that the evaluation tasks were chosen to flatter. Calibration: contact-map prediction is
  the hardest task in the suite, best reported correlation 0.233, by Akita.</p>

  <h3>The negative result the project rests on</h3>
  <p>A forward-citation sweep over 318 papers &mdash; 118 citing Orca, 200 citing Akita &mdash; found no
  work that takes either model's learned representations and evaluates them on a task other than
  predicting folding. Their variant applications all run the model twice, on reference and mutant
  sequence, and compare predicted contact maps. Internal embeddings are never extracted, frozen and
  probed, or fine-tuned into another head.</p>
</section>

<section class="part" id="apB">
  <div class="part-head"><span class="part-num">B</span><h2>Mechanism specification and its controls</h2></div>

  <h3>The baseline recurrence</h3>
  <div class="formula">(z, v)          = split(in_proj(u_t))
x_t             = SiLU(DepthwiseConv1d(v)_t)
(d', B_t, C_t)  = split(x_proj(x_t))
D_t             = softplus(W_dt d' + b_dt)          &gt; 0
A[i,n]          = -exp(A_log[i,n])                  &lt; 0
Abar_t[i,n]     = exp(D_t[i] * A[i,n])              in (0,1)
h_t[i,n]        = Abar_t[i,n] * h_{t-1}[i,n] + D_t[i]*B_t[n]*x_t[i]
y_t[i]          = sum_n C_t[n]*h_t[i,n] + D[i]*x_t[i]</div>

  <h3>The two added terms</h3>
  <div class="formula">s_t     = E(phi_bin(t))                     structural context, 8 -&gt; 2 dims

D_t     = softplus( W_dt d' + b_dt + <b>W_s . s_t</b> )     per-channel timescale
p_t     = softplus( <b>w_g . s_t + b_g</b> )   &gt;= 0        scalar permeability penalty
Abar_t  = exp( D_t * A <b>- p_t</b> )                       decay, now structure-aware</div>

  <p>Initialised at <code>W_s = 0</code>, <code>w_g = 0</code>, <code>b_g = &minus;8</code>, so the model is
  numerically indistinguishable from the baseline at step zero and any divergence is attributable to learned
  structural signal. As <code>p</code> grows without bound, <code>Abar</code> goes to zero and the state resets
  outright &mdash; so this mechanism contains hard TAD-boundary state resets as its limiting case.</p>

  <h3>Parameter accounting, computed by instantiating both models</h3>
  <div class="tw"><table>
    <thead><tr><th>Model</th><th class="num">parameters</th></tr></thead>
    <tbody>
      <tr><td>Sequence-only baseline</td><td class="num">7,725,312</td></tr>
      <tr><td>With structural bias</td><td class="num">7,758,354</td></tr>
      <tr><td>Added</td><td class="num">33,042</td></tr>
      <tr><td>Difference against a 5% budget</td><td class="num">+0.4277%</td></tr>
    </tbody>
  </table></div>

  <h3>The shuffled-structure controls</h3>
  <div class="tw"><table class="wide">
    <thead><tr><th>ID</th><th>Control</th><th>Destroys</th><th>Preserves</th><th>Rules out</th></tr></thead>
    <tbody>
      <tr><td>S0</td><td>zero the structural signal</td><td>everything</td><td>&mdash;</td><td>establishes the sequence-only floor at identical parameter count</td></tr>
      <tr><td>S1</td><td>permute across all bins</td><td>alignment and autocorrelation</td><td>the marginal distribution</td><td>primary reliance probe</td></tr>
      <tr><td>S2</td><td>circular shift by 10 Mb</td><td>alignment only</td><td>marginal and autocorrelation</td><td>"any smooth auxiliary channel would do"</td></tr>
      <tr><td>S3</td><td>rewire preserving distance decay</td><td>locus-specific structure</td><td>the distance-explainable part</td><td><strong>"structure is genomic distance in disguise"</strong></td></tr>
    </tbody>
  </table></div>

  <p>S3 is the control that can end the project. Contact probability falls smoothly with genomic distance
  under the fractal-globule model of Lieberman-Aiden 2009. If a model performs as well on distance-matched
  fake structure as on real structure, what it learned is a re-parameterised positional prior and the
  central claim is false however good the headline number looks.</p>

  <p>The gate passes only if, feeding shuffled structure to a real-structure-trained model, the degradation
  is positive with a 95% bootstrap confidence interval excluding zero <em>and</em> is at least twice the
  across-seed standard deviation of the real runs. Statistical significance alone is not enough; the
  effect-size floor is there to stop a p-value standing in for an effect.</p>

  <p>Expected ordering, as a falsifiable ladder: real &lt; zeroed &le; distance-matched &le; permuted.
  If distance-matched performs like real, stop. If permuted performs like real, the mechanism is inert
  and Phase 5 should not begin.</p>

  <h3>The eight ways this fails quietly</h3>
  <div class="tw"><table class="wide">
    <thead><tr><th>&nbsp;</th><th>Failure</th><th>What it looks like</th><th>Status</th></tr></thead>
    <tbody>
      <tr><td>F1</td><td>bias absorbed into the existing offset</td><td>trains to exactly baseline loss</td><td>highest probability; mitigated by genome-wide standardisation</td></tr>
      <tr><td>F2</td><td>softplus saturation kills the gradient</td><td>structural weights never move</td><td>keep the reference dt initialisation</td></tr>
      <tr><td>F3</td><td>structure inferable from sequence</td><td>structural input redundant</td><td><strong>largely defused</strong> by Lee's Evo 2 probe</td></tr>
      <tr><td>F4</td><td>Hi-C bin longer than the memory horizon</td><td>trains to baseline loss</td><td><strong>precondition confirmed present</strong>; see below</td></tr>
      <tr><td>F5</td><td>permeability gate never leaves zero</td><td>only half the mechanism is live</td><td>monitor the distribution of p</td></tr>
      <tr><td>F6</td><td>gradient starvation against the main loss</td><td>structural pathway never learns</td><td>0.43% of parameters behind two nonlinearities</td></tr>
      <tr><td>F7</td><td>forward and reverse passes cancel</td><td>trains to baseline loss</td><td>two of eight features flip sign under reverse complement</td></tr>
      <tr><td>F8</td><td>structural encoder collapses</td><td>reduces to F1</td><td>check the rank of the encoded signal</td></tr>
    </tbody>
  </table></div>

  <p>F1, F4 and F7 produce the same symptom: the model trains to baseline performance and raises no error.
  Four diagnostics therefore run continuously rather than at the end &mdash; the variance the structural
  pathway contributes to the timescale, the distribution of the permeability penalty, the growth of the
  structural weight norm, and a probe asking whether the baseline model's hidden states already predict
  the structural signal.</p>

  <div class="callout warn">
    <span class="lbl">F4, measured</span>
    At the reference initialisation for this model size, memory horizons run from 0.6 tokens at the
    fast end to 994 tokens at the slowest, with a median of 14.9. A 5 kb Hi-C bin is 5,000 tokens, so
    not one channel-state pair spans a single bin. Sixteen layers relaying extends the reach to roughly
    15,900 tokens, still short of a 385 kb TAD. The timescale is learned and can grow, so this is not
    proof of failure &mdash; but it constrains the sequence-only baseline exactly as much as the structural
    model, and Phase 3 must now measure the trained memory length before Phase 4 spends GPU time.
  </div>
</section>

<section class="part" id="apC">
  <div class="part-head"><span class="part-num">C</span><h2>Data card and the validation figure</h2></div>

  <h3>Provenance</h3>
  <div class="tw"><table>
    <thead><tr><th>Item</th><th>Source</th></tr></thead>
    <tbody>
      <tr><td>Experiment set</td><td>4DN 4DNES3JX38V5 &mdash; in situ Hi-C, GM12878, MboI + bio-dATP</td></tr>
      <tr><td>Publication</td><td>Rao SS et al. 2014, PMID 25497547, Lieberman Aiden lab</td></tr>
      <tr><td>Contact matrix</td><td>4DNFIXP4QG5B, 27.4 GB multi-resolution cooler &mdash; read remotely, never downloaded</td></tr>
      <tr><td>Boundary calls</td><td>4DNFIVK5JOFU &mdash; validation target, not model input</td></tr>
      <tr><td>Insulation track</td><td>4DNFIBMOGOZC &mdash; validation target</td></tr>
      <tr><td>Compartment track</td><td>4DNFILYQ1PAY &mdash; validation target</td></tr>
      <tr><td>Loop evidence</td><td>4DNFI9SL1WSF &mdash; GM12878 CTCF in situ ChIA-PET, GRCh38</td></tr>
      <tr><td>Reference sequence</td><td>Ensembl release 113, GRCh38 chromosome 9</td></tr>
      <tr><td>Annotation</td><td>GENCODE release 47</td></tr>
    </tbody>
  </table></div>

  <p>4D Nucleome's download endpoint returns HTTP 403 to unauthenticated clients; the pipeline resolves
  each accession through the API and reads the public bucket field instead, which supports range requests.
  Ensembl names the chromosome <code>9</code> while 4DN and GENCODE name it <code>chr9</code>; the pipeline
  rewrites the label and touches no coordinates. A mismatch there would produce an empty join rather than
  an error, and would look exactly like failure mode F1.</p>

  <h3>What was measured</h3>
  <div class="tw"><table>
    <thead><tr><th>Property</th><th class="num">value</th></tr></thead>
    <tbody>
      <tr><td>chr9 length, from the contact matrix</td><td class="num">138,394,717 bp</td></tr>
      <tr><td>chr9 length, from the Ensembl FASTA</td><td class="num">138,394,717 bp</td></tr>
      <tr><td>Bins at 5 kb</td><td class="num">27,679</td></tr>
      <tr><td>Band entries retained</td><td class="num">7,108,483</td></tr>
      <tr><td>Band occupancy</td><td class="num">0.6451</td></tr>
      <tr><td>Bins with no balancing weight</td><td class="num">19.31%</td></tr>
      <tr><td>Unresolved sequence (N)</td><td class="num">12.00%</td></tr>
      <tr><td>Bins with a complete feature vector</td><td class="num">21,519 (77.74%)</td></tr>
      <tr><td>Our 100 kb insulation vs 4DN's track</td><td class="num">r = +0.9969</td></tr>
      <tr><td>Our compartment eigenvector vs 4DN's</td><td class="num">r = +0.9759</td></tr>
    </tbody>
  </table></div>

  <h3>The eight structural features</h3>
  <div class="tw"><table>
    <thead><tr><th>&nbsp;</th><th>Feature</th><th>Window</th><th>Behaviour under reverse complement</th></tr></thead>
    <tbody>
      <tr><td>0&ndash;2</td><td>insulation score</td><td>100, 250, 500 kb</td><td>symmetric</td></tr>
      <tr><td>3</td><td>directionality index</td><td>2 Mb</td><td><strong>antisymmetric</strong></td></tr>
      <tr><td>4</td><td>log contact density</td><td>2 Mb</td><td>symmetric</td></tr>
      <tr><td>5</td><td>upstream mass fraction</td><td>2 Mb</td><td><strong>antisymmetric</strong></td></tr>
      <tr><td>6</td><td>short-to-long range ratio</td><td>100 kb / 2 Mb</td><td>symmetric</td></tr>
      <tr><td>7</td><td>A/B compartment eigenvector</td><td>250 kb, GC-oriented</td><td>symmetric</td></tr>
    </tbody>
  </table></div>

  <p>The last column is load-bearing rather than descriptive. Failure mode F7 is precisely the case where
  features 3 and 5 acquire opposite signs in the forward and reverse passes and cancel, leaving a mechanism
  that looks inert while in fact running two live pathways against each other.</p>

  <figure>
    <img src="../figures/phase1_validation.png" alt="Six-panel validation figure showing the Hi-C contact map for chr9:132-138 Mb with TAD boundaries and CTCF ChIA-PET loops, the insulation track, whole-chromosome compartments, gene positions, and zoomed views of one TAD and one loop."/>
    <figcaption>The Phase 1 validation figure. Panel A: balanced contacts with 4DN boundary calls in blue
    and CTCF ChIA-PET loops in green. B: our insulation score, minima falling on the boundary lines.
    C: compartments across all of chr9, with the centromeric gap at 43&ndash;60 Mb visible as a blank
    interval. D: GENCODE genes with coordinates read from our own annotation file. E: a 385 kb TAD,
    brighter inside than out. F: a 140 kb loop anchored at NOTCH1, seven and a half times its
    distance-matched expectation, with six CTCF ChIA-PET read pairs at the same coordinates.</figcaption>
  </figure>

  <h3>Limitations, written down while they are obvious</h3>
  <p>One chromosome, one cell line, one species. Bulk Hi-C averaged over millions of cells, so a TAD
  boundary here is a population statistic rather than a structure present in any individual cell.
  A single experiment with no replicate, so seed variance in Phase 5 captures model variance only and
  says nothing about measurement variance. Balancing, filtering and mapping were done by 4D Nucleome
  upstream of this project, along with the three tracks used for validation, so their processing choices
  and any artifacts are inherited. A fifth of the chromosome is structurally blank, dominated by a single
  17.25 Mb centromeric gap, which makes usable chr9 two arms of roughly 43 and 78 Mb rather than one
  continuous sequence.</p>

  <h3>Three assembly traps, caught</h3>
  <p>The first validation figure circled a bright pixel 1,425 kb off the diagonal and called it a loop.
  It was noise in a sparse corner; the detector had no separation ceiling and no local-maximum test.</p>
  <p>A region was selected on the belief that it contained ABL1 at chr9:133.7 Mb. That is an hg19
  coordinate. In GRCh38 ABL1 sits at 130,713,043&ndash;130,887,675, outside the region.</p>
  <p>The standard published loop list for this dataset, GSE63525 via GEO, is hg19. It was caught because
  its largest chr9 anchor is 140,570,000, past the end of GRCh38 chr9 at 138,394,717 &mdash; a file cannot
  describe a genome it runs off the end of. Nothing would have errored; every loop would have been drawn
  in the wrong place.</p>
</section>

<section class="part" id="apD">
  <div class="part-head"><span class="part-num">D</span><h2>Decisions, open questions, and the repository map</h2></div>

  <h3>Decisions of record</h3>
  <div class="tw"><table class="wide">
    <thead><tr><th>&nbsp;</th><th>Decision</th><th>Choice</th><th>Reasoning</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>Mechanism</td><td>structural bias in the recurrence</td><td>unreachable by contrastive alignment on output embeddings; contains hard state resets as its limit; cheap to falsify</td></tr>
      <tr><td>2</td><td>Reverse-complement handling</td><td>Caduceus-PH, data augmentation</td><td>equivariance is orthogonal to the hypothesis; coupling them makes a null result uninterpretable</td></tr>
      <tr><td>3</td><td>Hi-C resolution</td><td>5 kb, with 1 kb as an ablation</td><td>comparability with CHROME; F4 argues for 1 kb but it buys noisier contacts, and both levels sit in the same file</td></tr>
      <tr><td>4</td><td>Cross-species scope</td><td>out for this paper</td><td>Evo2HiC uses 177 species and HiCFoundation 316; entering that lane splits compute against better-resourced groups</td></tr>
    </tbody>
  </table></div>

  <h3>What is still open</h3>
  <p>CHROME needs a line-by-line human read; the summary in Appendix A came from automated extraction of
  the PMC full text. Orca's Results section is paywalled at <em>Nature Genetics</em> with no PubMed Central
  deposit, so the claim that its representations were never evaluated on a non-folding task rests on the
  abstract, all ten extended-data captions, and the section headings rather than the body. Lee's Evo 2
  probe is a preprint and its peer-review status should be rechecked before it carries weight in a
  methods section. Evo2HiC should be rechecked for a revision adding variant evaluation before Phase 4
  commits GPU time, since that would close the one contribution nothing currently contests.</p>

  <h3>Repository map</h3>
  <div class="tw"><table>
    <thead><tr><th>Path</th><th>Contents</th></tr></thead>
    <tbody>
      <tr><td>docs/related_work.md</td><td>eleven papers, the gap table, two follow-up appendices, verification status per claim</td></tr>
      <tr><td>docs/architecture_spec.md</td><td>the mechanism, parameter accounting, controls, eight failure modes, decisions of record</td></tr>
      <tr><td>docs/data_card.md</td><td>sources, measured properties, the feature definition, limitations, the Phase 1 gate</td></tr>
      <tr><td>docs/primer.html</td><td>this document, web edition</td></tr>
      <tr><td>scripts/phase1_acquire.py</td><td>pilot acquisition; resolves 4DN accessions, reads the 27 GB cooler remotely</td></tr>
      <tr><td>scripts/phase1_features.py</td><td>builds the eight structural features and validates against 4DN's tracks</td></tr>
      <tr><td>scripts/phase1_validate_visual.py</td><td>the validation figure</td></tr>
      <tr><td>scripts/param_accounting.py</td><td>instantiates both models and counts parameter tensors</td></tr>
      <tr><td>scripts/f4_memory_horizon.py</td><td>computes memory horizons against Hi-C bin size</td></tr>
      <tr><td>scripts/build_primer_pdf.py</td><td>assembles and renders this PDF</td></tr>
      <tr><td>data/pilot_manifest.json</td><td>accessions, URLs, sizes and checksums; the data itself is not committed</td></tr>
    </tbody>
  </table></div>

  <h3>What happens next</h3>
  <p>Phase 3 trains the sequence-only baseline on two L40S GPUs and logs every hyperparameter, seed and
  metric before any structural model exists, so that no flattering comparison can be chosen after the fact.
  That run also has to report the trained memory horizon, which decides whether a model this size can
  represent TAD-scale dependencies at all. If it cannot, the mechanism is re-scoped to sub-TAD structure
  or the model is made larger, and that decision happens before Phase 4 rather than after it.</p>

  <hr class="soft"/>
  <p class="quiet" style="font-size:8.6pt">Compiled 7 August 2026 by rendering docs/primer.html together with these
  appendices. Every measured number traces to a named script in the repository. No model has been trained
  and this document reports no experimental results, because there are none.</p>
</section>
"""


def build_html() -> Path:
    # take the artifact primer's body verbatim, minus its <title> and web <style>
    src = (DOCS / "primer.html").read_text(encoding="utf-8")
    body = src.split("</style>", 1)[1]

    # drop the web masthead
    body = re.sub(r'<header class="masthead">.*?</header>', "", body, flags=re.S)
    # drop the web table of contents (contains no nested divs)
    body = re.sub(r'<div class="col toc">.*?</div>', "", body, flags=re.S, count=1)
    # insert print front matter after the opening .page div
    body = body.replace('<div class="page">', '<div class="page">' + COVER, 1)
    # append appendices before the final closing div
    head, sep, tail = body.rpartition("</div>")
    body = head + APPENDICES + sep + tail

    doc = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>3D-Chromatin-Aware Genomic FM — primer</title>"
        f"<style>{PRINT_CSS}</style></head><body>{body}</body></html>"
    )
    out = DOCS / "primer_print.html"
    out.write_text(doc, encoding="utf-8")
    print(f"  print html: {out.relative_to(REPO)}  ({len(doc):,} chars)")
    return out


def render(html: Path) -> Path:
    browser = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)
    if not browser:
        raise SystemExit("no Chrome or Edge found")
    print(f"  renderer: {Path(browser).name}")
    raw = DOCS / "_primer_raw.pdf"
    cmd = [
        browser, "--headless", "--disable-gpu", "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={raw}",
        html.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    print(f"  rendered: {raw.stat().st_size/1e6:.2f} MB")
    return raw


def stamp(raw: Path) -> Path:
    """Chromium ignores CSS @page margin boxes, so page numbers go on afterwards."""
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    reader = PdfReader(str(raw))
    n = len(reader.pages)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        if i > 0:                      # no furniture on the cover
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            c.setFont("Helvetica", 7.4)
            c.setFillGray(0.48)
            c.drawString(48, 30, "3D-Chromatin-Aware Genomic FM \u2014 primer")
            c.drawRightString(A4[0] - 48, 30, f"{i + 1} / {n}")
            c.setStrokeGray(0.85)
            c.setLineWidth(0.5)
            c.line(48, 40, A4[0] - 48, 40)
            c.save()
            buf.seek(0)
            page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)

    writer.add_metadata({
        "/Title": "Teaching a DNA model that the genome is folded",
        "/Subject": "3D-Chromatin-Aware Genomic Foundation Model \u2014 project primer",
        "/Keywords": ("genomic foundation models; 3D genome organisation; Hi-C; "
                      "topologically associating domains; state-space models; Mamba; "
                      "self-supervised pretraining; CTCF; variant effect prediction; "
                      "representation transfer"),
        "/Creator": "scripts/build_primer_pdf.py",
    })
    with OUT_PDF.open("wb") as fh:
        writer.write(fh)
    print(f"  stamped {n} pages")
    return OUT_PDF


def main() -> None:
    print("building primer PDF")
    html = build_html()
    raw = render(html)
    out = stamp(raw)
    raw.unlink(missing_ok=True)
    print(f"\n{out.relative_to(REPO)}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
