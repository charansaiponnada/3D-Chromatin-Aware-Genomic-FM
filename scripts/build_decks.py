"""Regenerate every review deck by filling in the college template.

    python scripts/build_decks.py

The template's slides are edited IN PLACE and unused ones are deleted. They are
never rebuilt from layouts: the college header, the two logos and the per-slide
title geometry live on the template's own slides, not on the slide master, so
anything rebuilt from a layout silently loses the college identity.

Content lives here rather than in the .pptx files, so one change to the plan
updates all five decks and none can drift out of sync.

Honesty rule: any number that is not yet measured renders as "??". Nothing in a
results table may be typed here by hand -- it comes from a file under results/
or it stays "??".
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = Path.home() / "Downloads" / "Mini Project Presentation Template.pptx"
OUT = REPO / "presentations"
FIG = REPO / "figures"

SLIDE_W = Inches(10)
GREY = RGBColor(0x59, 0x59, 0x59)
ACCENT = RGBColor(0xB8, 0x54, 0x50)
RELNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

TEAM = [
    "P. Charan Sai (238W1A5447)",
    "G. Karthik (248W5A5404)",
    "P. Sanath (238W1A5446)",
]
GUIDE = ("Mrs. K. Divya Kothapali", "Assistant Professor")
COURSE = "23CS7552: B. Tech Mini Project"
TITLE = "ChromGraphFM: An End-to-End 3D Chromatin-Conditioned Genomic Foundation Model"

# Placeholder dates carried over from the original schedule. Correct them here
# once the department publishes the real review calendar.
REVIEWS = {
    "Mini_Project_Review_0":       ("0th Review",      "August 18, 2026"),
    "Mini_Project_Review_1":       ("1st Review",      "September 15, 2026"),
    "Mini_Project_Review_2":       ("2nd Review",      "October 16, 2026"),
    "Mini_Project_End_Sem_Review": ("End Sem. Review", "November 20, 2026"),
}

TBD = "??"  # never replace by hand -- only a file under results/ may fill this


# --------------------------------------------------------------------------- #
# pptx primitives
# --------------------------------------------------------------------------- #

def body_of(slide):
    """The template's content placeholder (idx 1), or None."""
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 1:
            return shape
    return None


def strip_examples(slide):
    """Drop the template's illustrative pictures, captions and sample tables.

    Only non-placeholder shapes go. Title, body and slide-number placeholders
    stay. Never called on the title slide, whose logos and college header are
    non-placeholder shapes.
    """
    for shape in list(slide.shapes):
        if not shape.is_placeholder:
            shape._element.getparent().remove(shape._element)


def drop_shape(shape):
    shape._element.getparent().remove(shape._element)


def set_text(tf, items, base_size=18):
    """items: str, or (level, text), or (level, text, size).

    python-pptx cannot measure text, so nothing autofits. Long slides shrink by
    line count instead -- crude, but it is the difference between a readable
    slide and one whose last bullets fall off the bottom in the review room.
    """
    tf.clear()
    tf.word_wrap = True
    # clear() keeps paragraph 0's own properties, including the template's
    # auto-numbering, which would render the first bullet as "1. 1. Item"
    # while every following bullet gets a plain bullet char.
    p0 = tf.paragraphs[0]._p
    pPr = p0.find(qn("a:pPr"))
    if pPr is not None:
        p0.remove(pPr)
    lines = sum(1 + len(str(i[1] if isinstance(i, tuple) else i)) // 95 for i in items)
    shrink = 0 if lines <= 11 else (1 if lines <= 14 else (2 if lines <= 18 else 3))

    first = True
    for item in items:
        if isinstance(item, str):
            level, text, size = 0, item, base_size - shrink
        elif len(item) == 2:
            level, text = item
            size = base_size - 2 * level - shrink
        else:
            level, text, size = item
            size -= shrink

        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.level = level
        run = para.add_run()
        bold = text.startswith("**")
        run.text = text.strip("*") if bold else text
        run.font.size = Pt(size)
        run.font.bold = bold
        if level >= 2:
            run.font.color.rgb = GREY


def fill(slide, items, base_size=18, notes=None):
    ph = body_of(slide)
    if ph is None:
        raise ValueError(f"no body placeholder on '{slide.shapes.title.text}'")
    strip_examples(slide)
    set_text(ph.text_frame, items, base_size)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def put_picture(slide, png, caption, notes=None):
    """Replace the slide body with a centred figure and its caption."""
    ph = body_of(slide)
    top = ph.top if ph is not None else Inches(1.1)
    avail_h = Inches(7.0) - Emu(top) - Inches(0.45)
    strip_examples(slide)
    if ph is not None:
        drop_shape(ph)

    from PIL import Image
    with Image.open(png) as im:
        w, h = im.size
    scale = min(Inches(9.5) / w, avail_h / h)
    pw, ph_px = int(w * scale), int(h * scale)
    slide.shapes.add_picture(str(png), int((SLIDE_W - pw) / 2), top, pw, ph_px)

    box = slide.shapes.add_textbox(Inches(0.4), top + Emu(ph_px) + Inches(0.06),
                                   Inches(9.2), Inches(0.4))
    box.text_frame.word_wrap = True
    p = box.text_frame.paragraphs[0]
    p.alignment = 2
    run = p.add_run()
    run.text = caption
    run.font.size = Pt(11)
    run.font.italic = True
    run.font.color.rgb = GREY
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def put_table(slide, headers, rows, caption=None, col_widths=None, font=11,
              style_from=None, notes=None):
    """Replace the slide body with a table, keeping the template's table style."""
    ph = body_of(slide)
    left = ph.left if ph is not None else Inches(0.4)
    top = ph.top if ph is not None else Inches(1.1)
    width = ph.width if ph is not None else Inches(9.2)

    # capture the template's own table style before deleting the sample table
    old_pr = None
    for shape in slide.shapes:
        if shape.has_table:
            old_pr = copy.deepcopy(shape.table._tbl.tblPr)
            break
    if old_pr is None and style_from is not None:
        old_pr = copy.deepcopy(style_from)

    strip_examples(slide)
    if ph is not None:
        drop_shape(ph)

    # Caption goes ABOVE the table, matching the template's own "Table No. :"
    # convention. It also has to: python-pptx cannot measure wrapped cell text,
    # so a table always grows past the height requested here, and a caption
    # placed below would sit on top of the last row.
    cap_h = Inches(0.42) if caption else Inches(0)
    if caption:
        box = slide.shapes.add_textbox(left, top, width, cap_h)
        box.text_frame.word_wrap = True
        run = box.text_frame.paragraphs[0].add_run()
        run.text = caption
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.color.rgb = GREY

    nrows = len(rows) + 1
    height = Inches(min(5.0, 0.34 * nrows + 0.2))
    gfx = slide.shapes.add_table(nrows, len(headers), left, top + cap_h, width, height)
    tbl = gfx.table
    if old_pr is not None:
        tbl._tbl.replace(tbl._tbl.tblPr, old_pr)

    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(width * w / total))

    def _cell(r, c, text, bold=False):
        cell = tbl.cell(r, c)
        cell.text = str(text)
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(font)
                run.font.bold = bold
                if str(text) == TBD:
                    run.font.color.rgb = ACCENT
                    run.font.bold = True

    for c, head in enumerate(headers):
        _cell(0, c, head, bold=True)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            _cell(r, c, val)

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def fill_title_slide(slide, review_label, date):
    """Fill the template's title slide without touching its header or logos."""
    slide.shapes.title.text_frame.text = TITLE
    for para in slide.shapes.title.text_frame.paragraphs:
        para.alignment = 2
        for run in para.runs:
            run.font.size = Pt(20)
            run.font.bold = True

    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text
        if text.startswith("Batch Members"):
            set_text(shape.text_frame,
                     [(0, "**Batch Members", 14)] + [(0, m, 13) for m in TEAM], 14)
        elif text.startswith("Under the Guidance"):
            set_text(shape.text_frame,
                     [(0, "**Under the Guidance of", 14), (0, GUIDE[0], 13), (0, GUIDE[1], 12)], 14)
        elif text.startswith("23CS7552"):
            set_text(shape.text_frame,
                     [(0, f"{COURSE} ({review_label})", 14), (0, date, 13),
                      (0, "Batch No: __", 13)], 14)
    return slide


def delete_slides(prs, keep: set[int]):
    """Keep the given template slide indices, in template order; drop the rest."""
    lst = prs.slides._sldIdLst
    for i, sld in reversed(list(enumerate(lst))):
        if i not in keep:
            prs.part.drop_rel(sld.get(RELNS))
            lst.remove(sld)


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #

ABSTRACT = (
    "Three-dimensional chromatin organisation lets distal regulatory elements control gene "
    "expression, yet most genomic sequence models treat the genome as a one-dimensional string. "
    "This project develops ChromGraphFM, a compact 3D chromatin-conditioned genomic representation "
    "model trained from scratch on DNA sequence and Hi-C contact maps. Genomic regions become nodes "
    "in a sparse contact graph, and measured Hi-C contact strengths bias attention among distant DNA "
    "windows throughout self-supervised pretraining, so structure shapes the sequence representation "
    "instead of being appended after it. Pretraining combines masked DNA modelling, distance-matched "
    "contact-aware contrastive learning, and chromatin contact prediction. The model is evaluated "
    "against sequence-only, shuffled-contact, distance-only, random-graph and late-fusion controls "
    "trained at matched parameters, data and compute, using chromosome-level and cell-type held-out "
    "splits for contact-map prediction, chromatin loop detection, enhancer-promoter linking and "
    "Hi-C-free transfer. The central question is whether experimentally measured 3D structure can be "
    "internalised into a reusable DNA representation rather than supplied as an extra input at "
    "prediction time."
)

OUTLINE = ["Research Domain", "SDG Categories", "Aim and Motivation", "Research Questions",
           "Title Justification", "Objectives", "Scope", "Timeline Chart", "Introduction",
           "Study on Existing Technologies", "Gap Analysis", "SDLC Model", "UML Diagrams",
           "Functional and Non-Functional Requirements", "Data Collection", "Data Preparation",
           "Proposed System Architecture", "Experimental Setup and Results",
           "Project Outcomes", "Conclusion and Future Work"]

TIMELINE = [
    ("Phase 0", "Wk 1",     "Scaffold, config system, split validator"),
    ("Phase 1", "Wk 2-5",   "hg38 windows, streamed Hi-C, 5 kb bins, QC report"),
    ("Phase 2", "Wk 6-8",   "Sparse contact graphs, held-out edges, control corruptions"),
    ("Phase 3", "Wk 9-11",  "Bi-Mamba encoder, B0 baseline, evaluation harness"),
    ("Phase 4", "Wk 12-15", "Contact-biased attention, three pretraining losses"),
    ("Phase 5", "Wk 16-18", "Matched B0-B5 sweep, held-out chromosome and cell line"),
    ("Phase 6", "Wk 19-20", "CTCF interpretation, report, poster, public repository"),
]

RELATED = [
    ("Akita", "Nature Methods, 2020", "CNN predicting 3D folding from DNA sequence",
     "Task-specific predictor; no reusable pretrained representation"),
    ("C.Origami", "Nature Biotech., 2023", "Cell-type-specific 3D organisation from sequence, CTCF, ATAC",
     "Hi-C is the target, not a signal shaping an encoder"),
    ("EPCOT", "Nucleic Acids Res., 2023", "Pretrain/fine-tune for epigenome, contacts, transcription",
     "Pretraining centres on epigenomic features"),
    ("Caduceus", "arXiv:2403.03234, 2024", "Bi-directional RC-equivariant long-range DNA model",
     "Strong sequence-only baseline; uses no Hi-C"),
    ("HiCFoundation", "Nature Methods, 2026", "Foundation model pretrained on Hi-C contact maps",
     "Represents contact maps, not DNA sequence"),
    ("MIX-HIC", "NeurIPS 2025", "Multimodal Hi-C and epigenomic pretraining",
     "No raw DNA input; names sequence integration as future work"),
    ("Evo2HiC", "bioRxiv, 2025", "Hi-C-guided contrastive distillation of Evo 2 into a CNN",
     "Closest precedent; frozen backbone, contacts never enter contextualisation"),
]

REFERENCES = [
    "[1] Fudenberg, G., Kelley, D. R., & Pollard, K. S. (2020). Predicting 3D genome folding from DNA "
    "sequence with Akita. Nature Methods, 17(11), 1111-1117.",
    "[2] Tan, J., et al. (2023). Cell-type-specific prediction of 3D chromatin organization enables "
    "high-throughput in silico genetic screening. Nature Biotechnology, 41(8), 1140-1150.",
    "[3] Zhang, Z., et al. (2023). A generalizable framework to comprehensively predict epigenome, "
    "chromatin organization, and transcriptome. Nucleic Acids Research, 51(12), 5931-5947.",
    "[4] Schiff, Y., et al. (2024). Caduceus: Bi-directional equivariant long-range DNA sequence "
    "modeling. arXiv:2403.03234.",
    "[5] A generalizable Hi-C foundation model for chromatin architecture, single-cell and multi-omics "
    "analysis across species (HiCFoundation). Nature Methods (2026).",
    "[6] MIX-HIC: Multimodal 3D Genome Pre-training. arXiv:2504.09060 (2025).",
    "[7] Evo2HiC: a multimodal foundation model for integrative analysis of genome sequence and "
    "architecture. bioRxiv 2025.11.18.689171 (2025).",
    "[8] Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. "
    "arXiv:2312.00752.",
    "[9] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 2017.",
    "[10] Rao, S. S. P., et al. (2014). A 3D map of the human genome at kilobase resolution reveals "
    "principles of chromatin looping. Cell, 159(7), 1665-1680.",
    "[11] Abdennur, N., & Mirny, L. A. (2020). Cooler: scalable storage for Hi-C data and other "
    "genomically labeled arrays. Bioinformatics, 36(1), 311-316.",
]


# --------------------------------------------------------------------------- #
# one filler per template slide index
# --------------------------------------------------------------------------- #

def s01_review_details(s, ctx):
    put_table(s, ["Item", "Detail"],
              [["Project title", "ChromGraphFM"],
               ["Course", COURSE],
               ["Review", ctx["label"]],
               ["Date", ctx["date"]],
               ["Guide", f"{GUIDE[0]}, {GUIDE[1]}"],
               ["Team", "; ".join(TEAM)],
               ["Repository", "3D-Chromatin-Aware-Genomic-FM (git)"],
               ["Status", ctx["status"]]],
              col_widths=[1, 3], font=13, style_from=ctx.get("tbl_style"))


def s02_abstract(s, ctx):
    fill(s, [(0, ABSTRACT, 15)])


def s03_outline(s, ctx):
    fill(s, [(0, f"{i}. {t}", 15) for i, t in enumerate(OUTLINE, 1)], 15)


def s04_domain(s, ctx):
    fill(s, [
        "**Primary domain: Computational Genomics / Bioinformatics",
        (1, "Three-dimensional genome organisation and gene regulation"),
        (1, "Chromosome conformation capture data (Hi-C, Micro-C)"),
        "**Secondary domain: Machine Learning / Representation Learning",
        (1, "Self-supervised pretraining and foundation models for biological sequences"),
        (1, "Attention and graph neural networks with a structural inductive bias"),
        "**Where the two meet",
        (2, "The genome is read as a 1D string but folds in 3D. Regulatory elements act on genes "
            "megabases away in sequence yet nanometres away in space. This project asks whether "
            "measured folding can be built into how a model reads the sequence."),
    ])


def s05_sdg(s, ctx):
    fill(s, [
        "**SDG 3 - Good Health and Well-being",
        (1, "Most disease-associated variants from genome-wide association studies fall in "
            "non-coding regions whose target genes are unknown"),
        (1, "Better enhancer-gene linking is a direct route to interpreting those variants"),
        "**SDG 9 - Industry, Innovation and Infrastructure",
        (1, "A compact, openly released pretrained encoder lowers the compute barrier for labs "
            "that cannot train billion-parameter genomic models"),
        (1, "The Hi-C-free setting matters here: most labs have sequence but no Hi-C"),
        "**SDG 4 - Quality Education",
        (1, "Fully reproducible pipeline, tracked data manifest and public repository"),
    ])


def s06_aim(s, ctx):
    fill(s, [
        "**Aim",
        (1, "Develop and evaluate a genomic representation model trained from scratch in which "
            "Hi-C contact information directly biases how a DNA sequence encoder exchanges "
            "information across distant genomic regions during self-supervised pretraining."),
        "**Motivation",
        (1, "Sequence-only models must infer long-range regulatory relationships from sequence "
            "alone, though those relationships have already been measured experimentally"),
        (1, "Existing 3D-genome models either predict Hi-C from sequence, represent Hi-C maps "
            "themselves, or align a frozen encoder to Hi-C embeddings after the fact"),
        (1, "None lets the measured contact map decide which distal windows communicate while "
            "the representation is being learned"),
        "**The bet",
        (2, "If real contacts carry information a sequence model cannot recover alone, "
            "conditioning on them should beat a matched model given shuffled contacts. If it "
            "does not, that null is itself a result about the limits of the signal."),
    ])


def s07_questions(s, ctx):
    fill(s, [
        "**RQ1 - Does structural conditioning help at all?",
        (1, "Does biasing attention with measured Hi-C improve held-out chromosome contact "
            "prediction over a matched sequence-only encoder?"),
        "**RQ2 - Is the gain from real biology or from any extra signal?",
        (1, "Does it beat matched controls given shuffled Hi-C, distance-only edges and a "
            "random contact graph?"),
        "**RQ3 - Is conditioning better than fusion?",
        (1, "Does contact-biased attention inside the encoder beat late fusion applied after "
            "the sequence has been encoded?"),
        "**RQ4 - Is the structure retained in the sequence representation?",
        (1, "Does any improvement survive removing the contact graph at inference time?"),
        "**RQ5 - Does it transfer?",
        (1, "Does the pretrained encoder beat a from-scratch model on loop detection and "
            "enhancer-promoter linking, including on a held-out cell line?"),
    ])


def s08_title_just(s, ctx):
    fill(s, [
        "**\"ChromGraphFM\"",
        (1, "Chrom - the object of study is chromatin, not sequence alone"),
        (1, "Graph - a contact is a relation between two loci, not a pixel in an image"),
        (1, "FM - the artefact is a pretrained encoder evaluated across several tasks"),
        "**\"End-to-End\"",
        (1, "Distinguishes this work from Evo2HiC, which distils a frozen Evo 2 backbone; here "
            "the sequence encoder trains from scratch under the structural objectives"),
        "**\"3D Chromatin-Conditioned\"",
        (1, "Conditioned, not fused: the contact map alters attention inside the encoder at "
            "every block rather than being concatenated to its output"),
        "**A deliberate restraint on the word \"Foundation\"",
        (2, "At 20-40M parameters this is a compact pretrained representation model, and the "
            "report will call it that unless multi-task transfer is demonstrated."),
    ])


def s09_objectives(s, ctx):
    fill(s, [
        "**O1 - Reproducible data foundation",
        (1, "Chromosome-split hg38 and Hi-C dataset at 5 kb, rebuildable from a tracked manifest"),
        "**O2 - Sparse contact graph construction",
        (1, "640 kb regions as 128-node graphs, with target edges provably held out from the "
            "conditioning edges"),
        "**O3 - Contact-biased sequence encoder",
        (1, "Bi-directional Mamba local encoder, contact-biased attention, structure dropout"),
        "**O4 - Structure-aware pretraining",
        (1, "Masked DNA modelling, distance-matched contrastive learning, contact prediction"),
        "**O5 - Controlled evaluation",
        (1, "B0-B5 controls at matched parameters, data and steps; held-out chromosomes, "
            "held-out cell line, Hi-C-free inference"),
        "**O6 - Biological interpretation",
        (1, "Test whether high-attention contact anchors are enriched for CTCF motifs and "
            "overlap known loop anchors"),
    ])


def s10_scope(s, ctx):
    fill(s, [
        "**In scope",
        (1, "Human genome hg38; bulk Hi-C from 4DN and ENCODE"),
        (1, "5 kb contact resolution; 640 kb context (128 nodes x 5 kb)"),
        (1, "2-3 cell lines, one fully held out; 20-40M parameters; DNA and Hi-C only"),
        (1, "Controlled comparison against our own matched baselines B0-B5"),
        "**Out of scope",
        (1, "Billion-parameter pretraining; million-base contexts"),
        (1, "ATAC-seq, DNase-seq or CTCF tracks as model inputs (stretch goal only)"),
        (1, "Single-cell Hi-C, cross-species transfer, variant-effect prediction (stretch)"),
        "**Claim discipline",
        (2, "No claim of superiority over a published model unless task, data, split, metric, "
            "resolution and input setting all match. Published numbers are reported as context, "
            "never as a head-to-head result."),
    ])


def s11_timeline(s, ctx):
    rows = []
    for i, (phase, weeks, deliv) in enumerate(TIMELINE):
        st = "Complete" if i < ctx["phase"] else ("In progress" if i == ctx["phase"] else "Planned")
        rows.append([phase, weeks, deliv, st])
    put_table(s, ["Phase", "Weeks", "Deliverable", "Status"], rows,
              caption="Table 1: Timeline chart for the mini project. Phases 0-2 need no GPU and "
                      "run on a laptop; Phases 3-6 require the L40S cluster.",
              col_widths=[1, 1, 4.6, 1.2], font=12, style_from=ctx.get("tbl_style"))


def s12_intro(s, ctx):
    fill(s, [
        "**The biology",
        (1, "Human DNA is about 2 metres long and folds into a nucleus about 6 micrometres "
            "across. The folding is not random: it forms loops, domains and compartments."),
        (1, "An enhancer can regulate a gene a megabase away because folding brings them into "
            "physical proximity. Sequence adjacency and functional adjacency differ."),
        "**The measurement",
        (1, "Hi-C sequences pairs of genomic fragments that were physically close, producing a "
            "contact matrix over genomic bins - the experimental readout of that folding."),
        "**The modelling gap",
        (1, "Genomic language models read the genome as text. Attention over sequence position "
            "cannot know two distant windows are neighbours in space unless it is told."),
        "**This project",
        (2, "Tell it, during pretraining, and measure whether that changes what the encoder "
            "learns - with controls strong enough that a positive result means something."),
    ])


def s13_related_a(s, ctx):
    fill(s, [
        "**Title: Evo2HiC - a multimodal foundation model for integrative analysis of genome "
        "sequence and architecture [7]",
        (1, "Venue: bioRxiv preprint, November 2025"),
        (1, "Method: contrastive distillation of a frozen 7B-parameter Evo 2 into a 3.6M "
            "parameter CNN sequence encoder, aligned against Hi-C patch embeddings"),
        (1, "Relevance: the closest precedent, and the reason this project cannot claim to be "
            "the first to use Hi-C in genomic representation learning"),
        (1, "Limitation we build on: the backbone stays frozen, and contacts never enter the "
            "encoder's own contextualisation"),
        "**Title: MIX-HIC - Multimodal 3D Genome Pre-training [6]",
        (1, "Venue: NeurIPS 2025 (arXiv:2504.09060)"),
        (1, "Method: joint pretraining over Hi-C contact maps and epigenomic tracks, more than "
            "1.2 million paired samples, 5 kb resolution, chromosome-level splits"),
        (1, "Limitation we build on: raw DNA sequence is not an input; the authors name "
            "sequence integration as future work"),
    ], 16)


def s14_related_b(s, ctx):
    fill(s, [
        "**Three families of prior work",
        (1, "Sequence-to-structure predictors - Akita, C.Origami, EPCOT. Sequence in, contact "
            "map out. Hi-C is the prediction target."),
        (1, "Hi-C representation models - HiCFoundation, MIX-HIC. Hi-C in, Hi-C representation "
            "out. DNA sequence absent or secondary."),
        (1, "Sequence foundation models - Caduceus, HyenaDNA, Evo 2. Long-range sequence "
            "modelling with no structural signal."),
        "**What the field has already settled",
        (1, "Hi-C can be predicted from sequence, and Hi-C maps can be pretrained on"),
        (1, "Hi-C can shape a sequence encoder by distillation (Evo2HiC)"),
        "**What this means for our claim",
        (2, "\"First to use Hi-C in a genomic foundation model\" is false and must not be "
            "claimed. The defensible contribution is the mechanism and the controls that "
            "isolate it."),
    ], 16)


def s15_related_table(s, ctx):
    put_table(s, ["Model", "Venue", "Core idea", "What it does not do"],
              [list(r) for r in RELATED],
              caption="Table 2: Summary of existing implementations and the gap each leaves open.",
              col_widths=[1.1, 1.4, 2.8, 3.1], font=10, style_from=ctx.get("tbl_style"))


def s16_gap(s, ctx):
    fill(s, [
        "**Established in the literature",
        (1, "Foundation models pretrained on Hi-C maps exist (HiCFoundation)"),
        (1, "Multimodal Hi-C and epigenomic pretraining exists (MIX-HIC)"),
        (1, "Hi-C-guided distillation into a DNA encoder exists (Evo2HiC)"),
        "**Still open",
        (1, "No published model trains a DNA encoder end to end while measured contact graphs "
            "control which distal windows exchange information at every layer"),
        (1, "Post-hoc alignment and late fusion have not been separated from in-encoder "
            "conditioning under matched parameters, data and compute"),
        (1, "Whether structural knowledge survives removal of Hi-C at inference is asserted "
            "more often than it is tested against a shuffled-contact control"),
        "**Gap statement",
        (2, "It remains untested whether contact graphs conditioning sequence contextualisation "
            "throughout pretraining yield a more transferable DNA representation than the "
            "alternatives, measured against distance-matched and shuffled-contact controls."),
    ], 16)


def s17_sdlc(s, ctx):
    put_picture(s, FIG / "sdlc.png",
                "Figure 1: Incremental model with an explicit decision gate per phase.",
                notes="Incremental, not waterfall: the deliverable is a comparison, so every "
                      "phase must yield something measurable against a control. A waterfall "
                      "plan reaches its first measurement in the final weeks, leaving no time "
                      "to respond to a null.\n\n"
                      "Incremental, not agile sprints: the scientific claim is fixed in "
                      "advance; the backlog does not change, only the evidence accumulates.\n\n"
                      "Phase 4 carries a stop gate. If the model given real Hi-C does not beat "
                      "the same model given shuffled Hi-C, the central claim fails and we "
                      "report that rather than tuning until the number moves.")


def s18_usecase(s, ctx):
    put_picture(s, FIG / "usecase.png",
                "Figure 2: Actors are the external roles the system serves, plus the two "
                "external systems it depends on.",
                notes="The Downstream ML Practitioner is the actor that exercises the central "
                      "claim: reusing the pretrained encoder when no Hi-C is available at "
                      "inference.")


def s19_sequence(s, ctx):
    put_picture(s, FIG / "sequence.png",
                "Figure 3: One pretraining run, from manifest resolution to tracked metrics.",
                notes="Note message 8: run_config.yaml is written BEFORE the first training "
                      "step, not after the last, so a run that dies mid-way still leaves a "
                      "complete record of what it was.")


def s20_activity(s, ctx):
    put_picture(s, FIG / "activity.png",
                "Figure 4: Experiment workflow, including the QC gate and the shuffled-Hi-C "
                "decision point.")


def s21_requirements(s, ctx):
    fill(s, [
        "**Functional requirements",
        (1, "FR1: Rebuild the full dataset on any machine from data/manifest.json alone"),
        (1, "FR2: Construct sparse contact graphs with configurable top-k and local radius"),
        (1, "FR3: Hold out a configurable fraction of edges as prediction targets, excluded "
            "from the conditioning graph"),
        (1, "FR4: Train any of the seven arms (B0-B5, full) from one configuration field"),
        (1, "FR5: Write run_config.yaml and metrics.json for every run"),
        (1, "FR6: Run inference with the contact graph removed"),
        "**Non-functional requirements",
        (1, "NFR1 Reproducibility: seed, commit hash, config and accessions recorded per run"),
        (1, "NFR2 Portability: no hardcoded paths; laptop (CPU) and cluster (CUDA) from one tree"),
        (1, "NFR3 Memory: no dense chromosome-scale contact matrix is ever materialised"),
        (1, "NFR4 Integrity: the config loader rejects chromosome leakage and unknown keys"),
        (1, "NFR5 Auditability: no reported number without a tracked file behind it"),
    ], 16)


def s22_data(s, ctx):
    put_table(s, ["Data", "Source", "Format / resolution", "Use"],
              [["Reference genome", "UCSC hg38", "FASTA + .fai", "5 kb DNA window per node"],
               ["Bulk Hi-C", "4DN Data Portal", ".mcool, 5 kb, ICE", "Contact graph, pretraining"],
               ["Hi-C, 2nd cell line", "ENCODE", ".mcool, 5 kb", "Held-out cell-type test"],
               ["Chromatin loops", "CTCF ChIA-PET / HiChIP", "BEDPE", "Loop detection labels"],
               ["Gene expression", "ENCODE CAGE / RNA-seq", "bigWig / TSV", "Downstream transfer"],
               ["CTCF motif", "JASPAR MA0139", "PWM", "Interpretation only"]],
              caption="Table 3: Accessions, URLs and md5 checksums live in data/manifest.json, "
                      "the only data artefact carried in git. Splits are chromosome-level: "
                      "train on autosomes, validate on chr10 and chr11, test on chr3, chr13 "
                      "and chr17.",
              col_widths=[1.5, 1.7, 1.8, 2.4], font=11, style_from=ctx.get("tbl_style"),
              notes="Never split genomic windows randomly. Hi-C is autocorrelated over "
                    "megabases, so a random split puts near-identical regions on both sides of "
                    "the boundary and inflates every metric.\n\n"
                    "Two harder evaluations sit on top: a fully held-out cell line, and "
                    "Hi-C-free inference where the contact graph is removed after pretraining.")


def s23_prep(s, ctx):
    fill(s, [
        "**Step 1 - Resolve and verify",
        (1, "Read accessions from the manifest; verify md5 after transfer"),
        "**Step 2 - Stream, never densify",
        (1, "Read .mcool pixels chromosome by chromosome with cooler. A dense chr1 matrix at "
            "5 kb is about 49,800 x 49,800 floats, roughly 10 GB, so only sparse pixels are "
            "ever held in memory."),
        "**Step 3 - Align coordinates",
        (1, "One node = one 5 kb Hi-C bin = one 5 kb DNA window. Keeping these identical "
            "removes a pooling layer and an entire class of off-by-one coordinate bugs."),
        "**Step 4 - Filter",
        (1, "Drop nodes whose DNA is more than 50% N; drop bins with zero coverage"),
        "**Step 5 - Shard and record",
        (1, "Write 128-node samples plus a QC report: coordinate spot checks, the P(s) contact "
            "decay curve, split disjointness and N-content distribution"),
        "**Decision gate",
        (2, "No training starts until DNA window and Hi-C bin coordinates are manually verified "
            "to agree at several independent loci."),
    ], 16)


def s24_sysarch(s, ctx):
    put_picture(s, FIG / "system_architecture.png",
                "Figure 5: End-to-end system, from public data to tracked metrics.")


def s25_modules(s, ctx):
    fill(s, [
        "**Module 1 - Data (chromgraph/datasets)",
        (1, "Manifest resolution, Hi-C streaming and binning, DNA window extraction, QC report"),
        "**Module 2 - Graph (chromgraph/preprocessing)",
        (1, "Top-k distal and local edge selection, edge features, target-edge hold-out"),
        (1, "The six control corruptions: random graph, distance-only edges, shuffled contacts"),
        "**Module 3 - Model (chromgraph/models)",
        (1, "Bi-Mamba local DNA encoder, contact-biased multi-head attention, learned edge "
            "bias, structure dropout, three pretraining heads"),
        "**Module 4 - Training (chromgraph/training)",
        (1, "Arm selection, mixed precision, checkpointing, run configuration capture"),
        "**Module 5 - Evaluation (chromgraph/evaluation)",
        (1, "Distance-stratified contact metrics, loop AUPRC, enhancer-promoter ranking, "
            "Hi-C-free probe, CTCF motif enrichment"),
    ], 16)


def s26_model(s, ctx):
    put_picture(s, FIG / "chromgraphfm_architecture_wide.png",
                "Figure 6: ChromGraphFM. The Hi-C-derived edge bias enters the attention score, "
                "so the contact map decides which distal windows communicate.",
                notes="Contact-biased attention:\n"
                      "  a_ij = softmax_j ( q_i . k_j / sqrt(d) + b_HiC(c_ij) + b_dist(d_ij) "
                      "+ b_scale(r) )\n"
                      "c_ij is normalised contact strength, d_ij linear genomic distance, r the "
                      "contact resolution. The bias terms are learned. Attention runs over graph "
                      "edges only, which is what makes 640 kb of context fit on one 48 GB GPU.\n\n"
                      "Objective: L = 1.0*L_DNA + 0.5*L_contrast + 1.0*L_contact\n\n"
                      "Two design choices make the claim testable. Structure dropout (p = 0.3) "
                      "drops the whole graph during pretraining, so Hi-C-free inference is "
                      "in-distribution rather than an untested hope. Held-out target edges keep "
                      "the contact head from copying its own input.")


def s27_algorithms(s, ctx):
    fill(s, [
        "**Algorithm 1: Sparse contact graph construction",
        (1, "Input: contact pixels C, bin size b, k, local radius m, hold-out fraction f"),
        (1, "Output: conditioning edges E_in, target edges E_tgt"),
        (2, "1. Normalise C by the expected contact at each genomic separation"),
        (2, "2. For each node i, keep the k strongest j with |i - j| >= min_separation"),
        (2, "3. Add local edges (i, j) for all 0 < |i - j| <= m"),
        (2, "4. Sample a fraction f of edges into E_tgt using control_seed"),
        (2, "5. E_in = E \\ E_tgt          # assert E_in and E_tgt are disjoint"),
        (2, "6. Attach features (c_ij, d_ij, r, cell_type) to every edge"),
        "**Algorithm 2: One pretraining step",
        (2, "1. h = BiMambaEncoder(dna)                # one embedding per 5 kb node"),
        (2, "2. if uniform() < p_structure_dropout: E_in = local edges only"),
        (2, "3. for layer in 1..N: h = Block(h, E_in)  # contact-biased attention"),
        (2, "4. L = L_DNA(h) + 0.5*L_contrast(h) + L_contact(h, E_tgt)"),
        (2, "5. backward, clip gradients to 1.0, optimiser step"),
    ], 15)


def s28_setup(s, ctx):
    fill(s, [
        "**Hardware",
        (1, "Development (Phases 0-2): laptop, Intel i5-12450H, 16 GB RAM, no CUDA"),
        (1, "Training (Phases 3-6): NVIDIA L40S, 48 GB VRAM, university cluster"),
        (2, "Concurrent GPU count, wall-time limit and scratch quota still to be confirmed with "
            "the cluster administrator; the plan scales from 1 to 4 GPUs."),
        "**Software",
        (1, "PyTorch with bfloat16 mixed precision and gradient checkpointing"),
        (1, "cooler and pyfaidx for Hi-C and sequence access; NumPy, SciPy, scikit-learn"),
        (1, "YAML configuration, seed and commit hash recorded per run"),
        "**Experimental discipline",
        (1, "Every arm shares parameter count, dataset, optimiser, schedule and step budget; "
            "only the structural signal differs"),
        (1, "Three training seeds per arm; controls corrupted with a separate fixed seed so "
            "seed variance and shuffle variance are never confounded"),
        (1, "Hyperparameters tuned on chr10 and chr11 only; test chromosomes touched once"),
    ], 16)


def s29_results(s, ctx):
    put_table(s, ["ID", "Arm", "Contact r (long-range)", "Loop AUPRC", "Hi-C-free probe"],
              [["B0", "DNA-only", TBD, TBD, TBD],
               ["B1", "Random contact graph", TBD, TBD, TBD],
               ["B2", "Distance-only edges", TBD, TBD, TBD],
               ["B3", "Shuffled Hi-C", TBD, TBD, TBD],
               ["B4", "Late fusion", TBD, TBD, TBD],
               ["B5", "Contrastive alignment only", TBD, TBD, TBD],
               ["Ours", "Contact-biased attention", TBD, TBD, TBD]],
              caption="Table 4: Main comparison on held-out chromosomes (chr3, chr13, chr17), "
                      "mean over 3 seeds. Entries remain ?? until results/final_comparison.csv "
                      "exists. The decisive row is Ours against B3.",
              col_widths=[0.6, 2.4, 2.1, 1.4, 1.6], font=12, style_from=ctx.get("tbl_style"))


def s30_results_reading(s, ctx):
    fill(s, [
        "**Distance-stratified, never a single correlation",
        (1, "Hi-C is dominated by short-range contacts. One overall Pearson r can look strong "
            "while long-range prediction is at chance."),
        (1, "Every contact metric is reported separately for short, medium and long separations"),
        "**Distance-matched negatives everywhere",
        (1, "Loop and enhancer-promoter negatives are matched on genomic distance, or the model "
            "is rewarded for learning only that nearby loci contact more often"),
        "**What would falsify the hypothesis",
        (1, "Ours failing to exceed B3 (shuffled Hi-C) on long-range contact prediction"),
        (1, "Ours matching B2 (distance-only), which would mean the model learned separation "
            "rather than structure"),
        (2, "Either outcome is reported as measured. A null on a well-controlled experiment is "
            "a result about the signal, not a failure of the project."),
    ], 16)


def s33_comparison(s, ctx):
    put_table(s, ["Model", "Setting", "Reported metric", "Ours (matched setting)"],
              [["Akita", "Sequence to contact map", "Published value", TBD],
               ["C.Origami", "Sequence + CTCF + ATAC", "Published value", TBD],
               ["EPCOT", "Multi-task epigenome + contacts", "Published value", TBD],
               ["Evo2HiC", "Hi-C-guided distillation", "Published value", TBD],
               ["Caduceus", "Sequence-only backbone", "Published value", TBD]],
              caption="Table 5: Context, not a ranking. A cell is filled only where our task, "
                      "data, split, metric, resolution and input setting match the published "
                      "one. Where they do not match, the row stays descriptive.",
              col_widths=[1.2, 2.4, 2.0, 2.0], font=12, style_from=ctx.get("tbl_style"))


def s34_poc(s, ctx):
    fill(s, [
        "**Proof of concept delivered so far",
        (1, "A configuration system that refuses a chromosome leak between splits, an unknown "
            "configuration key, or a graph whose local and distal edge classes overlap"),
        (1, "A reproducible build: every figure regenerates from its draw.io source and every "
            "deck regenerates from one script"),
        (1, f"Pilot dataset build on two small chromosomes: {TBD}"),
        (1, f"End-to-end smoke run, forward and backward pass: {TBD}"),
        "**How to verify it",
        (2, "git clone, pip install -e ., python tests/test_config.py, then "
            "python scripts/build_data.py --pilot. No manual step and no downloaded artefact "
            "outside the manifest."),
    ], 16)


def s35_outcomes(s, ctx):
    fill(s, [
        "**Technical outcomes",
        (1, "An open, reproducible 5 kb hg38 and Hi-C dataset builder driven by a tracked manifest"),
        (1, "A contact-biased sequence encoder with structure dropout, released with weights"),
        (1, "Six matched control arms, usable by others as a benchmark harness"),
        "**Scientific outcome",
        (1, "A controlled answer, in either direction, to whether measured 3D structure improves "
            "a DNA representation beyond what distance and auxiliary signal already provide"),
        "**EPICS / dissemination",
        (1, f"Title: {TBD}"),
        (1, f"Type: Journal / Conference - {TBD}"),
        (1, f"Venue: {TBD}"),
        (2, "Under consideration: a bioinformatics workshop track, or a bioRxiv preprint "
            "released together with the repository."),
        "**Artefacts",
        (1, "Public repository, reproducibility appendix, conference-style poster, report"),
    ], 16)


def s36_conclusion(s, ctx):
    fill(s, [
        "**Here, we have discussed",
        (1, "1. The regulatory problem: enhancers act across megabases, and sequence-only "
            "models must infer relationships that have already been measured"),
        (1, "2. A mechanism: measured contact graphs biasing attention inside the encoder "
            "throughout pretraining, rather than fusion or post-hoc alignment"),
        (1, "3. A controlled test: matched B0-B5 arms, chromosome and cell-line held-out "
            "splits, and a Hi-C-free inference probe"),
        (1, "4. A discipline: no number without a tracked file, no superiority claim without "
            "a matched setting"),
        "**Future work",
        (1, "Multi-scale contacts (5 kb plus 25-50 kb) for domains and compartments"),
        (1, "More cell lines; epigenomic tracks as additional node or edge features"),
        (1, "Variant-effect prediction on non-coding GWAS variants"),
        (1, "Cross-species transfer to mouse; single-cell Hi-C with sparse contacts"),
        (1, "A stronger backbone via parameter-efficient adaptation of Caduceus"),
    ], 16)


def s37_references(s, ctx):
    fill(s, [(0, r, 12) for r in REFERENCES], 12,
         notes="Author lists for [5], [6] and [7] are to be completed against the published "
               "record before submission. They are cited here by title and identifier so that "
               "every entry is verifiable exactly as written.")


FILLERS = {
    1: s01_review_details, 2: s02_abstract, 3: s03_outline, 4: s04_domain, 5: s05_sdg,
    6: s06_aim, 7: s07_questions, 8: s08_title_just, 9: s09_objectives, 10: s10_scope,
    11: s11_timeline, 12: s12_intro, 13: s13_related_a, 14: s14_related_b,
    15: s15_related_table, 16: s16_gap, 17: s17_sdlc, 18: s18_usecase, 19: s19_sequence,
    20: s20_activity, 21: s21_requirements, 22: s22_data, 23: s23_prep, 24: s24_sysarch,
    25: s25_modules, 26: s26_model, 27: s27_algorithms, 28: s28_setup, 29: s29_results,
    30: s30_results_reading, 33: s33_comparison, 34: s34_poc, 35: s35_outcomes,
    36: s36_conclusion, 37: s37_references,
}

# NOTE: template slide 1 is deliberately absent from every deck. It is the
# marking-scheme slide and carries the template's own instruction not to
# present it; its text also lives in mc:AlternateContent blocks that
# python-pptx cannot see or delete.
PROPOSAL = [0] + list(range(2, 17)) + [37]                     # sections 1-11 + references
DESIGN = [0] + list(range(2, 28)) + [37]                       # + SDLC, UML, data, architecture
INTERIM = [0] + list(range(2, 31)) + [34, 37]                  # + setup and results placeholders
FULL = [0] + list(range(2, 31)) + [33, 34, 35, 36, 37]         # everything

DECKS = {
    "Mini_Project_Review_0":       (PROPOSAL, 0),
    "Mini_Project_Review_1":       (DESIGN, 1),
    "Mini_Project_Review_2":       (INTERIM, 3),
    "Mini_Project_End_Sem_Review": (FULL, 6),
}

# Short talk, still on the college template: title, abstract, gap, model, conclusion.
PITCH = [0, 2, 16, 26, 36]


def build(name: str, keep: list[int], phase: int) -> Path:
    label, date = REVIEWS.get(name, ("Project Pitch", "2026"))
    prs = Presentation(str(TEMPLATE))

    # the template's own table style, reused on slides that had no sample table
    tbl_style = None
    for shape in prs.slides[15].shapes:
        if shape.has_table:
            tbl_style = shape.table._tbl.tblPr
            break

    ctx = {"label": label, "date": date, "phase": phase, "tbl_style": tbl_style,
           "status": f"Phase {phase} in progress"}

    keep_set = set(keep)
    for i, slide in enumerate(prs.slides):
        if i not in keep_set:
            continue
        if i == 0:
            fill_title_slide(slide, label, date)
            continue
        if i in FILLERS:
            FILLERS[i](slide, ctx)
        # the template titles carry editorial markers -- "15. Data Collection
        # [Example]" -- that must not reach a review room
        title = slide.shapes.title
        if title is not None:
            cleaned = re.sub(r"\s*\[[^\]]*\]", "", title.text_frame.text).strip()
            if cleaned != title.text_frame.text:
                for para in title.text_frame.paragraphs:
                    for run in para.runs:
                        run.text = ""
                title.text_frame.paragraphs[0].runs[0].text = cleaned

    delete_slides(prs, keep_set)
    out = OUT / f"{name}.pptx"
    prs.save(str(out))
    return out


def main() -> int:
    if not TEMPLATE.exists():
        print(f"template not found: {TEMPLATE}", file=sys.stderr)
        return 1
    needed = ["chromgraphfm_architecture_wide", "system_architecture", "usecase",
              "sequence", "activity", "sdlc"]
    missing = [n for n in needed if not (FIG / f"{n}.png").exists()]
    if missing:
        print(f"missing figures: {missing}\nrun scripts/export_diagrams.py first", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    for name, (keep, phase) in list(DECKS.items()) + [("pitch_deck", (PITCH, 1))]:
        p = build(name, keep, phase)
        print(f"{p.name:38s} {len(Presentation(str(p)).slides._sldIdLst):>3} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
