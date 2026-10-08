// Build paper/master_copy.docx from content.js, figures/ and bibliography.json.
//
//   cd paper && npm install && node make_figures… (see README)  &&  node build_master.js
//
// Single column, A4, 1-inch margins, Times New Roman 11 pt. Citations are
// numbered in order of first appearance; every key must resolve to a VERIFIED
// entry of bibliography.json or the build fails -- a reference nobody checked
// does not reach the paper.

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat,
  Packer, PageNumber, Paragraph, ShadingType, Table, TableCell, TableRow, TabStopType,
  TextRun, WidthType,
} = require("docx");

const C = require("./content.js");
const HERE = __dirname;
const BIB = JSON.parse(fs.readFileSync(path.join(HERE, "bibliography.json"), "utf8"));

// Short keys used in the text -> bibliography.json keys.
const ALIAS = {
  akita: "fudenberg2020akita", alphagenome: "avsec2026alphagenome", basenji: "kelley2018basenji",
  caduceus: "schiff2024caduceus", chimaera: "shkolikov2026chimaera", chrome: "ye2026chrome",
  chromoformer: "lee2022chromoformer", corigami: "tan2023corigami", cpc: "oord2018cpc",
  dekker4dn: "dekker2017_4dn", "4dn": "reiff2022_4dnportal", dixon2012tads: "dixon2012tads",
  dnabert: "ji2021dnabert", dnabert2: "zhou2024dnabert2", dropedge: "rong2020dropedge",
  enformer: "avsec2021enformer", epcot: "zhang2023epcot", evo: "nguyen2024evo",
  evo2: "brixi2026evo2", evo2hic: "fang2025evo2hic", fudenberg2016: "fudenberg2016loopextrusion",
  gat: "velickovic2018gat", graphormer: "ying2021graphormer", graphreg: "karbalayghareh2022graphreg",
  hicfoundation: "wang2026hicfoundation", hinton2015distill: "hinton2015distilling",
  hyenadna: "nguyen2023hyenadna", imakaev2012ice: "imakaev2012ice",
  liebermanaiden2009: "liebermanaiden2009hic", mamba: "gu2024mamba", mixhic: "yang2025mixhic",
  moddrop: "neverova2016moddrop", ntransformer: "dallatorre2025nt", orca: "zhou2022orca",
  puget: "hang2025puget", rao2014: "rao2014kilobase", s4: "gu2022s4", s4d: "gu2022s4d",
  simclr: "chen2020simclr", transformer: "vaswani2017attention", vapnik2009lupi: "vapnik2009lupi",
};

const FONT = "Times New Roman";
const BODY = 22;          // half-points: 11 pt
const SMALL = 18;         // 9 pt captions
const TEXT_WIDTH_IN = 6.27;

// ---------------------------------------------------------------- numbering --
const citeOrder = [];
function citeNumber(shortKey) {
  const key = ALIAS[shortKey] || shortKey;
  const entry = BIB.find((e) => e.key === key);
  if (!entry) throw new Error(`citation [@${shortKey}] -> '${key}' is not in bibliography.json`);
  if (entry.status !== "VERIFIED") throw new Error(`citation '${key}' is ${entry.status}; verify it first`);
  let n = citeOrder.indexOf(key);
  if (n < 0) { citeOrder.push(key); n = citeOrder.length - 1; }
  return n + 1;
}

const figNum = {}, tabNum = {}, eqNum = {};
let nFig = 0, nTab = 0, nEq = 0;
// Pre-assign numbers in document order so forward references resolve.
(function preassign() {
  for (const sec of C.method) {
    if (sec.figure && !figNum[sec.figure]) figNum[sec.figure] = ++nFig;
    for (const k of ["eqs", "eqs2", "eqs3", "eqs4"]) for (const e of sec[k] || []) eqNum[e] = ++nEq;
    for (const item of sec.flow || []) if (item.eq) eqNum[item.eq] = ++nEq;
    if (sec.table && !tabNum[sec.table]) tabNum[sec.table] = ++nTab;
  }
})();

function compressRange(nums) {
  const s = [...new Set(nums)].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < s.length; i++) {
    let j = i;
    while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
    out.push(j - i >= 2 ? `${s[i]}–${s[j]}` : j === i ? `${s[i]}` : `${s[i]},${s[j]}`);
    i = j;
  }
  return out.join(",");
}

// ------------------------------------------------------------ inline markup --
// Returns TextRun[] for a string with citations, refs, **bold**, *italic*, $math$.
function runs(text, base = {}) {
  text = text
    .replace(/\[@([^\]]+)\]/g, (_, keys) => `[${compressRange(keys.split(",").map((k) => citeNumber(k.trim())))}]`)
    .replace(/\{fig:(\w+)\}/g, (_, k) => { if (!figNum[k]) throw new Error(`no figure ${k}`); return `Figure ${figNum[k]}`; })
    .replace(/\{tab:(\w+)\}/g, (_, k) => { if (!tabNum[k]) throw new Error(`no table ${k}`); return `Table ${tabNum[k]}`; })
    .replace(/\{eq:(\w+)\}/g, (_, k) => { if (!eqNum[k]) throw new Error(`no equation ${k}`); return `Eq. (${eqNum[k]})`; });

  const out = [];
  const re = /(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, ...base }));
    const tok = m[0];
    if (tok.startsWith("$")) out.push(...mathRuns(tok.slice(1, -1), base));
    else if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, size: BODY, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, size: BODY, ...base }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: BODY, ...base }));
  return out;
}

// Minimal TeX-ish inline maths: Greek/macros to Unicode, _/^ to sub/superscript,
// Latin letters italic, digits and operators upright.
const MACROS = {
  "\\alpha": "α", "\\lambda": "λ", "\\tau": "τ", "\\sigma": "σ", "\\rho": "ρ", "\\xi": "ξ",
  "\\ell": "ℓ", "\\pm": "±", "\\times": "×", "\\in": "∈", "\\cup": "∪", "\\le": "≤", "\\ge": "≥",
  "\\mathbb{R}": "ℝ", "\\mathcal{N}": "𝒩", "\\mathcal{M}": "ℳ", "\\mathcal{T}": "𝒯",
  "\\mathcal{D}": "𝒟", "\\mathcal{F}": "ℱ", "\\mathcal{E}": "ℰ", "\\mathcal{L}": "ℒ",
  "\\log": "log", "\\tilde z": "z̃", "\\,": " ", "\\ ": " ",
};
function mathRuns(src, base) {
  // \mathrm{..} -> upright text marker
  let s = src;
  for (const [k, v] of Object.entries(MACROS)) s = s.split(k).join(v);
  const out = [];
  let i = 0;
  const push = (t, opts = {}) => out.push(new TextRun({ text: t, font: FONT, size: base.size || BODY, ...base, ...opts }));
  const readGroup = () => {
    if (s[i] === "{") {
      let depth = 1, j = i + 1;
      while (j < s.length && depth) { if (s[j] === "{") depth++; else if (s[j] === "}") depth--; j++; }
      const g = s.slice(i + 1, j - 1); i = j; return g;
    }
    const g = s[i]; i += 1; return g;
  };
  const emit = (t, opts) => {
    // \mathrm{x} upright, else letters italic
    const rm = /\\mathrm\{([^}]*)\}/g;
    let last = 0, m;
    while ((m = rm.exec(t))) {
      if (m.index > last) letters(t.slice(last, m.index), opts);
      push(m[1], opts);
      last = rm.lastIndex;
    }
    if (last < t.length) letters(t.slice(last), opts);
  };
  const letters = (t, opts) => {
    const parts = t.split(/([A-Za-zα-ωℓ]+)/);
    for (const p of parts) if (p) push(p, /[A-Za-zα-ωℓ]/.test(p) && p !== "log" ? { italics: true, ...opts } : opts);
  };
  let buf = "";
  while (i < s.length) {
    const ch = s[i];
    if (ch === "_" || ch === "^") {
      if (buf) { emit(buf, {}); buf = ""; }
      i += 1;
      const g = readGroup();
      emit(g, ch === "_" ? { subScript: true } : { superScript: true });
    } else if (ch === "\\" && s.startsWith("\\mathrm{", i)) {
      const j = s.indexOf("}", i);
      buf += s.slice(i, j + 1); i = j + 1;
    } else { buf += ch; i += 1; }
  }
  if (buf) emit(buf, {});
  return out;
}

// ------------------------------------------------------------------ blocks --
const P = (text, opts = {}) => new Paragraph({
  children: runs(text),
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 276 },
  ...opts,
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text, bold: true, font: FONT, size: 26 })],
  spacing: { before: 280, after: 120 },
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text, bold: true, font: FONT, size: BODY })],
  spacing: { before: 200, after: 80 },
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figure(key) {
  const f = C.figures[key];
  const file = path.join(HERE, f.file);
  const { w, h } = pngSize(file);
  const widthIn = Math.min(f.width, TEXT_WIDTH_IN);
  const px = (inch) => Math.round(inch * 96);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      keepNext: true,
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(file),
        transformation: { width: px(widthIn), height: px(widthIn * h / w) },
        altText: { title: `Figure ${figNum[key]}`, description: f.caption.replace(/[*$\\{}]/g, ""), name: key },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 200, line: 240 },
      children: [new TextRun({ text: `Figure ${figNum[key]}. `, bold: true, font: FONT, size: SMALL }),
        ...runs(f.caption, { size: SMALL })],
    }),
  ];
}

function equation(name) {
  const file = path.join(HERE, "figures", `eq_${name}.png`);
  const { w, h } = pngSize(file);
  let widthIn = w / 600;                                   // rendered at 600 dpi, 11 pt
  let heightIn = h / 600;
  const maxW = TEXT_WIDTH_IN - 0.7;
  if (widthIn > maxW) { heightIn *= maxW / widthIn; widthIn = maxW; }
  const px = (inch) => Math.round(inch * 96);
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: 4514 }, { type: TabStopType.RIGHT, position: 9026 }],
    spacing: { before: 100, after: 100 },
    children: [
      new TextRun({ text: "\t", font: FONT, size: BODY }),
      new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: px(widthIn), height: px(heightIn) },
        altText: { title: `Equation ${eqNum[name]}`, description: name, name: `eq_${name}` } }),
      new TextRun({ text: `\t(${eqNum[name]})`, font: FONT, size: BODY }),
    ],
  });
}

const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function table(key) {
  const t = C.tables[key];
  const total = t.widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, { head = false, last = false } = {}) => new TableCell({
    width: { size: t.widths[i], type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    borders: { top: head ? thin : none, bottom: head || last ? thin : none, left: none, right: none },
    shading: head ? { type: ShadingType.CLEAR, color: "auto", fill: "F2F2F2" } : undefined,
    children: [new Paragraph({ children: runs(text, { size: SMALL, ...(head ? { bold: true } : {}) }) })],
  });
  return [
    new Paragraph({
      keepNext: true,
      spacing: { before: 200, after: 80 },
      children: [new TextRun({ text: `Table ${tabNum[key]}. `, bold: true, font: FONT, size: SMALL }),
        ...runs(t.caption, { size: SMALL })],
    }),
    new Table({
      width: { size: total, type: WidthType.DXA },
      columnWidths: t.widths,
      rows: [
        new TableRow({ tableHeader: true, children: t.header.map((h, i) => cell(h, i, { head: true })) }),
        ...t.rows.map((r, ri) => new TableRow({
          children: r.map((v, i) => cell(v, i, { last: ri === t.rows.length - 1 })),
        })),
      ],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ assemble --
const body = [];

body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: C.title, bold: true, font: FONT, size: 32 })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: C.authors, font: FONT, size: BODY })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: C.affiliation, italics: true, font: FONT, size: SMALL })],
}));

// draft-status box (shaded paragraphs, removed at submission)
for (const [k, note] of C.draftNote.entries()) {
  body.push(new Paragraph({
    shading: { type: ShadingType.CLEAR, color: "auto", fill: "FFF4E5" },
    border: k === 0 ? { top: { style: BorderStyle.SINGLE, size: 6, color: "EB6834", space: 4 } } : undefined,
    spacing: { after: k === C.draftNote.length - 1 ? 240 : 0 },
    children: [new TextRun({ text: k === 0 ? "Draft status. " : "", bold: true, font: FONT, size: SMALL }),
      ...runs(note, { size: SMALL })],
  }));
}

body.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({ text: "Abstract", bold: true, font: FONT, size: BODY })],
}));
for (const p of C.abstract) body.push(P(p));
body.push(new Paragraph({
  spacing: { after: 240 },
  children: [new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: BODY }),
    new TextRun({ text: C.keywords, font: FONT, size: BODY })],
}));

// 1 Introduction
body.push(H1("1  Introduction"));
for (const p of C.introduction) body.push(P(p));
for (const c of C.contributions) {
  body.push(new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 276 },
    children: runs(c),
  }));
}

// 2 Related work
body.push(H1("2  Related Work"));
C.related.forEach((sec, k) => {
  body.push(H2(`2.${k + 1}  ${sec.heading}`));
  for (const p of sec.paras) body.push(P(p));
});

// 3 Methodology
body.push(H1("3  Methodology"));
C.method.forEach((sec, k) => {
  body.push(H2(`3.${k + 1}  ${sec.heading}`));
  for (const item of sec.flow || []) body.push(item.eq ? equation(item.eq) : P(item.p));
  for (const p of sec.paras || []) body.push(P(p));
  for (const e of sec.eqs || []) body.push(equation(e));
  for (const p of sec.parasAfter || []) body.push(P(p));
  for (const p of sec.parasAfter2 || []) body.push(P(p));
  for (const e of sec.eqs2 || []) body.push(equation(e));
  for (const p of sec.parasAfter3 || []) body.push(P(p));
  for (const e of sec.eqs3 || []) body.push(equation(e));
  for (const p of sec.parasAfter4 || []) body.push(P(p));
  for (const e of sec.eqs4 || []) body.push(equation(e));
  for (const p of sec.parasAfter5 || []) body.push(P(p));
  if (sec.figure) body.push(...figure(sec.figure));
  if (sec.table) body.push(...table(sec.table));
});

// Sections still to be written
body.push(H1("4  Results"));
body.push(P("[PENDING RESULTS — written from the final multi-seed validation results and the single sealed test evaluation.]"));
body.push(H1("5  Discussion"));
body.push(P("[PENDING]"));
body.push(H1("6  Conclusion"));
body.push(P("[PENDING]"));

// References
body.push(H1("References"));
citeOrder.forEach((key, n) => {
  const e = BIB.find((b) => b.key === key);
  const vp = e.volume_pages ? `;${e.volume_pages}` : "";
  body.push(new Paragraph({
    alignment: AlignmentType.LEFT,
    indent: { left: 440, hanging: 440 },
    spacing: { after: 60, line: 240 },
    children: [
      new TextRun({ text: `[${n + 1}]\t`, font: FONT, size: SMALL }),
      new TextRun({ text: `${e.authors}. ${e.title}. `, font: FONT, size: SMALL }),
      new TextRun({ text: e.venue, italics: true, font: FONT, size: SMALL }),
      new TextRun({ text: `. ${e.year}${vp}. ${e.doi_or_url}`, font: FONT, size: SMALL }),
    ],
    tabStops: [{ type: TabStopType.LEFT, position: 440 }],
  }));
});

const doc = new Document({
  creator: "ChromGraphFM authors",
  title: C.title,
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "000000" },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, bold: true, color: "000000" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "(%1)", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 360 } } } }],
    }],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 },               // A4
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SMALL })] })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(HERE, "master_copy.docx");
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out}: ${citeOrder.length} references, ${nFig} figures, ${nTab} tables, ${nEq} equations`);
});
