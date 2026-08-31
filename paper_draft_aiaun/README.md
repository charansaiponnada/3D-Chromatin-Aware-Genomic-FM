# ChromaTau — paper draft (AIAYN format)

**Framing: this is a MODEL paper.** The earlier negative-results framing is kept
at `main_negative_framing.tex.bak` in case the Phase D result argues for
reverting to it.

`main.tex` → `main.pdf`. **10 pages: 9 of write-up, references alone on page 10**,
matching the requested format (arXiv:1706.03762).

Build:
```
cd paper_draft_aiaun && pdflatex main.tex && pdflatex main.tex
```
Two passes are needed for cross-references. Compiles with 0 errors.

## Figures

All three are **TikZ/pgfplots, drawn in-document** — no external image files, no
draw.io round-trip. They are vector, restyleable, and their coordinates are the
measured numbers, so a figure cannot drift from the result it plots.

| figure | what it shows |
|---|---|
| Fig. 1 | the timescale ceiling $\tau_{\max}=1/\Delta_{\min}$ against window and TAD scale |
| Fig. 2 | architecture — structural pathway biasing $\Delta$, bottom-up like AIAYN Fig. 1 |
| Fig. 3 | keep($\varphi$) across window width (a) and bin resolution (b) |

## Provenance

**Every number in the draft came from a command actually run in this repo**, per
standing rule 1. Sources:

| claim | file |
|---|---|
| $\tau$ ceiling, F4 fix, main null, controls, D1/D3 | `results/novel_model/`, `results/baselines/` |
| memory + throughput (Table 2) | `results/p5_memcheck.json` |
| keep($\varphi$) window scan (Fig. 3a) | `results/novel_model/p5_window_scan.json` |
| bin resolution (Fig. 3b) | `results/b2_phi_resolution_probe.json`, `..._control_5kb.json` |
| benchmark table | `results/c1_genomic_benchmarks.json` |
| cross-cell-line conservation | `results/c2_differential_power.json` |

## Status — read before circulating

- **Phase D is RUNNING (launched 2026-08-31).** The paper proposes ChromaTau =
  dual-scale conditioning + permeability disabled, which is exactly what Phase D
  trains. §7 says the main evaluation is in progress and reports no numbers for
  it. **Fill Table 3's successor from `results/phase_d/` only when runs reach
  `status: COMPLETED`.**
- **The pilot is labelled "superseded" and the paper states the three reasons it
  could not have produced a positive** (permeability ceiling, per-position
  conditioning, 3v3 permutation floor 0.100). Do not soften that paragraph — it
  is what makes reporting the pilot at all defensible.
- **The benchmark table is a partial sweep** (3 of 8 tasks complete at time of
  writing). Regenerate the table from `c1_genomic_benchmarks.json` when the
  sweep finishes.
- **Author block, affiliation and acknowledgements are placeholders.**
- The permeability ceiling means the reported null is partly a null about a
  handicapped mechanism. This is stated in "What we do not claim" and must stay.
