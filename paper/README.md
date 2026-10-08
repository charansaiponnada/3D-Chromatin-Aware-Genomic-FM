# Paper

`master_copy.docx` is the working manuscript: single column, A4, Times New Roman 11 pt.

| File | Role |
|---|---|
| `content.js` | the text: sections, captions, tables, citation keys |
| `build_master.js` | assembles `master_copy.docx` from the text, figures and bibliography |
| `bibliography.json` | references, each verified against the publisher, arXiv, PubMed or OpenReview (2026-10-08) |
| `make_figures.py` | Figures 1–3, drawn from `chromgraph/` and the built GM12878 data |
| `make_equations.py` | display equations, rendered to `figures/eq_*.png` |
| `REVIEW_NOTES.md` | self-review and claim–evidence map (not part of the paper) |

```bash
cd paper && npm install
../.venv/bin/python make_figures.py && ../.venv/bin/python make_equations.py
node build_master.js
```

## Rules the build enforces

- A citation key that is missing from `bibliography.json`, or not marked VERIFIED there, fails the build.
- Figures, tables and equations are numbered automatically, and `{fig:…}` and `{eq:…}` references resolve to those numbers.

## Rules the text follows

- No experimental outcome is stated until it has been decided on validation and confirmed once on the sealed test split. Every placeholder is marked `[PENDING RESULTS]`.

## Hand edits

Once someone edits `master_copy.docx` in Word, stop rebuilding it from `content.js`, because a rebuild overwrites those edits. From then on, edit the `.docx` in place.
