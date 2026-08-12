"""Compile project-docs/project.tex and leave only the .tex and .pdf behind.

The project rule is that project-docs/ holds exactly two files: the LaTeX source
and its compiled PDF. Every auxiliary file LaTeX produces is removed after the
run, and the build happens in a scratch directory so intermediates never touch
the tracked folder in the first place.

Run:  python scripts/build_project_doc.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCDIR = REPO / "project-docs"
TEX = DOCDIR / "project.tex"
PDF = DOCDIR / "project.pdf"

ENGINES = ["pdflatex", "xelatex", "lualatex"]
AUX_SUFFIXES = {".aux", ".log", ".out", ".toc", ".lof", ".lot", ".fls",
                ".fdb_latexmk", ".synctex.gz", ".bbl", ".blg", ".nav",
                ".snm", ".vrb", ".xdv"}


def find_engine() -> str:
    for e in ENGINES:
        if shutil.which(e):
            return e
    raise SystemExit("no LaTeX engine found (tried: " + ", ".join(ENGINES) + ")")


def main() -> int:
    if not TEX.exists():
        raise SystemExit(f"missing {TEX}")
    engine = find_engine()
    print(f"engine: {engine}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # two passes so the table of contents resolves
        for i in (1, 2):
            proc = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error",
                 "--enable-installer", f"-output-directory={tmpdir}",
                 str(TEX)],
                cwd=DOCDIR, capture_output=True, text=True, timeout=900,
            )
            print(f"  pass {i}: exit {proc.returncode}")
            if proc.returncode != 0:
                log = tmpdir / "project.log"
                tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
                print("\n".join(tail))
                print(proc.stdout[-2000:])
                return 1

        built = tmpdir / "project.pdf"
        if not built.exists():
            print("no PDF produced")
            return 1
        shutil.copyfile(built, PDF)

    # belt and braces: clear anything a previous run may have left in place
    removed = []
    for f in DOCDIR.iterdir():
        if f.suffix in AUX_SUFFIXES or f.name.endswith(".synctex.gz"):
            f.unlink()
            removed.append(f.name)
        elif f.name == ".ipynb_checkpoints" and f.is_dir():
            # JupyterLab recreates this whenever project.tex or project.pdf is
            # open in it, so deleting it by hand does not stay deleted -- it
            # reappeared within two minutes of the previous build. It holds only
            # Jupyter's autosave copies of files this script has just rebuilt,
            # so nothing unique is lost. Removing it here is what keeps the
            # "project-docs/ contains only project.tex and project.pdf" rule
            # true after a build rather than only until Jupyter next notices.
            shutil.rmtree(f)
            removed.append(f.name + "/")

    kept = sorted(p.name for p in DOCDIR.iterdir())
    print(f"  removed: {removed if removed else 'nothing'}")
    print(f"  project-docs/ now contains: {kept}")
    print(f"\n{PDF.relative_to(REPO)}  ({PDF.stat().st_size/1e6:.2f} MB)")
    return 0 if kept == ["project.pdf", "project.tex"] else 1


if __name__ == "__main__":
    sys.exit(main())
