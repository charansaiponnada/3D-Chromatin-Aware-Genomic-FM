"""Export every docs/diagrams/*.drawio to figures/*.png.

    python scripts/export_diagrams.py

Diagrams are authored in draw.io and exported here so the decks and the report
never contain a hand-placed image that has drifted from its source.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "diagrams"
OUT = REPO / "figures"

CANDIDATES = [
    Path(r"C:\Program Files\draw.io\draw.io.exe"),
    Path(r"C:\Program Files (x86)\draw.io\draw.io.exe"),
    Path.home() / "AppData/Local/Programs/draw.io/draw.io.exe",
]


def find_drawio() -> Path | None:
    for c in CANDIDATES:
        if c.exists():
            return c
    which = shutil.which("drawio") or shutil.which("draw.io")
    return Path(which) if which else None


def main() -> int:
    exe = find_drawio()
    if exe is None:
        print("draw.io desktop not found. Install it, or export manually from "
              "app.diagrams.net (File > Export as > PNG, zoom 200%).", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    failed = []
    for src in sorted(SRC.glob("*.drawio")):
        png = OUT / f"{src.stem}.png"
        r = subprocess.run([str(exe), "-x", "-f", "png", "--scale", "2", "--border", "10",
                            "-o", str(png), str(src)],
                           capture_output=True, text=True)
        ok = png.exists() and r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {src.name} -> {png.name}")
        if not ok:
            failed.append(src.name)

    if failed:
        # The exporter reports only "Export failed" with no cause. The one we have
        # actually hit: a cell whose id is literally "join" kills the export.
        print(f"\nfailed: {failed}\nCheck for reserved cell ids (e.g. id=\"join\").",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
