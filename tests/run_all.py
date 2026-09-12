"""Run every check the repository has.

    python tests/run_all.py

Three suites, none of which needs a GPU, a download, or real data:

    tests/test_config.py     the config refuses the mistakes that corrupt a run
    docs/verify_math.py      32 numerical checks on the maths in the explainer
    scripts/smoke_test.py    34 end-to-end checks on synthetic chromosomes

If all three pass, the code is internally consistent. That is a different claim
from "the model works", which only Phase 5 on real data can settle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITES = [
    ("config", ROOT / "tests" / "test_config.py"),
    ("maths", ROOT / "docs" / "verify_math.py"),
    ("pipeline", ROOT / "scripts" / "smoke_test.py"),
]


def main() -> int:
    results = []
    for name, script in SUITES:
        print(f"\n{'=' * 62}\n{name}  --  {script.relative_to(ROOT)}\n{'=' * 62}")
        code = subprocess.call([sys.executable, str(script)], cwd=ROOT)
        results.append((name, code == 0))

    print(f"\n{'=' * 62}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} suite(s) failed: {', '.join(failed)}")
        return 1
    print("\nall suites passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
