"""Check the mathematics in the explainer, numerically.

    python docs/verify_math.py

Every claim the write-up makes about the maths is re-derived here and compared
against the worked example. The derivation that matters most -- the softmax
gradient with respect to the Hi-C bias slope -- is checked against a central
finite difference, which will disagree if the algebra is wrong.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import worked_example as we  # noqa: E402  (importing runs the forward pass)

TOL = 1e-9
failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  --  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


print("\nSOFTMAX")

check("attention weights sum to 1",
      abs(we.attn.sum() - 1.0) < TOL,
      f"sum = {we.attn.sum():.12f}")

check("attention weights are all positive",
      bool((we.attn > 0).all()))


def attn_from_logits(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


# The claim in Box 5 of the write-up: adding the same constant to every score
# for a fixed i leaves the weights unchanged.
shifted = attn_from_logits(we.logits + 137.0)
check("softmax is shift-invariant (Box 5)",
      np.allclose(shifted, we.attn, atol=TOL),
      f"max deviation {np.abs(shifted - we.attn).max():.2e}")

# The consequence the write-up draws from it: b_scale(r) at a single resolution
# adds a per-row constant, so it cannot affect anything.
b_scale_constant = 2.5
with_bscale = attn_from_logits(we.logits + b_scale_constant)
check("a constant b_scale(r) term changes nothing",
      np.allclose(with_bscale, we.attn, atol=TOL),
      "confirms removing it was correct, not cosmetic")


print("\nGRADIENT  (the derivation in Equation 7)")

FOCUS = we.FOCUS
nbrs = we.nbrs
c_nb = np.array([we.C[FOCUS, j] for j in nbrs])
dna_scores = np.array([float(we.Q[FOCUS] @ we.K[j]) / np.sqrt(we.D) for j in nbrs])
dist_bias = np.array([we.b_dist(abs(FOCUS - j)) for j in nbrs])


def attn_at_alpha(alpha: float) -> np.ndarray:
    return attn_from_logits(dna_scores + alpha * c_nb + dist_bias)


analytic = we.attn * (c_nb - float(we.attn @ c_nb))

h = 1e-6
numeric = (attn_at_alpha(we.ALPHA + h) - attn_at_alpha(we.ALPHA - h)) / (2 * h)

check("d(attention)/d(alpha) matches central finite difference",
      np.allclose(analytic, numeric, atol=1e-7),
      f"max |analytic - numeric| = {np.abs(analytic - numeric).max():.2e}")

check("the gradient sums to zero (weights must keep summing to 1)",
      abs(analytic.sum()) < 1e-12,
      f"sum = {analytic.sum():.2e}")

check("the gradient is not identically zero",
      np.abs(analytic).max() > 1e-6,
      f"max |grad| = {np.abs(analytic).max():.6f}")

# The sign claim made in the write-up.
above = c_nb > float(we.attn @ c_nb)
check("neighbours above the weighted-average contact gain weight",
      bool((analytic[above] > 0).all() and (analytic[~above] < 0).all()))


print("\nATTENTION OUTPUT")

V_nb = we.V[nbrs]
lo, hi = V_nb.min(axis=0), V_nb.max(axis=0)
check("output is a convex combination of the value vectors",
      bool((we.attn_out >= lo - TOL).all() and (we.attn_out <= hi + TOL).all()))

check("attention uses graph neighbours only",
      set(nbrs) == {j for (a, b) in we.cond_edges for j in (a, b)
                    if FOCUS in (a, b) and j != FOCUS})

check("the held-out edge is absent from the conditioning graph",
      we.HELD_OUT not in we.cond_edges,
      f"held out {we.HELD_OUT}")


print("\nLAYER NORM")

for label, vec in [("after attention", we.u_norm), ("final z", we.Z[FOCUS])]:
    check(f"{label}: mean is 0", abs(vec.mean()) < 1e-9, f"mean = {vec.mean():.2e}")
    check(f"{label}: variance is 1", abs(vec.var() - 1.0) < 1e-4, f"var = {vec.var():.8f}")


print("\nHI-C DETRENDING")

for s in range(1, we.W):
    oe = np.array([we.OE[i, i + s] for i in range(we.W - s)])
    check(f"O/E at separation {s} averages to 1",
          abs(oe.mean() - 1.0) < 1e-9,
          f"mean = {oe.mean():.12f}")

check("contact strengths lie strictly inside (0, 1)",
      bool(((we.C > 0) & (we.C < 1))[~np.eye(we.W, dtype=bool)].all()))

check("the loop is the strongest detrended contact",
      we.OE[we.HELD_OUT] == we.OE[~np.eye(we.W, dtype=bool)].max(),
      f"O/E = {we.OE[we.HELD_OUT]:.4f}")

check("the loop is NOT the strongest raw contact",
      we.RAW[we.HELD_OUT] < we.RAW[~np.eye(we.W, dtype=bool)].max(),
      "which is the whole reason detrending is required")


print("\nLOSSES")

i_h, j_h = we.HELD_OUT
p, y = we.c_hat, we.c_true
bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
check("contact loss matches the binary cross-entropy formula",
      abs(bce - we.l_contact) < 1e-12,
      f"{bce:.10f}")

perfect = -(y * np.log(y) + (1 - y) * np.log(1 - y))
check("cross-entropy is minimised by a perfect prediction",
      perfect < we.l_contact,
      f"floor {perfect:.4f} < achieved {we.l_contact:.4f}")

check("contrastive loss is non-negative", we.l_contrast >= 0.0)
check("masked-window loss is non-negative", we.l_dna >= 0.0)

check("cosine similarities are in [-1, 1]",
      -1.0 <= we.sim_pos <= 1.0 and all(-1.0 <= s <= 1.0 for s in we.sims_neg))

total = (we.LAM_DNA * we.l_dna + we.LAM_CON * we.l_contrast
         + we.LAM_CONTACT * we.l_contact)
check("total loss is the stated weighted sum",
      abs(total - we.l_total) < 1e-12,
      f"{total:.10f}")


print("\nCONTRASTIVE NEGATIVES  (the claim that they are distance-matched)")

sep_pos = abs(we.positive - we.anchor)
seps = [abs(b - a) for (a, b) in we.negative_pairs]
check("every negative pair spans the same number of bins as the positive",
      all(s == sep_pos for s in seps),
      f"positive spans {sep_pos}; negatives span {seps}")

check("the positive pair is not among the negatives",
      (min(we.anchor, we.positive), max(we.anchor, we.positive)) not in we.negative_pairs)


print("\nDETERMINISM")

import importlib  # noqa: E402

first_Z = we.Z.copy()
importlib.reload(we)
check("re-running reproduces identical numbers",
      np.array_equal(first_Z, we.Z),
      "no hidden randomness")


print(f"\n{'-' * 62}")
if failures:
    print(f"{len(failures)} of {checks} checks FAILED:")
    for f in failures:
        print(f"    - {f}")
    raise SystemExit(1)
print(f"all {checks} checks passed")
