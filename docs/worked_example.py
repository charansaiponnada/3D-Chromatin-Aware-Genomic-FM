"""A complete, hand-checkable forward pass through a toy ChromGraphFM.

    python docs/worked_example.py

Writes docs/generated/we_values.tex -- LaTeX macros holding every number that
appears in the explainer document. The PDF reads its numbers from here, so the
worked example in the write-up can never drift from the arithmetic that
produced it. Same rule the decks and the website follow: no number by hand.

The toy is deliberately tiny so a reader can redo any step on paper:

    6 windows (not 128)      8 bp per window (not 5000)
    d = 4 (not 256)          1 attention head (not 8)
    local radius 1           top-1 distal edge per node
    1 held-out target edge

Everything is float64 and rounded only at print time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

np.set_printoptions(precision=4, suppress=True)

OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# 0. the toy setting
# --------------------------------------------------------------------------- #

W = 6          # windows in the sample
LBP = 8        # bases per window
D = 4          # model dimension
LOCAL_R = 1    # sequential neighbours kept each side
TOPK = 1       # strongest distal partners kept per window
MIN_SEP = 2    # a "distal" edge must span at least this many windows
BIN_KB = 5     # each window is 5 kb, as in the real config

# Six 8-base windows. Window 1 and window 5 are the pair that will turn out to
# be in contact: 1 carries a CTCF-like motif core (CCCTC) and 5 carries its
# reverse complement (GAGGG). That is the pattern that anchors real loops.
DNA = [
    "ATGCATGC",   # 0
    "CCCTCAGT",   # 1  <- motif
    "TTATATAA",   # 2
    "GCGCGCGC",   # 3
    "AATTAATT",   # 4
    "TGAGGGCA",   # 5  <- reverse-complement motif
]

BASES = "ACGT"


def one_hot(seq: str) -> np.ndarray:
    """(4, L) one-hot. Rows are A, C, G, T."""
    m = np.zeros((4, len(seq)))
    for t, ch in enumerate(seq):
        m[BASES.index(ch), t] = 1.0
    return m


# --------------------------------------------------------------------------- #
# 1. local encoder  (a transparent stand-in for the Bi-Mamba encoder)
# --------------------------------------------------------------------------- #
# The real encoder is a bidirectional state-space model. Its internals are not
# what the worked example is about, and its arithmetic cannot be done on paper,
# so here it is replaced by the smallest function with the same TYPE: it eats a
# window of DNA and emits one d-dimensional vector.
#
#   f_i  = base composition of window i          (4 numbers, sums to 1)
#   h_i  = tanh(W_enc f_i + b_enc)               (d numbers)
#
# Everything downstream -- the graph, the attention, the losses -- is the real
# mechanism, unchanged.

W_ENC = np.array([
    [ 1.2, -0.8,  0.6,  0.4],
    [-0.5,  1.4, -0.2,  0.7],
    [ 0.9,  0.3, -1.1,  0.2],
    [ 0.2, -0.6,  1.0, -0.9],
])
B_ENC = np.array([0.10, -0.20, 0.05, 0.15])

comp = np.stack([one_hot(s).sum(axis=1) / LBP for s in DNA])       # (W, 4)
H = np.tanh(comp @ W_ENC.T + B_ENC)                                # (W, D)

# --------------------------------------------------------------------------- #
# 2. Hi-C: raw counts -> distance-detrended strength
# --------------------------------------------------------------------------- #
# Raw counts fall off steeply with genomic separation. Window 1 and window 5
# carry an extra 40 counts on top of that background: the loop.

# Three things shape a real contact count, and all three are in here:
#   1. genomic separation      -- the steep background decay
#   2. compartment membership  -- A-type regions prefer A, B prefers B
#   3. anchor strength         -- some windows are simply stickier than others
# Without ingredient 2 and 3 every observed/expected ratio would come out at
# exactly 1.0, the bias would be constant across a node's neighbours, and the
# softmax would cancel it completely. See the "when a bias does nothing" note
# in the write-up -- that is not a hypothetical, it is what this script printed
# the first time it was run.
COMPARTMENT = [0, 1, 0, 1, 0, 1]          # A, B, A, B, A, B
ANCHOR = [1.00, 1.25, 0.90, 1.10, 0.95, 1.30]

RAW = np.zeros((W, W))
for i in range(W):
    for j in range(W):
        if i == j:
            continue
        background = 100.0 / (abs(i - j) ** 1.5)
        same_compartment = 1.30 if COMPARTMENT[i] == COMPARTMENT[j] else 0.75
        RAW[i, j] = round(background * same_compartment * ANCHOR[i] * ANCHOR[j], 1)
RAW[1, 5] += 40.0
RAW[5, 1] += 40.0

# Expected counts at each separation = mean of the observed counts at that
# separation. This is the detrending step, and it is the reason the model
# cannot win by learning "near things touch".
sep_expected = {}
for s in range(1, W):
    vals = [RAW[i, i + s] for i in range(W - s)]
    sep_expected[s] = float(np.mean(vals))

OE = np.zeros((W, W))          # observed / expected
for i in range(W):
    for j in range(W):
        if i == j:
            continue
        OE[i, j] = RAW[i, j] / sep_expected[abs(i - j)]

# Squash into (0, 1) so the bias function sees a bounded input.
C = OE / (1.0 + OE)

# --------------------------------------------------------------------------- #
# 3. build the sparse graph
# --------------------------------------------------------------------------- #

local_edges, distal_edges = set(), set()
for i in range(W):
    for d in range(1, LOCAL_R + 1):
        if i + d < W:
            local_edges.add((i, i + d))

for i in range(W):
    cands = [(j, C[i, j]) for j in range(W) if abs(i - j) >= MIN_SEP]
    cands.sort(key=lambda t: -t[1])
    for j, _ in cands[:TOPK]:
        distal_edges.add((min(i, j), max(i, j)))

HELD_OUT = (1, 5)                                   # the loop becomes the target
cond_distal = sorted(distal_edges - {HELD_OUT})
cond_edges = sorted(local_edges | set(cond_distal))

def neighbours(i: int) -> list[int]:
    out = [j for (a, b) in cond_edges for j in (a, b)
           if (a == i or b == i) and j != i]
    return sorted(set(out))

# --------------------------------------------------------------------------- #
# 4. contact-biased attention for one node
# --------------------------------------------------------------------------- #

FOCUS = 2          # the window we follow all the way through

W_Q = np.array([
    [ 0.5,  0.1, -0.3,  0.2],
    [-0.2,  0.4,  0.1,  0.3],
    [ 0.3, -0.1,  0.5, -0.2],
    [ 0.1,  0.2, -0.4,  0.6],
])
W_K = np.array([
    [ 0.4, -0.2,  0.1,  0.3],
    [ 0.2,  0.5, -0.3,  0.1],
    [-0.1,  0.3,  0.4, -0.2],
    [ 0.3,  0.1,  0.2,  0.5],
])
W_V = np.array([
    [ 0.6,  0.2, -0.1,  0.1],
    [ 0.1,  0.5,  0.2, -0.3],
    [-0.2,  0.1,  0.7,  0.2],
    [ 0.2, -0.1,  0.3,  0.4],
])

Q = H @ W_Q.T
K = H @ W_K.T
V = H @ W_V.T

# The two learned bias functions, kept linear so they can be differentiated by
# hand. In the real model each is a small MLP.
ALPHA = 3.0        # slope of b_HiC
BETA = 0.8         # slope of b_dist
GAMMA = 0.0        # intercept of b_dist


def b_hic(c: float) -> float:
    return ALPHA * c


def b_dist(sep_bins: int) -> float:
    return -BETA * np.log2(sep_bins + 1) + GAMMA


nbrs = neighbours(FOCUS)
raw_scores, bias_hic, bias_dist, logits = [], [], [], []
for j in nbrs:
    s = float(Q[FOCUS] @ K[j]) / np.sqrt(D)
    bh = b_hic(C[FOCUS, j])
    bd = b_dist(abs(FOCUS - j))
    raw_scores.append(s)
    bias_hic.append(bh)
    bias_dist.append(bd)
    logits.append(s + bh + bd)

logits = np.array(logits)
shift = logits.max()
exps = np.exp(logits - shift)
attn = exps / exps.sum()
attn_out = (attn[:, None] * V[nbrs]).sum(axis=0)

# --------------------------------------------------------------------------- #
# 5. residual, LayerNorm, feed-forward
# --------------------------------------------------------------------------- #

EPS = 1e-5


def layer_norm(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(x.mean())
    var = float(x.var())
    return (x - mu) / np.sqrt(var + EPS), mu, var


u = H[FOCUS] + attn_out
u_norm, u_mu, u_var = layer_norm(u)

W1 = np.array([
    [ 0.8, -0.3,  0.2,  0.5],
    [ 0.1,  0.6, -0.4,  0.2],
    [-0.5,  0.2,  0.7, -0.1],
    [ 0.3,  0.4,  0.1,  0.6],
    [ 0.2, -0.2,  0.5,  0.3],
    [-0.4,  0.7,  0.1,  0.2],
    [ 0.6,  0.1, -0.3,  0.4],
    [ 0.1,  0.3,  0.6, -0.5],
])
B1 = np.zeros(8)
W2 = np.array([
    [ 0.4,  0.1, -0.2,  0.3,  0.2, -0.1,  0.5,  0.2],
    [-0.1,  0.5,  0.3, -0.2,  0.4,  0.2, -0.3,  0.1],
    [ 0.2, -0.3,  0.6,  0.1, -0.2,  0.4,  0.1,  0.3],
    [ 0.3,  0.2, -0.1,  0.5,  0.1,  0.3,  0.2, -0.4],
])
B2 = np.zeros(4)

ffn_hidden = np.maximum(0.0, W1 @ u_norm + B1)
ffn_out = W2 @ ffn_hidden + B2
z_focus, z_mu, z_var = layer_norm(u_norm + ffn_out)

# Run every node through the same block so the loss terms below are honest.
Z = np.zeros((W, D))
for i in range(W):
    nb = neighbours(i)
    lg = np.array([
        float(Q[i] @ K[j]) / np.sqrt(D) + b_hic(C[i, j]) + b_dist(abs(i - j))
        for j in nb
    ])
    a = np.exp(lg - lg.max())
    a = a / a.sum()
    o = (a[:, None] * V[nb]).sum(axis=0)
    un, _, _ = layer_norm(H[i] + o)
    fh = np.maximum(0.0, W1 @ un + B1)
    Z[i], _, _ = layer_norm(un + W2 @ fh + B2)

# --------------------------------------------------------------------------- #
# 6. the three losses
# --------------------------------------------------------------------------- #

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# 6a. contact prediction on the held-out edge
W_G = np.array([0.6, -0.2, 0.4, 0.3, 0.5, 0.1, -0.3, 0.2, -0.7])
B_G = 0.1
i_h, j_h = HELD_OUT
feat = np.concatenate([Z[i_h] * Z[j_h], np.abs(Z[i_h] - Z[j_h]), [np.log2(abs(i_h - j_h) + 1)]])
logit_c = float(W_G @ feat) + B_G
c_hat = sigmoid(logit_c)
c_true = float(C[i_h, j_h])
l_contact = -(c_true * np.log(c_hat) + (1 - c_true) * np.log(1 - c_hat))

# 6b. distance-matched contrastive, anchor = window 1
TAU = 0.5


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


anchor, positive = HELD_OUT
sep_pos = abs(positive - anchor)

# Distance matching compares PAIRS at equal separation, not other partners of
# the same anchor. Two reasons. There is often no second partner at a given
# separation from one anchor -- at W = 6 and separation 4 there is exactly one
# -- and more fundamentally, the quantity being controlled for is the
# separation of the pair, so the comparison has to be pair against pair.
negative_pairs = [
    (a, a + sep_pos)
    for a in range(W - sep_pos)
    if (a, a + sep_pos) != (min(anchor, positive), max(anchor, positive))
]
sim_pos = cosine(Z[anchor], Z[positive])
sims_neg = [cosine(Z[a], Z[b]) for (a, b) in negative_pairs]
num = np.exp(sim_pos / TAU)
den = num + sum(np.exp(s / TAU) for s in sims_neg)
l_contrast = float(-np.log(num / den))

# 6c. masked-window reconstruction
# NOTE: this replaces per-base masked reconstruction. A d-dimensional pooled
# window vector cannot carry 8 bases (let alone 5000) of sequence, so a
# per-base head reading only h_i is information-theoretically unable to
# succeed. Masking the whole window embedding and rebuilding it from graph
# neighbours keeps the objective well posed AND tests the mechanism directly:
# the only route to the answer is through the contacts.
MASKED = 3
nb_m = neighbours(MASKED)
h_recon = H[nb_m].mean(axis=0)
l_dna = float(np.mean((h_recon - H[MASKED]) ** 2))

LAM_DNA, LAM_CON, LAM_CONTACT = 1.0, 0.5, 1.0
l_total = LAM_DNA * l_dna + LAM_CON * l_contrast + LAM_CONTACT * l_contact

# --------------------------------------------------------------------------- #
# 7. one gradient, by hand
# --------------------------------------------------------------------------- #
# d(attention weight a_k) / d(alpha), for the focus node. Softmax derivative:
#   da_k/dalpha = a_k * (c_k - sum_m a_m c_m)
# A non-zero value here is what "the model learns how much to trust Hi-C" means.

c_nb = np.array([C[FOCUS, j] for j in nbrs])
mean_c = float(attn @ c_nb)
dattn_dalpha = attn * (c_nb - mean_c)

# --------------------------------------------------------------------------- #
# 8. emit LaTeX macros
# --------------------------------------------------------------------------- #

WORDS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def num_to_words(n: int | str) -> str:
    return "".join(WORDS[int(ch)] for ch in str(n))


macros: dict[str, str] = {}


def put(name: str, value, fmt: str = "{:.4f}") -> None:
    macros[name] = fmt.format(value) if isinstance(value, (int, float, np.floating)) else str(value)


def put_vec(prefix: str, vec: np.ndarray, fmt: str = "{:.4f}") -> None:
    for i, v in enumerate(vec):
        put(f"{prefix}{num_to_words(i)}", float(v), fmt)


# setting
put("WEwindows", W, "{:d}")
put("WEbp", LBP, "{:d}")
put("WEdim", D, "{:d}")
put("WEbinkb", BIN_KB, "{:d}")
put("WEfocus", FOCUS, "{:d}")
put("WEmasked", MASKED, "{:d}")
put("WEalpha", ALPHA, "{:.1f}")
put("WEbeta", BETA, "{:.1f}")
put("WEtau", TAU, "{:.1f}")
put("WEheldi", HELD_OUT[0], "{:d}")
put("WEheldj", HELD_OUT[1], "{:d}")
put("WEnedges", len(cond_edges), "{:d}")
put("WEnbrs", ", ".join(str(j) for j in nbrs))

for i, s in enumerate(DNA):
    put(f"WEdna{num_to_words(i)}", s)

# composition + embeddings
for i in range(W):
    put_vec(f"WEcomp{num_to_words(i)}", comp[i], "{:.3f}")
    put_vec(f"WEh{num_to_words(i)}", H[i])
    put_vec(f"WEz{num_to_words(i)}", Z[i])

# Hi-C
for s in range(1, W):
    put(f"WEexp{num_to_words(s)}", sep_expected[s], "{:.2f}")
for (a, b) in [(0, 1), (1, 2), (2, 3), (1, 5), (0, 5), (1, 3), (2, 4)]:
    tag = num_to_words(a) + num_to_words(b)
    put(f"WEraw{tag}", RAW[a, b], "{:.1f}")
    put(f"WEoe{tag}", OE[a, b], "{:.3f}")
    put(f"WEc{tag}", C[a, b], "{:.4f}")

# attention for the focus node
for idx, j in enumerate(nbrs):
    t = num_to_words(j)
    put(f"WEscore{t}", raw_scores[idx])
    put(f"WEbhic{t}", bias_hic[idx])
    put(f"WEbdist{t}", bias_dist[idx])
    put(f"WElogit{t}", logits[idx])
    put(f"WEexpo{t}", float(exps[idx]))
    put(f"WEattn{t}", float(attn[idx]))
    put(f"WEcnb{t}", float(c_nb[idx]))
    put(f"WEgrad{t}", float(dattn_dalpha[idx]))
put("WEshift", float(shift))
put("WEexpsum", float(exps.sum()))
put("WEmeanc", mean_c)

put_vec("WEq", Q[FOCUS])
put_vec("WEattnout", attn_out)
put_vec("WEu", u)
put_vec("WEunorm", u_norm)
put_vec("WEffnout", ffn_out)
put("WEumu", u_mu)
put("WEuvar", u_var)

# losses
put("WEchat", c_hat)
put("WEctrue", c_true)
put("WElogitc", logit_c)
put("WElcontact", l_contact)
put("WEsimpos", sim_pos)
put("WEnegs", ", ".join("(%d,%d)" % ab for ab in negative_pairs))
put("WEseppos", sep_pos, "{:d}")
put("WEsimnegone", sims_neg[0] if sims_neg else 0.0)
put("WElcontrast", l_contrast)
put("WEldna", l_dna)
put("WEltotal", l_total)
put("WElamdna", LAM_DNA, "{:.1f}")
put("WElamcon", LAM_CON, "{:.1f}")
put("WElamcontact", LAM_CONTACT, "{:.1f}")

lines = [
    "% Generated by docs/worked_example.py -- do not edit by hand.",
    "% Every number in the worked example of the explainer comes from here.",
    "",
]
for name, value in macros.items():
    lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

(OUT / "we_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# a machine-readable copy, and a console trace for checking by hand
dump = {
    "dna": DNA,
    "composition": comp.round(4).tolist(),
    "H": H.round(4).tolist(),
    "raw_counts": RAW.round(2).tolist(),
    "expected_by_separation": {str(k): round(v, 3) for k, v in sep_expected.items()},
    "obs_over_exp": OE.round(4).tolist(),
    "contact_strength": C.round(4).tolist(),
    "local_edges": sorted(local_edges),
    "distal_edges": sorted(distal_edges),
    "held_out": list(HELD_OUT),
    "negative_pairs": [list(p) for p in negative_pairs],
    "conditioning_edges": cond_edges,
    "focus": FOCUS,
    "neighbours": nbrs,
    "raw_scores": [round(v, 4) for v in raw_scores],
    "b_hic": [round(v, 4) for v in bias_hic],
    "b_dist": [round(v, 4) for v in bias_dist],
    "logits": logits.round(4).tolist(),
    "attention": attn.round(4).tolist(),
    "Z": Z.round(4).tolist(),
    "losses": {
        "dna": round(l_dna, 4),
        "contrast": round(l_contrast, 4),
        "contact": round(l_contact, 4),
        "total": round(l_total, 4),
    },
    "d_attn_d_alpha": dattn_dalpha.round(4).tolist(),
}
(OUT / "we_values.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")

print(f"windows                 {W} x {LBP} bp")
print(f"local edges             {sorted(local_edges)}")
print(f"distal edges            {sorted(distal_edges)}")
print(f"held out (target)       {HELD_OUT}")
print(f"conditioning edges      {cond_edges}")
print(f"neighbours of {FOCUS}         {nbrs}")
print(f"raw scores              {np.round(raw_scores, 4)}")
print(f"b_HiC                   {np.round(bias_hic, 4)}")
print(f"b_dist                  {np.round(bias_dist, 4)}")
print(f"logits                  {logits.round(4)}")
print(f"attention               {attn.round(4)}  (sums to {attn.sum():.4f})")
print(f"z_{FOCUS}                     {Z[FOCUS].round(4)}")
print(f"contact  target {c_true:.4f}  predicted {c_hat:.4f}  loss {l_contact:.4f}")
print(f"contrast sim+ {sim_pos:.4f}  negative pairs {negative_pairs} (all span {sep_pos})  loss {l_contrast:.4f}")
print(f"masked window {MASKED}         loss {l_dna:.4f}")
print(f"total loss              {l_total:.4f}")
print(f"d(attn)/d(alpha)        {dattn_dalpha.round(4)}")
print(f"\nwrote {OUT/'we_values.tex'} ({len(macros)} macros)")
