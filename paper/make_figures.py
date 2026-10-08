"""Figures for the paper (paper/figures/*.png), drawn from the code and the built data.

    python paper/make_figures.py

Every figure here is either a schematic that follows chromgraph/ exactly, or a
plot of the real GM12878 data. Nothing is a mock-up: if the code or the data
change, re-run this and the figures follow.

  fig1_overview.png        the model, as chromgraph/model.py and train.py build it
  fig2_detrending.png      distance decay P(s) and detrended strength, from the data
  fig3_graph_sample.png    one real training sample's graph, and what each control keeps
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from chromgraph.config import data_dir, load_config               # noqa: E402
from chromgraph.data import load_chromosome                        # noqa: E402

OUT = REPO / "paper" / "figures"
SAMPLE = REPO / "website" / "src" / "content" / "graph-sample.json"

# Validated categorical palette (light mode), one hue per ROLE, fixed everywhere.
SEQ = "#2a78d6"       # sequence stream / local backbone
STRUCT = "#4a3aa7"    # Hi-C structure
TARGET = "#eb6834"    # held-out targets
NEUTRAL = "#52514e"   # secondary ink
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": NEUTRAL,
    "axes.labelcolor": INK,
    "xtick.color": NEUTRAL,
    "ytick.color": NEUTRAL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
})

HEAT = LinearSegmentedColormap.from_list("struct_seq", ["#ffffff", "#d9d4f3", "#8e81dc", STRUCT, "#241a63"])


def tint(hex_color: str, amount: float) -> str:
    """Mix a colour toward white: amount 0 = colour, 1 = white."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = c + (255 - c) * amount
    return "#" + "".join(f"{int(round(v)):02x}" for v in c)


# --------------------------------------------------------------------------- #
# Figure 1 -- overview schematic
# --------------------------------------------------------------------------- #

def fig1_overview(cfg) -> None:
    m, d, t = cfg.model, cfg.data, cfg.train
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", color=NEUTRAL, fill=0.88, bold=True, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=tint(color, fill), ec=color, lw=0.9,
                                    ls=(0, (3, 2)) if dashed else "-"))
        ty = y + h / 2 + (1.25 if sub else 0)
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=6.5,
                fontweight="bold" if bold else "normal", color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.55, sub, ha="center", va="center",
                    fontsize=5.5, color=NEUTRAL, linespacing=1.15)
        return (x, y, w, h)

    def arrow(p, q, color=NEUTRAL, dashed=False, rad=0.0, lw=0.9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw,
                                     color=color, ls=(0, (3, 2)) if dashed else "-",
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))

    ax.text(1, 58.2, "Sequence stream", fontsize=7, color=SEQ, fontweight="bold")
    ax.text(1, 21.3, "Structure stream (training only)", fontsize=7, color=STRUCT, fontweight="bold")

    # sequence stream
    box(1, 45, 16, 10, "DNA windows",
        f"{d.nodes_per_sample} × {d.bin_size // 1000} kb = "
        f"{d.nodes_per_sample * d.bin_size // 1000} kb\none-hot A/C/G/T/N", SEQ)
    box(21, 45, 17, 10, "Convolutional tower",
        f"{m.conv_depth} residual blocks,\nmax-pool ×2 each (÷{2 ** m.conv_depth})", SEQ)
    box(42, 45, 17, 10, "Bidirectional SSM",
        f"{m.encoder_layers} diagonal state-space\nlayers, mean-pooled", SEQ)
    box(63, 45, 13, 10, r"$h_i$", "window\nembedding", SEQ)
    arrow((17.3, 50), (20.7, 50))
    arrow((38.3, 50), (41.7, 50))
    arrow((59.3, 50), (62.7, 50))

    # structure stream
    box(1, 7, 16, 11, "Hi-C contact map",
        f"GM12878, {d.bin_size // 1000} kb bins,\nICE-balanced", STRUCT)
    box(21, 7, 17, 11, "Distance detrending",
        r"O/E over $P(s)$; strength" + "\n" + r"$c_{ij} = \mathrm{O/E}\,/\,(1+\mathrm{O/E})$", STRUCT)
    box(42, 7, 17, 11, "Sparse contact graph",
        f"±{d.local_radius} local + top-{d.top_k_edges} distal\n"
        f"{int(d.held_out_edge_frac * 100)}% distal edges held out", STRUCT)
    box(63, 7, 13, 11, "Edge bias",
        r"$b_{\mathrm{HiC}}(c_{ij})$" + "\n" + r"$+\,b_{\mathrm{dist}}(d_{ij})$", STRUCT)
    arrow((17.3, 12.5), (20.7, 12.5))
    arrow((38.3, 12.5), (41.7, 12.5))
    arrow((59.3, 12.5), (62.7, 12.5))

    # contact-biased blocks
    box(42, 25.5, 34, 13, f"Contact-biased attention  × {m.block_layers}",
        r"$\alpha_{ij} = \mathrm{softmax}_{j \in \mathcal{N}(i)}"
        r"\left(q_i^\top k_j/\sqrt{d_h} + b_{ij}\right)$" + "\n"
        "attention over graph edges only; add & norm, FFN", STRUCT, fill=0.92)
    arrow((69.5, 44.7), (69.5, 38.8), SEQ)                       # h -> blocks
    arrow((69.5, 18.3), (69.5, 25.2), STRUCT)                    # bias -> blocks
    ax.text(68.6, 21.8, f"structure dropout\np = {m.structure_dropout}",
            fontsize=5.4, color=STRUCT, va="center", ha="right", style="italic")

    # outputs
    box(81, 45.5, 18, 9, r"$z_i$", "structure-conditioned\nrepresentation", STRUCT, fill=0.9)
    box(81, 30.5, 18, 9, r"$z_i^{\mathrm{free}}$", "Hi-C-free pass\n(local edges only)", SEQ, fill=0.9,
        dashed=True)
    arrow((76.3, 34.5), (80.7, 49.5), STRUCT, rad=-0.2)
    arrow((76.3, 31.5), (80.7, 34.5), SEQ, dashed=True)
    ax.annotate("", xy=(90, 39.8), xytext=(90, 45.2),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=0.9, ls=(0, (3, 2)),
                                mutation_scale=7))
    ax.text(91.2, 42.5, "distill\n(stop-grad)", fontsize=5.9, color=NEUTRAL, va="center")

    # objectives
    ox, ow = 80.2, 19.6
    objs = [
        ("Masked-window recon.", rf"$\lambda={t.lambda_dna:g}$"),
        ("Distance-matched InfoNCE", rf"$\lambda={t.lambda_contrast:g}$"),
        ("Contact regression", rf"$\log\,\mathrm{{O/E}}$, held-out edges, $\lambda={t.lambda_contact:g}$"),
        ("Structure distillation", rf"$\|z^{{\mathrm{{free}}}}-\mathrm{{sg}}(z)\|^2$, $\lambda={t.lambda_distill:g}$"),
    ]
    for k, (name, sub) in enumerate(objs):
        y = 20.6 - k * 5.6
        ax.add_patch(FancyBboxPatch((ox, y), ow, 4.8, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#ffffff", ec=GRID, lw=0.8))
        ax.text(ox + 0.8, y + 3.3, name, fontsize=5.3, fontweight="bold", color=INK, va="center")
        ax.text(ox + 0.8, y + 1.2, sub, fontsize=4.9, color=NEUTRAL, va="center")
    ax.text(ox, 26.4, "Pretraining objectives", fontsize=6.6, color=INK, fontweight="bold")

    # held-out edges feed only the contact objective
    ax.add_patch(FancyArrowPatch((50.5, 6.7), (80.7, 7.5), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=TARGET, ls=(0, (3, 2)),
                                 connectionstyle="arc3,rad=0.25", shrinkA=1, shrinkB=1))
    ax.text(57, 1.2, "held-out target edges: predicted, never attended over",
            fontsize=5.9, color=TARGET, style="italic")

    fig.savefig(OUT / "fig1_overview.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 -- distance decay and detrending, from the built data
# --------------------------------------------------------------------------- #

def fig2_detrending(cfg) -> None:
    chroms = ["chr1", "chr8", "chr15", "chr20"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"wspace": 0.32})

    sep_kb, raw_by, oe_by = None, {}, {}
    for chrom in chroms:
        cd = load_chromosome(data_dir(), "GM12878", chrom, cfg.data.bin_size)
        sep = (cd["col"] - cd["row"]).astype(int)
        keep = (sep >= 1) & (sep < 128)
        sep, oe, strength = sep[keep], cd["oe"][keep], cd["strength"][keep]
        # mean O/E over observed pixels is biased by sparsity; show the median
        med = np.array([np.median(oe[sep == s]) for s in range(1, 128)])
        oe_by[chrom] = med
        exp = cd["qc"].get("expected_by_separation", {})
        raw_by[chrom] = {int(k): v for k, v in exp.items()}
        if chrom == "chr20":
            strength_all = strength
            sep_all = sep
    sep_kb = np.arange(1, 128) * cfg.data.bin_size / 1000

    # (a) P(s): the expected curve the detrending divides by
    shades = [tint(NEUTRAL, f) for f in (0.0, 0.3, 0.5, 0.65)]
    for chrom, col in zip(chroms, shades):
        ks = sorted(raw_by[chrom])
        if ks:
            a.plot(np.array(ks) * cfg.data.bin_size / 1000, [raw_by[chrom][k] for k in ks],
                   "-o", color=col, lw=1.2, ms=2.5, label=chrom)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Genomic separation (kb)")
    a.set_ylabel("Expected balanced contact $P(s)$")
    a.legend(frameon=False, fontsize=6.5, loc="lower left")
    a.set_title("a   Contact frequency decays with distance", loc="left", fontsize=8, fontweight="bold")
    a.grid(True, which="major", color=GRID, lw=0.5)

    # (b) after detrending: strength is flat in separation; distal edges are its upper tail
    bins = np.linspace(0, 1, 41)
    near = strength_all[sep_all < cfg.data.min_separation]
    far = strength_all[sep_all >= cfg.data.min_separation]
    b.hist(far, bins=bins, color=tint(STRUCT, 0.55), label=f"pixels ≥ {cfg.data.min_separation} bins apart",
           density=True, edgecolor="white", lw=0.4)
    b.hist(near, bins=bins, histtype="step", color=SEQ, lw=1.2, density=True,
           label=f"pixels < {cfg.data.min_separation} bins apart")
    sample = json.loads(SAMPLE.read_text())
    edge_s = np.array([e[2] for e in sample["arms"]["full"]])
    b.hist(edge_s, bins=bins, histtype="step", color=STRUCT, lw=1.4, density=True,
           label="selected top-k distal edges")
    b.axvline(0.5, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)))
    b.text(0.49, b.get_ylim()[1] * 0.97, "O/E = 1", fontsize=6.2, color=NEUTRAL, ha="right", va="top")
    b.set_xlabel(r"Detrended strength $c_{ij}=\mathrm{O/E}/(1+\mathrm{O/E})$")
    b.set_ylabel("Density")
    b.legend(frameon=False, fontsize=6.0, loc="upper right", bbox_to_anchor=(1.02, 1.0))
    b.set_xlim(0, 1.0)
    b.set_title("b   Detrended strength (chr20)", loc="left", fontsize=8, fontweight="bold")

    fig.savefig(OUT / "fig2_detrending.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 -- one real sample's graph, and the controls
# --------------------------------------------------------------------------- #

def fig3_graph_sample() -> None:
    s = json.loads(SAMPLE.read_text())
    n, radius, src = s["n"], s["local_radius"], s["source"]

    def matrix_for(arm: str) -> np.ndarray:
        """Upper: measured strength. Lower: the strength this arm conditions on."""
        mtx = np.full((n, n), np.nan)
        for i, j, v in s["matrix"]:
            mtx[i, j] = v
        for i, j, v in s["arms"][arm]:
            mtx[j, i] = v
        return mtx

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.62], hspace=0.42, wspace=0.18)

    # (a) full matrix
    a = fig.add_subplot(gs[0, 0:2])
    mtx = matrix_for("full")
    a.imshow(np.clip((mtx - 0.4) / 0.45, 0, 1), cmap=HEAT, vmin=0, vmax=1, interpolation="nearest")
    ii = np.arange(n)
    for dd in range(1, radius + 1):
        a.scatter(ii[:-dd], ii[dd:], s=1.2, color=SEQ, marker="s", lw=0)
    tg = np.array(s["targets"])
    a.scatter(tg[:, 0], tg[:, 1], s=3.5, color=TARGET, marker="s", lw=0)
    a.plot([0, n - 1], [0, n - 1], color=GRID, lw=0.6)
    a.set_xticks([0, 63, 127])
    a.set_yticks([0, 63, 127])
    a.tick_params(labelsize=6)
    a.text(1.03, 0.97, "upper:\nmeasured\nstrength", transform=a.transAxes, ha="left", va="top",
           fontsize=6, color=NEUTRAL)
    a.text(1.03, 0.03, "lower:\nencoder\ninput", transform=a.transAxes, ha="left", va="bottom",
           fontsize=6, color=NEUTRAL)
    a.set_title("a   Contact map and graph (full)", loc="left", fontsize=8, fontweight="bold")
    a.set_xlabel("window", fontsize=6.5)
    a.set_ylabel("window", fontsize=6.5)
    for side in ("top", "right"):
        a.spines[side].set_visible(True)

    # (b) arcs
    b = fig.add_subplot(gs[0, 2:4])
    for i, j, v in s["arms"]["full"]:
        x = np.linspace(i, j, 40)
        h = (j - i) / (n - 1)
        y = h * np.sin(np.pi * (x - i) / (j - i))
        b.plot(x, y, color=STRUCT, lw=0.3, alpha=0.03 + 0.3 * max(0, (v - 0.55) / 0.35))
    for i, j, _ in s["targets"]:
        x = np.linspace(i, j, 40)
        h = (j - i) / (n - 1)
        y = -0.65 * h * np.sin(np.pi * (x - i) / (j - i))
        b.plot(x, y, color=TARGET, lw=0.35, alpha=0.3)
    b.axhline(0, color=SEQ, lw=1.6)
    b.set_xlim(-1, n)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xticks([0, 63, 127])
    b.tick_params(labelsize=6)
    b.set_xlabel(f"window  ({src['chrom']}:{src['start_bp'] / 1e6:.2f}–"
                 f"{(src['start_bp'] + n * src['bin_size']) / 1e6:.2f} Mb)", fontsize=6.5)
    b.text(0, 1.04, f"{len(s['arms']['full'])} conditioning edges (attended over)",
           fontsize=6.3, color=STRUCT, va="top")
    b.text(0, -0.72, f"{len(s['targets'])} held-out targets (predicted only)",
           fontsize=6.3, color=TARGET, va="bottom")
    b.set_title("b   The same graph as arcs", loc="left", fontsize=8, fontweight="bold")

    # (c) controls: lower triangle only
    full_keys = {(i, j) for i, j, _ in s["arms"]["full"]}
    measured = {(i, j): v for i, j, v in s["matrix"]}
    labels = [("full", "Full"), ("b3_shuffled_hic", "B3 shuffled Hi-C"),
              ("b2_distance_only", "B2 distance-only"), ("b1_random_graph", "B1 random graph")]
    for k, (arm, name) in enumerate(labels):
        c = fig.add_subplot(gs[1, k])
        low = np.full((n, n), np.nan)
        for i, j, v in s["arms"][arm]:
            low[j, i] = v
        c.imshow(np.clip((low - 0.4) / 0.45, 0, 1), cmap=HEAT, vmin=0, vmax=1, interpolation="nearest")
        c.plot([0, n - 1], [0, n - 1], color=GRID, lw=0.6)
        c.set_xticks([])
        c.set_yticks([])
        for side in ("top", "right"):
            c.spines[side].set_visible(True)
        edges = s["arms"][arm]
        shared = np.mean([(i, j) in full_keys for i, j, _ in edges])
        pairs = [(v, measured[(i, j)]) for i, j, v in edges if (i, j) in measured]
        g, mm = np.array(pairs).T
        r = np.corrcoef(g, mm)[0, 1] if np.std(g) > 0 else float("nan")
        c.set_title(name, fontsize=7, fontweight="bold", color=INK, pad=3)
        c.set_xlabel(f"topology {shared * 100:.0f}% · r = " + ("—" if np.isnan(r) else f"{r:.2f}"),
                     fontsize=6.2, color=NEUTRAL)
        if k == 0:
            c.text(-0.08, 1.14, "c   What each arm conditions on", transform=c.transAxes,
                   fontsize=8, fontweight="bold")

    fig.savefig(OUT / "fig3_graph_sample.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs" / "v2.yaml")
    fig1_overview(cfg)
    fig2_detrending(cfg)
    fig3_graph_sample()
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
