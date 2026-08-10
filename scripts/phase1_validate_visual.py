"""Phase 1, step 3 -- visual validation of the pilot pipeline.

Produces the figure the Phase 1 gate requires: a TAD block on the diagonal, a
loop as an off-diagonal focal dot, and evidence that sequence coordinates and
contact-matrix coordinates actually agree.

Every annotation is externally sourced, not asserted by this script:
  * TAD boundaries      4DN 4DNFIVK5JOFU (their own caller)
  * loop support        4DN 4DNFI9SL1WSF, GM12878 CTCF in situ ChIA-PET, GRCh38
  * gene coordinates    GENCODE v47, looked up from our own GTF at runtime

Note on loop references: the widely used Rao 2014 HiCCUPS loop list
(GSE63525) is **hg19** and is deliberately NOT used here -- its maximum chr9
anchor is 140,570,000, beyond GRCh38 chr9's 138,394,717, which is how the
assembly mismatch was caught.

Run:  python scripts/phase1_validate_visual.py
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

REPO = Path(__file__).resolve().parents[1]
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
FIGS = REPO / "figures"

CHROM = "chr9"
RES = 5_000
REGION = (132_000_000, 138_000_000)
LANDMARKS = ["VAV2", "COL5A1", "NOTCH1", "TRAF2", "GRIN1"]

# Loop-detection constraints. The first version of this script used a 1.5 Mb
# ceiling and no local-maximum test, and its top hit was a sparse-corner
# artifact at 1425 kb separation. These bounds are the fix.
LOOP_MIN_SEP_BP = 100_000
LOOP_MAX_SEP_BP = 1_000_000
LOOP_MIN_OE = 2.0
LOOP_MIN_ENRICH = 1.8


def load_band_dense(lo_bin: int, hi_bin: int) -> np.ndarray:
    z = np.load(INTERIM / f"hic_band_{CHROM}_{RES}bp.npz")
    w = int(z["band_bp"]) // RES
    row, col, val = z["row"], z["col"], z["val"]
    m = (row >= lo_bin) & (row < hi_bin) & (col >= lo_bin) & (col < hi_bin)
    k = hi_bin - lo_bin
    M = np.full((k, k), np.nan, dtype=np.float64)
    r, c = row[m] - lo_bin, col[m] - lo_bin
    M[r, c] = val[m]
    M[c, r] = val[m]
    ii, jj = np.indices((k, k))
    inband = np.abs(ii - jj) <= w
    M[inband & ~np.isfinite(M)] = 0.0
    return M


def to_oe(M: np.ndarray) -> np.ndarray:
    k = M.shape[0]
    oe = np.full_like(M, np.nan)
    for d in range(k):
        i = np.arange(k - d)
        v = M[i, i + d]
        if np.isfinite(v).sum() < 10:
            continue
        mu = np.nanmean(v)
        if np.isfinite(mu) and mu > 0:
            oe[i, i + d] = v / mu
            oe[i + d, i] = v / mu
    return oe


def find_loops(M: np.ndarray, oe: np.ndarray, n_top: int = 15) -> list[tuple]:
    """Focal-dot detection with a donut background, restricted to the
    separation range where 5 kb Hi-C loop signal is actually reliable."""
    k = oe.shape[0]
    lo, hi = LOOP_MIN_SEP_BP // RES, LOOP_MAX_SEP_BP // RES
    pad = 6
    # per-separation median of observed, so a candidate must carry real signal
    med_by_d = np.full(k, np.nan)
    for d in range(k):
        i = np.arange(k - d)
        v = M[i, i + d]
        if np.isfinite(v).sum() >= 10:
            med_by_d[d] = np.nanmedian(v)

    cands = []
    for i in range(pad, k - pad):
        for j in range(i + lo, min(i + hi, k - pad)):
            c = oe[i, j]
            if not np.isfinite(c) or c < LOOP_MIN_OE:
                continue
            if not np.isfinite(M[i, j]) or M[i, j] <= med_by_d[j - i]:
                continue
            nb = oe[i - 2:i + 3, j - 2:j + 3]
            if not np.isfinite(nb).all() or c < np.nanmax(nb):
                continue                      # must be a local maximum
            ring = oe[i - pad:i + pad + 1, j - pad:j + pad + 1].copy()
            ring[pad - 2:pad + 3, pad - 2:pad + 3] = np.nan
            bg = np.nanmean(ring)
            if np.isfinite(bg) and bg > 0 and c / bg >= LOOP_MIN_ENRICH:
                cands.append((c / bg, c, i, j))
    cands.sort(reverse=True)
    kept: list[tuple] = []
    for s in cands:
        if all(abs(s[2] - t[2]) > 6 or abs(s[3] - t[3]) > 6 for t in kept):
            kept.append(s)
        if len(kept) >= n_top:
            break
    return kept


def load_boundaries(lo_bin: int, hi_bin: int) -> np.ndarray:
    out = []
    with gzip.open(RAW / "4DNFIVK5JOFU_boundaries_bed.bed.gz", "rt") as f:
        for line in f:
            p = line.split("\t")
            if p[0] == CHROM:
                b = int(p[1]) // RES
                if lo_bin <= b < hi_bin:
                    out.append(b - lo_bin)
    return np.array(sorted(set(out)))


def load_chiapet(lo_bin: int, hi_bin: int, min_pet: int = 4) -> list[tuple]:
    z = np.load(INTERIM / "ctcf_chiapet_chr9.npz")
    pairs, counts = z["pairs"], z["counts"]
    out = []
    for (a, b), c in zip(pairs, counts):
        if c >= min_pet and lo_bin <= a < hi_bin and lo_bin <= b < hi_bin:
            out.append((int(a - lo_bin), int(b - lo_bin), int(c)))
    return sorted(out, key=lambda t: -t[2])


def gene_coords(names: list[str]) -> dict:
    want, found = set(names), {}
    with (INTERIM / f"gencode.v47.{CHROM}.gtf").open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            if p[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]+)"', p[8])
            if m and m.group(1) in want:
                found[m.group(1)] = (int(p[3]), int(p[4]))
    return found


def load_genes_region(lo: int, hi: int) -> list[tuple]:
    genes = []
    with (INTERIM / f"gencode.v47.{CHROM}.gtf").open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            if p[2] != "gene" or 'gene_type "protein_coding"' not in p[8]:
                continue
            s, e = int(p[3]), int(p[4])
            if e >= lo and s <= hi:
                genes.append((s, e))
    return genes


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    lo_bin, hi_bin = REGION[0] // RES, REGION[1] // RES
    k = hi_bin - lo_bin
    print(f"region {CHROM}:{REGION[0]:,}-{REGION[1]:,}  ({k} bins @ {RES} bp)")

    M = load_band_dense(lo_bin, hi_bin)
    oe = to_oe(M)
    z = np.load(PROCESSED / f"phi_{CHROM}_{RES}bp.npz", allow_pickle=True)
    names = list(z["feature_names"])
    ins = z["phi_raw"][lo_bin:hi_bin, names.index("insulation_100kb")]
    comp_all = z["phi_raw"][:, names.index("compartment_pc1")]

    bnds = load_boundaries(lo_bin, hi_bin)
    chia = load_chiapet(lo_bin, hi_bin)
    loops = find_loops(M, oe)
    print(f"  4DN boundaries in region: {len(bnds)}")
    print(f"  CTCF ChIA-PET loops (>=4 PET) in region: {len(chia)}")
    for a, b, c in chia:
        print(f"    {c} PET  chr9:{(lo_bin+a)*RES:,} <-> {(lo_bin+b)*RES:,}"
              f"  sep {(b-a)*RES/1000:.0f} kb")
    print(f"  our loop candidates: {len(loops)}")
    for s, c, i, j in loops[:6]:
        print(f"    enr {s:5.2f} O/E {c:5.2f}  chr9:{(lo_bin+i)*RES:,} <-> "
              f"{(lo_bin+j)*RES:,}  sep {(j-i)*RES/1000:.0f} kb")

    # which of ours is corroborated by ChIA-PET
    corroborated = []
    for s, c, i, j in loops:
        for a, b, pet in chia:
            if abs(i - a) <= 3 and abs(j - b) <= 3:
                corroborated.append((s, c, i, j, pet))
                break
    print(f"  corroborated by CTCF ChIA-PET (within 15 kb): {len(corroborated)}")

    genes = gene_coords(LANDMARKS)
    for g, (s, e) in sorted(genes.items(), key=lambda x: x[1]):
        print(f"    {g:<8} chr9:{s:,}-{e:,}")

    mb = np.array([(lo_bin + i) * RES / 1e6 for i in range(k)])
    ext = [mb[0], mb[-1], mb[-1], mb[0]]

    fig = plt.figure(figsize=(15, 16.5))
    gs = fig.add_gridspec(6, 2, height_ratios=[3.0, 0.45, 0.45, 0.5, 0.55, 1.9],
                          hspace=0.40, wspace=0.18)

    # A -- overview contact map
    axA = fig.add_subplot(gs[0, :])
    pos = M[np.isfinite(M) & (M > 0)]
    axA.imshow(M, cmap="afmhot_r",
               norm=LogNorm(vmin=np.percentile(pos, 1), vmax=np.percentile(pos, 99.7)),
               extent=ext, interpolation="none")
    for b in bnds:
        axA.axvline(mb[b], color="#1f77b4", lw=0.7, alpha=0.5)
    for a, b, c in chia:
        axA.add_patch(Circle((mb[b], mb[a]), 0.06, fill=False, ec="#2ca02c", lw=1.8))
    axA.set_title(
        f"A  Balanced Hi-C, GM12878 (4DNES3JX38V5)  {CHROM}:"
        f"{REGION[0]/1e6:.0f}-{REGION[1]/1e6:.0f} Mb @ {RES//1000} kb\n"
        "blue = 4DN TAD boundary calls   |   green = CTCF ChIA-PET loops "
        "(independent assay, GRCh38)", fontsize=11)
    axA.set_ylabel("position (Mb)")
    axA.set_xlabel("position (Mb)")

    # B -- insulation
    axB = fig.add_subplot(gs[1, :])
    axB.plot(mb, ins, lw=0.9, color="#333333")
    for b in bnds:
        axB.axvline(mb[b], color="#1f77b4", lw=0.7, alpha=0.5)
    axB.axhline(0, color="grey", lw=0.5, ls=":")
    axB.set_xlim(mb[0], mb[-1])
    axB.set_ylabel("insulation", fontsize=9)
    axB.set_title("B  Our insulation score (r = +0.997 vs 4DN's own track) — "
                  "minima fall on the boundary lines", fontsize=10, loc="left")
    axB.tick_params(labelbottom=False)

    # C -- compartments across the WHOLE chromosome, region marked
    axC = fig.add_subplot(gs[2, :])
    xall = np.arange(len(comp_all)) * RES / 1e6
    axC.fill_between(xall, 0, comp_all, where=comp_all >= 0, color="#d62728", lw=0)
    axC.fill_between(xall, 0, comp_all, where=comp_all < 0, color="#1f77b4", lw=0)
    axC.axvspan(REGION[0] / 1e6, REGION[1] / 1e6, color="k", alpha=0.10)
    axC.axhline(0, color="grey", lw=0.5)
    axC.set_xlim(0, xall[-1])
    axC.set_ylabel("PC1", fontsize=9)
    axC.set_xlabel("chr9 position (Mb)", fontsize=9)
    axC.set_title("C  Compartments across all of chr9 (r = +0.976 vs 4DN)  "
                  "red = A, blue = B; grey band = region above; "
                  "gap at 43-60 Mb = centromere/9q12", fontsize=10, loc="left")

    # D -- genes
    axD = fig.add_subplot(gs[3, :])
    pc = load_genes_region(*REGION)
    for s, e in pc:
        axD.add_patch(Rectangle((s / 1e6, 0.30), max((e - s) / 1e6, 0.004), 0.28,
                                color="#444444"))
    for g, (s, e) in genes.items():
        axD.add_patch(Rectangle((s / 1e6, 0.24), max((e - s) / 1e6, 0.012), 0.40,
                                color="#d62728"))
        axD.annotate(g, ((s + e) / 2e6, 0.72), color="#d62728", fontsize=8.5,
                     ha="center", fontweight="bold")
    axD.set_xlim(mb[0], mb[-1])
    axD.set_ylim(0, 1)
    axD.set_yticks([])
    axD.set_xlabel("position (Mb)")
    axD.set_title(f"D  GENCODE v47 protein-coding genes ({len(pc)} in region); "
                  "labelled coordinates read from our own GTF — sequence and "
                  "Hi-C coordinates agree", fontsize=10, loc="left")

    # E -- TAD zoom
    if len(bnds) >= 2:
        gi = int(np.argmax(np.diff(bnds)))
        t0, t1 = bnds[gi], bnds[gi + 1]
    else:
        t0, t1 = 0, min(200, k)
    pad = max(20, (t1 - t0) // 3)
    a0, a1 = max(0, t0 - pad), min(k, t1 + pad)
    axE = fig.add_subplot(gs[5, 0])
    sub = M[a0:a1, a0:a1]
    p2 = sub[np.isfinite(sub) & (sub > 0)]
    axE.imshow(sub, cmap="afmhot_r",
               norm=LogNorm(vmin=np.percentile(p2, 1), vmax=np.percentile(p2, 99.7)),
               extent=[mb[a0], mb[a1 - 1], mb[a1 - 1], mb[a0]], interpolation="none")
    axE.plot([mb[t0], mb[t1], mb[t1], mb[t0], mb[t0]],
             [mb[t0], mb[t0], mb[t1], mb[t1], mb[t0]], color="#1f77b4", lw=1.8)
    axE.set_title(f"E  TAD — self-interacting block, brighter inside than out\n"
                  f"{CHROM}:{(lo_bin+t0)*RES:,}-{(lo_bin+t1)*RES:,} "
                  f"({(t1-t0)*RES/1000:.0f} kb), edges = 4DN boundary calls",
                  fontsize=10)
    axE.set_xlabel("position (Mb)")

    # F -- loop zoom, anchored on ChIA-PET evidence
    axF = fig.add_subplot(gs[5, 1])
    if chia:
        a, b, pet = chia[0]
        pad2 = 60
        b0, b1 = max(0, a - pad2), min(k, b + pad2)
        sub2 = M[b0:b1, b0:b1]
        p3 = sub2[np.isfinite(sub2) & (sub2 > 0)]
        axF.imshow(sub2, cmap="afmhot_r",
                   norm=LogNorm(vmin=np.percentile(p3, 1), vmax=np.percentile(p3, 99.7)),
                   extent=[mb[b0], mb[b1 - 1], mb[b1 - 1], mb[b0]], interpolation="none")
        axF.add_patch(Circle((mb[b], mb[a]), 0.022, fill=False, ec="#2ca02c", lw=2.2))
        axF.add_patch(Circle((mb[a], mb[b]), 0.022, fill=False, ec="#2ca02c", lw=2.2))
        oev = oe[a, b]
        near = [g for g, (s, e) in genes.items()
                if abs((s + e) / 2 - (lo_bin + a) * RES) < 60_000]
        tag = f"  anchor at {near[0]}" if near else ""
        axF.set_title(f"F  Loop — focal off-diagonal dot, O/E = {oev:.1f}{tag}\n"
                      f"{CHROM}:{(lo_bin+a)*RES:,} <-> {(lo_bin+b)*RES:,} "
                      f"(sep {(b-a)*RES/1000:.0f} kb), {pet} CTCF ChIA-PET PETs",
                      fontsize=10)
        axF.set_xlabel("position (Mb)")

    fig.suptitle("Phase 1 step 3 — pilot pipeline visual validation "
                 "(GM12878 in situ Hi-C, Rao et al. 2014, via 4DN, GRCh38)",
                 fontsize=13, y=0.995)
    out = FIGS / "phase1_validation.png"
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
