"""Phase 1 --- turn public Hi-C and hg38 into training shards.

Nothing here needs a GPU. The whole phase runs on a laptop for the pilot
chromosomes and on the cluster with --full.

The output of this module is, per (cell line, chromosome):

    bins.npz      bin table + per-bin N-fraction + usable mask
    pixels.npz    upper-triangle sparse contacts, detrended
    seq.npy       uint8 DNA codes, one row per bin
    qc.json       everything a reader needs to trust the above

Two rules shape the code:

  * No dense chromosome-scale matrix is ever materialised. chr1 at 5 kb would
    be 49,800 x 49,800 float32 -- about 10 GB for a single chromosome.
  * Detrending happens here, not in the model. Raw Hi-C is dominated by the
    fact that near things touch; a top-k over raw counts selects only near
    neighbours and the whole experiment becomes a test of genomic distance.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chromgraph.config import Config, data_dir

# A=0 C=1 G=2 T=3 N=4. uint8 keeps a 5 kb window in 5 kB instead of 20 kB of
# one-hot float; the model one-hots on the fly, on device.
BASE_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    BASE_CODE[ord(_b)] = _i
    BASE_CODE[ord(_b.lower())] = _i
N_CODE = 4


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def load_manifest(path: Path | None = None) -> dict:
    path = path or (Path(__file__).resolve().parents[1] / "data" / "manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expect_md5: str | None = None) -> Path:
    """Fetch to dest unless it is already there. Resumes are not attempted --
    a partial file is deleted and refetched, because a silently truncated
    .mcool produces a dataset that looks fine and is wrong."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expect_md5 is None or md5_of(dest) == expect_md5:
            print(f"    have  {dest.name}")
            return dest
        print(f"    md5 mismatch on {dest.name}, refetching")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"    fetch {dest.name}  <-  {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 22)

    if expect_md5 is not None:
        got = md5_of(tmp)
        if got != expect_md5:
            tmp.unlink()
            raise RuntimeError(f"md5 mismatch for {url}\n  expected {expect_md5}\n  got      {got}")
    tmp.rename(dest)
    return dest


# --------------------------------------------------------------------------- #
# genome
# --------------------------------------------------------------------------- #

def ensure_genome(manifest: dict, root: Path) -> Path:
    """Download and gunzip hg38 once, then index it with pyfaidx."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    fa = raw / "hg38.fa"
    if fa.exists():
        print(f"    have  {fa.name}")
    else:
        gz = download(manifest["assembly"]["fasta_url"], raw / "hg38.fa.gz")
        print("    gunzip hg38.fa.gz (about 3.1 GB uncompressed)")
        with gzip.open(gz, "rb") as src, fa.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)

    if not (fa.parent / (fa.name + ".fai")).exists():
        from pyfaidx import Faidx
        print("    indexing hg38.fa")
        Faidx(str(fa))
    return fa


def encode_windows(fa_path: Path, chrom: str, n_bins: int, bin_size: int) -> np.ndarray:
    """(n_bins, bin_size) uint8. Bases outside the chromosome are coded N."""
    from pyfaidx import Fasta

    fasta = Fasta(str(fa_path), sequence_always_upper=True, as_raw=True)
    if chrom not in fasta:
        raise KeyError(f"{chrom} not in {fa_path.name}; contigs look like "
                       f"{list(fasta.keys())[:3]}")
    length = len(fasta[chrom])
    out = np.full((n_bins, bin_size), N_CODE, dtype=np.uint8)
    for i in range(n_bins):
        start = i * bin_size
        stop = min(start + bin_size, length)
        if start >= length:
            break
        seq = str(fasta[chrom][start:stop])
        codes = BASE_CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        out[i, : stop - start] = codes
    return out


# --------------------------------------------------------------------------- #
# Hi-C
# --------------------------------------------------------------------------- #

@dataclass
class ContactBand:
    """Upper-triangle contacts within a genomic band, already detrended."""

    n_bins: int
    bin_size: int
    chrom: str
    row: np.ndarray          # int32
    col: np.ndarray          # int32
    raw: np.ndarray          # float32, balanced counts
    oe: np.ndarray           # float32, observed / expected-at-this-separation
    strength: np.ndarray     # float32 in (0, 1), oe / (1 + oe)
    expected: np.ndarray     # float32, expected value per separation s (index = s)
    coverage: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    valid: np.ndarray = field(default_factory=lambda: np.empty(0, bool))  # ICE-kept bins


def read_band(mcool_path: Path, chrom: str, bin_size: int, band_bp: int) -> ContactBand:
    """Read the near-diagonal band of one chromosome at one resolution.

    `band_bp` caps how far from the diagonal we look. Beyond a megabase or two
    the signal is compartment-scale and the pixel count explodes, so the band
    is both a memory guard and a modelling decision.
    """
    import cooler

    uri = f"{mcool_path}::resolutions/{bin_size}"
    clr = cooler.Cooler(uri)
    if chrom not in clr.chromnames:
        raise KeyError(f"{chrom} not in {mcool_path.name}: {clr.chromnames[:5]}")

    offset = clr.offset(chrom)
    n_bins = clr.extent(chrom)[1] - offset
    max_sep = band_bp // bin_size

    # Balanced values; cooler returns NaN where a bin was filtered out by ICE.
    selector = clr.matrix(balance=True, sparse=True, as_pixels=False)
    rows, cols, vals = [], [], []

    # Walk the chromosome in blocks so peak memory stays bounded by block size
    # rather than by chromosome length.
    block = max(2_000, max_sep * 4)
    for start in range(0, n_bins, block):
        stop = min(start + block + max_sep, n_bins)
        sub = selector[offset + start : offset + stop, offset + start : offset + stop]
        sub = sub.tocoo()
        keep = (sub.col > sub.row) & ((sub.col - sub.row) <= max_sep) & np.isfinite(sub.data)
        # Only emit pixels whose row falls in this block's own range, or blocks
        # would double-count their overlap region.
        keep &= (sub.row + start) < min(start + block, n_bins)
        rows.append(sub.row[keep] + start)
        cols.append(sub.col[keep] + start)
        vals.append(sub.data[keep])

    row = np.concatenate(rows).astype(np.int32) if rows else np.zeros(0, np.int32)
    col = np.concatenate(cols).astype(np.int32) if cols else np.zeros(0, np.int32)
    raw = np.concatenate(vals).astype(np.float32) if vals else np.zeros(0, np.float32)

    order = np.lexsort((col, row))
    row, col, raw = row[order], col[order], raw[order]

    # Bins ICE kept (finite weight). Every pair of them at separation s counts
    # toward expected[s], observed or not -- an unobserved pair is a zero.
    weight = clr.bins().fetch(chrom)["weight"].to_numpy()
    valid = np.isfinite(weight)

    oe, strength, expected = detrend(row, col, raw, max_sep, valid)
    return ContactBand(n_bins, bin_size, chrom, row, col, raw, oe, strength, expected,
                       valid=valid)


def oe_separation_rho(row: np.ndarray, col: np.ndarray, oe: np.ndarray,
                      valid: np.ndarray, max_sep: int,
                      sample: int = 1_000_000, seed: int = 0) -> dict:
    """Spearman(O/E, separation) over separations 1..max_sep only.

    `max_sep` should be the longest edge a training graph can hold
    (nodes_per_sample - 1). Beyond it the pixels never reach the model, and on
    sparse long-range data they distort the statistic: with zeros in
    `expected`, the mean O/E over *observed* pixels at s is 1/obs_frac(s),
    which rises as the map thins out -- a selection effect, not a residual
    distance trend.

    Two values, because neither alone is the whole picture:
      observed    -- over stored (nonzero) pixels. The gated number.
      with_zeros  -- unobserved ICE-valid pairs included as O/E = 0. Detrending
                     equalises the mean at each s, not the distribution shape,
                     so on sparse data this leans negative even when correct.
    """
    from scipy.stats import spearmanr

    sep = (col - row).astype(np.int64)
    keep = (sep >= 1) & (sep <= max_sep)
    s_obs, oe_obs = sep[keep], oe[keep].astype(np.float64)

    both = valid[row[keep]] & valid[col[keep]] if valid.size else np.ones(s_obs.size, bool)
    pairs = valid_pairs_by_separation(valid, max_sep) if valid.size else None
    if pairs is not None:
        seen = np.bincount(s_obs[both], minlength=max_sep + 1)[: max_sep + 1]
        zeros = np.maximum(pairs - seen, 0)
        zeros[0] = 0
        s_all = np.concatenate([s_obs[both], np.repeat(np.arange(max_sep + 1), zeros)])
        oe_all = np.concatenate([oe_obs[both], np.zeros(int(zeros.sum()))])
    else:
        s_all, oe_all = s_obs, oe_obs

    rng = np.random.default_rng(seed)

    def rho(x: np.ndarray, s: np.ndarray) -> float | None:
        if s.size < 3:
            return None
        if s.size > sample:
            pick = rng.choice(s.size, size=sample, replace=False)
            x, s = x[pick], s[pick]
        return float(spearmanr(x, s)[0])

    return {"observed": rho(oe_obs, s_obs), "with_zeros": rho(oe_all, s_all),
            "n_observed": int(s_obs.size), "n_with_zeros": int(s_all.size)}


def valid_pairs_by_separation(valid: np.ndarray, max_sep: int) -> np.ndarray:
    """n[s] = number of bin pairs (i, i+s) where both bins passed ICE."""
    n = np.zeros(max_sep + 1, dtype=np.int64)
    for s in range(1, min(max_sep, valid.size - 1) + 1):
        n[s] = np.count_nonzero(valid[:-s] & valid[s:])
    return n


def detrend(row: np.ndarray, col: np.ndarray, raw: np.ndarray,
            max_sep: int, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide out the distance-decay curve P(s).

    expected[s] = total balanced count at separation s / number of VALID bin
    pairs at s, counting pairs the matrix never observed as zeros. Averaging
    only the observed pixels instead overstates expected wherever the map is
    sparse -- long range, above ~500 kb -- which flattens P(s), deflates O/E
    there, and biases which distal edges top-k selects.

    Dividing by it leaves a value near 1 for "as much contact as is normal at
    this distance" and above 1 for a genuine enrichment. This single step is
    what stops the model from learning "near things touch", which it can
    already read off the input.
    """
    sep = (col - row).astype(np.int64)
    expected = np.ones(max_sep + 1, dtype=np.float32)
    if sep.size:
        total = np.bincount(sep, weights=raw.astype(np.float64), minlength=max_sep + 1)
        pairs = valid_pairs_by_separation(valid, max_sep)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(pairs > 0, total[: max_sep + 1] / np.maximum(pairs, 1), 1.0)
        # A separation with no valid pairs or no contact keeps expected = 1
        # rather than 0, so a later division can never produce inf.
        expected = np.where(mean > 0, mean, 1.0).astype(np.float32)

    oe = (raw / expected[sep]).astype(np.float32)
    strength = (oe / (1.0 + oe)).astype(np.float32)
    return oe, strength, expected


# --------------------------------------------------------------------------- #
# building a chromosome
# --------------------------------------------------------------------------- #

def build_chromosome(cfg: Config, manifest: dict, cell_line: str, chrom: str,
                     root: Path, band_bp: int, force: bool = False) -> dict:
    """Produce bins.npz / pixels.npz / seq.npy / qc.json for one chromosome."""
    bin_size = cfg.data.bin_size
    out = root / "processed" / cell_line / f"{chrom}_{bin_size}"
    out.mkdir(parents=True, exist_ok=True)
    qc_path = out / "qc.json"
    if qc_path.exists() and not force:
        print(f"  {cell_line} {chrom}: already built")
        return json.loads(qc_path.read_text(encoding="utf-8"))

    entry = manifest["cell_lines"][cell_line]["mcool"]
    mcool = root / "raw" / f"{entry['accession']}.mcool"
    if not mcool.exists():
        # A .mcool with no md5 in the manifest is fetched unverified, which is
        # the one case download()'s "a silently truncated .mcool produces a
        # dataset that looks fine and is wrong" guard cannot catch. Pass the
        # digest through when the manifest has one. Only reached when the file
        # is absent, so a present .mcool is never re-hashed.
        expect = entry.get("md5")
        try:
            download(entry["url"], mcool, expect_md5=expect)
        except Exception as primary:                       # noqa: BLE001
            if "mirror" not in entry:
                raise
            print(f"    portal fetch failed ({primary}); trying mirror")
            download(entry["mirror"], mcool, expect_md5=expect)

    print(f"  {cell_line} {chrom}: reading contacts")
    band = read_band(mcool, chrom, bin_size, band_bp)

    print(f"  {cell_line} {chrom}: extracting sequence")
    fa = ensure_genome(manifest, root)
    seq = encode_windows(fa, chrom, band.n_bins, bin_size)

    n_frac = (seq == N_CODE).mean(axis=1).astype(np.float32)
    coverage = np.bincount(band.row, weights=band.raw.astype(np.float64),
                           minlength=band.n_bins)
    coverage += np.bincount(band.col, weights=band.raw.astype(np.float64),
                            minlength=band.n_bins)
    usable = (n_frac <= cfg.data.max_n_frac) & (coverage > 0)

    np.savez_compressed(out / "bins.npz", n_frac=n_frac,
                        coverage=coverage.astype(np.float32), usable=usable)
    np.savez_compressed(out / "pixels.npz", row=band.row, col=band.col,
                        raw=band.raw, oe=band.oe, strength=band.strength,
                        expected=band.expected)
    np.save(out / "seq.npy", seq)

    qc = quality_report(cfg, cell_line, chrom, band, n_frac, usable, out)
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"  {cell_line} {chrom}: {band.n_bins} bins, {band.row.size} pixels, "
          f"{int(usable.sum())} usable")
    return qc


def quality_report(cfg: Config, cell_line: str, chrom: str, band: ContactBand,
                   n_frac: np.ndarray, usable: np.ndarray, out: Path) -> dict:
    """Numbers a reader needs before trusting the shards.

    The checksums matter for the laptop-to-cluster move: if the cluster's
    rebuild produces different digests, something environment-dependent crept
    into the pipeline and you find out here rather than after Phase 4.
    """
    sep = band.col - band.row
    ps_curve = {}
    for s in (1, 2, 5, 10, 20, 50, 100):
        if s < band.expected.size:
            ps_curve[str(s)] = float(band.expected[s])

    # Two checks that can actually fail. (Mean O/E over observed pixels is
    # ~1 by construction and caught nothing, so it is gone.)
    #  - P(s) log-log slope, 10 kb .. 1 Mb: mammalian Hi-C is roughly -0.75 to
    #    -1.2. Near 0 or positive means expected is wrong.
    #  - Spearman(O/E, separation): near 0 if the distance trend is removed;
    #    strongly negative if it survived detrending. See oe_separation_rho.
    s_lo = max(1, 10_000 // band.bin_size)
    s_hi = min(band.expected.size - 1, 1_000_000 // band.bin_size)
    s = np.arange(s_lo, s_hi + 1)
    ps_slope = (float(np.polyfit(np.log10(s), np.log10(band.expected[s]), 1)[0])
                if s.size >= 2 else None)
    # Spearman only over separations a training graph can hold: a sample is
    # nodes_per_sample contiguous bins, so no edge is longer than that minus 1.
    graph_max = min(cfg.data.nodes_per_sample - 1, band.expected.size - 1)
    rho = oe_separation_rho(band.row, band.col, band.oe, band.valid, graph_max)

    return {
        "cell_line": cell_line,
        "chrom": chrom,
        "bin_size": band.bin_size,
        "n_bins": int(band.n_bins),
        "n_pixels": int(band.row.size),
        "n_usable_bins": int(usable.sum()),
        "frac_usable": float(usable.mean()),
        "n_frac_mean": float(n_frac.mean()),
        "n_frac_over_threshold": int((n_frac > cfg.data.max_n_frac).sum()),
        "max_separation_bins": int(sep.max()) if sep.size else 0,
        "expected_by_separation": ps_curve,
        "ps_slope_10kb_1mb": ps_slope,
        "spearman_oe_vs_separation": rho["observed"],
        "spearman_oe_vs_separation_with_zeros": rho["with_zeros"],
        "spearman_separation_range_bins": [1, int(graph_max)],
        "strength_min": float(band.strength.min()) if band.strength.size else 0.0,
        "strength_max": float(band.strength.max()) if band.strength.size else 0.0,
        "checksums": {
            "bins.npz": md5_of(out / "bins.npz"),
            "pixels.npz": md5_of(out / "pixels.npz"),
            "seq.npy": md5_of(out / "seq.npy"),
        },
    }


def load_chromosome(root: Path, cell_line: str, chrom: str, bin_size: int) -> dict:
    """Read back what build_chromosome wrote."""
    d = root / "processed" / cell_line / f"{chrom}_{bin_size}"
    if not (d / "qc.json").exists():
        raise FileNotFoundError(
            f"{cell_line}/{chrom} not built. Run scripts/build_data.py first.")
    bins = np.load(d / "bins.npz")
    pixels = np.load(d / "pixels.npz")
    return {
        "dir": d,
        "seq": np.load(d / "seq.npy", mmap_mode="r"),
        "n_frac": bins["n_frac"],
        "coverage": bins["coverage"],
        "usable": bins["usable"],
        "row": pixels["row"],
        "col": pixels["col"],
        "strength": pixels["strength"],
        "oe": pixels["oe"],
        "qc": json.loads((d / "qc.json").read_text(encoding="utf-8")),
    }


def processed_root(cfg: Config | None = None) -> Path:
    return data_dir()
