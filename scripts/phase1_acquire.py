"""Phase 1, step 1 -- pilot dataset acquisition.

Downloads a single-chromosome pilot: GM12878 in situ Hi-C (Rao et al. 2014,
via 4D Nucleome) plus the matching GRCh38 sequence and GENCODE annotation.

Design notes
------------
* Accessions are resolved through the 4DN API at runtime; no S3 paths are
  hardcoded. The 4DN ``@@download`` endpoint returns 403 to unauthenticated
  clients, so we read ``open_data_url`` from the file record instead, which
  points at the public bucket and supports HTTP range requests.
* The multires cooler is 27 GB. We never download it. ``cooler`` is pointed at
  an fsspec HTTP file handle and we pull only a near-diagonal band, tiled along
  chr9. That is all the insulation / directionality features in
  architecture_spec.md 4.1 require; compartments are computed separately at
  coarse resolution.
* Everything is resumable and writes a manifest with sizes and checksums.

Run:  python scripts/phase1_acquire.py
      python scripts/phase1_acquire.py --dry-run
      python scripts/phase1_acquire.py --chrom chr10 --dry-run
      python scripts/phase1_acquire.py --chrom chr10
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
INTERIM = REPO / "data" / "interim"

FDN_API = "https://data.4dnucleome.org"

# --- pilot configuration -------------------------------------------------
# 4DNES3JX38V5 = in situ Hi-C on GM12878, MboI + bio-dATP, Rao SS et al. 2014
# (PMID:25497547), Lieberman Aiden lab. The densest GM12878 map in Rao 2014.
EXPSET = "4DNES3JX38V5"

FILES = {
    # accession    : (short name, what it is)
    "4DNFIXP4QG5B": ("mcool", "multires contact matrix (27 GB, never downloaded)"),
    "4DNFIVK5JOFU": ("boundaries_bed", "TAD boundary calls"),
    "4DNFIBMOGOZC": ("insulation_bw", "insulation score, diamond method"),
    "4DNFILYQ1PAY": ("compartments_bw", "A/B compartment eigenvector"),
}
# Files small enough to mirror locally. The mcool is deliberately excluded.
DOWNLOAD = ["4DNFIVK5JOFU", "4DNFIBMOGOZC", "4DNFILYQ1PAY"]

# docs/RESEARCH_PLAN_2026-08-26.md B3: a second cell line, same pipeline, same
# features. Accessions verified live against the 4DN API 2026-08-26 (not
# guessed): each experiment set's own `processed_files` list, filtered to the
# same four file_types FILES above uses. Both mcools confirmed to carry the
# 1,000/2,000/5,000 bp resolutions this pipeline needs.
# CAVEAT, disclosed, not hidden: GM12878's set is 72 experiments merged;
# K562 is 6, IMR90 is 7 -- both companion maps are markedly shallower than the
# pilot's GM12878 map. Confirm this doesn't starve the balancing weights
# (the usable-bin fraction) before treating results as comparable.
CELL_LINES = {
    "GM12878": {
        "expset": "4DNES3JX38V5",
        "description": "in situ Hi-C on gm12878 with MboI and bio-dATP",
        "files": {
            "4DNFIXP4QG5B": ("mcool", "multires contact matrix"),
            "4DNFIVK5JOFU": ("boundaries_bed", "TAD boundary calls"),
            "4DNFIBMOGOZC": ("insulation_bw", "insulation score, diamond method"),
            "4DNFILYQ1PAY": ("compartments_bw", "A/B compartment eigenvector"),
        },
    },
    "K562": {
        "expset": "4DNESI7DEJTM",
        "description": ("in situ Hi-C on K562 with MboI and bio-dATP "
                         "(higher crosslinker concentration); 6 experiments, "
                         "vs GM12878's 72 -- shallower map"),
        "files": {
            "4DNFI18UHVRO": ("mcool", "multires contact matrix"),
            "4DNFI4EFYN3Q": ("boundaries_bed", "TAD boundary calls"),
            "4DNFIXU7QLG6": ("insulation_bw", "insulation score, diamond method"),
            "4DNFIWUAO2QI": ("compartments_bw", "A/B compartment eigenvector"),
        },
    },
    "IMR90": {
        "expset": "4DNES1ZEJNRU",
        "description": ("in situ Hi-C on IMR90 with MboI and bio-dATP; "
                         "7 experiments, vs GM12878's 72 -- shallower map"),
        "files": {
            "4DNFIJTOIGOI": ("mcool", "multires contact matrix"),
            "4DNFIMNT2VYL": ("boundaries_bed", "TAD boundary calls"),
            "4DNFIZFI8U3R": ("insulation_bw", "insulation score, diamond method"),
            "4DNFIHM89EGL": ("compartments_bw", "A/B compartment eigenvector"),
        },
    },
}

# Rebound by --chrom / --resolution / --cell-line in main(). chr9 stays the
# default and, in the multi-chromosome build, stays the TEST split: CHROME
# holds chr9 out as test and matching that keeps the two comparable.
CHROM = "chr9"
RESOLUTION = 5_000      # matches CHROME's 5 kb bulk Hi-C
BAND_BP = 2_000_000     # +/- 2 Mb around the diagonal
TILE_BP = 10_000_000    # tile size for the banded fetch
COARSE_RESOLUTION = 250_000   # for compartment calling

# Ensembl uses '9', GENCODE and 4DN use 'chr9'. We normalise Ensembl -> chr9
# on write; this is recorded in the manifest and must appear in the data card.
def fasta_url(chrom: str) -> str:
    """Ensembl per-chromosome FASTA. Ensembl names it '9', 4DN/GENCODE 'chr9';
    the header is normalised to the 'chr' form on write (see fetch_fasta), and
    coordinates are unchanged because both are GRCh38."""
    return ("https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/"
            f"Homo_sapiens.GRCh38.dna.chromosome.{chrom.removeprefix('chr')}"
            ".fa.gz")


FASTA_URL = fasta_url(CHROM)
GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_47/gencode.v47.annotation.gtf.gz"
)


@dataclass
class Manifest:
    created: str = ""
    config: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)

    def add(self, **kw) -> None:
        self.artifacts.append(kw)
        print(f"    recorded: {kw.get('path')}")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        print(f"\nmanifest -> {path.relative_to(REPO)}")


def md5(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fdn_record(accession: str) -> dict:
    r = requests.get(
        f"{FDN_API}/files-processed/{accession}/",
        headers={"Accept": "application/json"},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()


def stream_download(url: str, dest: Path, expect_md5: str | None = None) -> Path:
    """Download with resume. Skips if a complete file is already present.

    Locked with flock on a sibling .lock file: some destinations (the
    genome-wide GTF, in particular) are SHARED across chromosomes in the
    multi-chromosome build, and multiple phase1_acquire.py --chrom invocations
    now run concurrently. Without the lock, concurrent writers to the same
    dest corrupt each other's .part file (observed 2026-08-26: chr13 got
    "zlib.error: invalid block type" mid-decompress, chr14 got
    FileNotFoundError on tmp.replace(dest) because a sibling process's rename
    had already consumed the shared .part). The lock serialises all access to
    one dest path; a process that loses the race for the lock finds dest
    already complete when it wakes up and skips its own download.
    """
    import fcntl
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.parent / (dest.name + ".lock")
    with lock_path.open("a+") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            if dest.exists() and expect_md5 and md5(dest) == expect_md5:
                print(f"    cached (md5 ok): {dest.name}")
                return dest
            if dest.exists() and not expect_md5:
                # No checksum to verify against (e.g. the GTF), but a
                # complete file is already at this path -- possibly written
                # by a concurrent chromosome's invocation while we waited on
                # the lock. Trust it rather than re-fetching.
                print(f"    cached (present, no checksum available): {dest.name}")
                return dest

            tmp = dest.with_suffix(dest.suffix + ".part")
            pos = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, headers=headers, stream=True, timeout=180) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for {url}")
                mode = "ab" if r.status_code == 206 and pos else "wb"
                with tmp.open(mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            if expect_md5:
                got = md5(dest)
                status = "OK" if got == expect_md5 else f"MISMATCH (expected {expect_md5})"
                print(f"    md5 {got} {status}")
            return dest
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def fetch_hic_band(mcool_url: str) -> dict:
    """Pull a near-diagonal band of the chr9 contact matrix at RESOLUTION.

    Tiles along the diagonal with TILE_BP windows overlapping by BAND_BP so no
    band entry is dropped at a tile seam. Returns COO arrays plus the bin table.
    """
    import cooler
    import fsspec
    import h5py
    from scipy import sparse

    print(f"  opening remote mcool (range reads, no full download)")
    fh = fsspec.open(mcool_url, mode="rb", block_size=4 << 20).open()
    h5 = h5py.File(fh, "r")
    clr = cooler.Cooler(h5[f"resolutions/{RESOLUTION}"])

    lo, hi = clr.extent(CHROM)
    n_bins = hi - lo
    chrom_len = int(clr.chromsizes[CHROM])
    band_bins = BAND_BP // RESOLUTION
    print(f"  {CHROM}: {chrom_len:,} bp -> {n_bins:,} bins @ {RESOLUTION:,} bp")
    print(f"  band: +/-{BAND_BP:,} bp ({band_bins} bins)")

    rows, cols, vals = [], [], []
    starts = range(0, chrom_len, TILE_BP - BAND_BP)
    t0 = time.time()
    for k, s in enumerate(starts, 1):
        e = min(s + TILE_BP, chrom_len)
        if e <= s:
            break
        region = f"{CHROM}:{s}-{e}"
        m = clr.matrix(balance=True, sparse=True).fetch(region).tocoo()
        off = s // RESOLUTION
        keep = np.abs(m.row - m.col) <= band_bins
        keep &= np.isfinite(m.data)
        keep &= (m.row + off) <= (m.col + off)      # upper triangle only
        rows.append(m.row[keep] + off)
        cols.append(m.col[keep] + off)
        vals.append(m.data[keep])
        print(f"    tile {k:>2}/{len(starts)} {region:<28} "
              f"kept {keep.sum():>9,}  [{time.time()-t0:6.1f}s]")
        if e >= chrom_len:
            break

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals).astype(np.float32)

    # tiles overlap by BAND_BP, so identical (row, col) pairs recur -- dedupe
    key = row.astype(np.int64) * n_bins + col.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    row, col, val = row[uniq], col[uniq], val[uniq]
    print(f"  band assembled: {len(val):,} unique upper-triangle entries "
          f"in {time.time()-t0:.1f}s")

    bins = clr.bins()[lo:hi]
    weight = (bins["weight"].to_numpy(dtype=np.float32)
              if "weight" in bins.columns else np.full(n_bins, np.nan, np.float32))

    return {
        "row": row.astype(np.int32),
        "col": col.astype(np.int32),
        "val": val,
        "n_bins": np.int64(n_bins),
        "resolution": np.int64(RESOLUTION),
        "band_bp": np.int64(BAND_BP),
        "chrom_len": np.int64(chrom_len),
        "bin_start": bins["start"].to_numpy(dtype=np.int64),
        "bin_end": bins["end"].to_numpy(dtype=np.int64),
        "weight": weight,
    }


def fetch_coarse_matrix(mcool_url: str) -> dict:
    """Dense chr9 matrix at COARSE_RESOLUTION, for compartment calling."""
    import cooler
    import fsspec
    import h5py

    fh = fsspec.open(mcool_url, mode="rb", block_size=4 << 20).open()
    h5 = h5py.File(fh, "r")
    clr = cooler.Cooler(h5[f"resolutions/{COARSE_RESOLUTION}"])
    t0 = time.time()
    m = clr.matrix(balance=True).fetch(CHROM).astype(np.float32)
    print(f"  coarse {COARSE_RESOLUTION//1000} kb matrix {m.shape} "
          f"in {time.time()-t0:.1f}s")
    return {"matrix": m, "resolution": np.int64(COARSE_RESOLUTION)}


def fetch_sequence(dest: Path) -> tuple[Path, int]:
    """Ensembl per-chromosome FASTA -> plain .fa, header normalised to 'chrN'."""
    gz = RAW / f"Homo_sapiens.GRCh38.dna.chromosome.{CHROM.removeprefix('chr')}.fa.gz"
    stream_download(FASTA_URL, gz)
    n = 0
    with gzip.open(gz, "rt") as src, dest.open("w", encoding="utf-8") as out:
        for line in src:
            if line.startswith(">"):
                out.write(f">{CHROM}\n")     # Ensembl '9' -> 'chr9'
            else:
                out.write(line)
                n += len(line.strip())
    print(f"    {dest.name}: {n:,} bp (header normalised to '{CHROM}')")
    return dest, n


def fetch_annotation(dest: Path) -> tuple[Path, int]:
    """GENCODE GTF, stream-filtered to CHROM. Already uses 'chr' naming."""
    gz = RAW / "gencode.v47.annotation.gtf.gz"
    stream_download(GTF_URL, gz)
    kept = 0
    with gzip.open(gz, "rt") as src, dest.open("w", encoding="utf-8") as out:
        for line in src:
            if line.startswith("#"):
                out.write(line)
            elif line.split("\t", 1)[0] == CHROM:
                out.write(line)
                kept += 1
    print(f"    {dest.name}: {kept:,} feature rows on {CHROM}")
    return dest, kept


def main() -> None:
    global CHROM, FASTA_URL, RESOLUTION, EXPSET, FILES, DOWNLOAD
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve URLs and print the plan, download nothing")
    ap.add_argument("--chrom", default=CHROM,
                    help="chromosome to fetch, e.g. chr9, chr10 (default: chr9). "
                         "chr9 stays the TEST split in the multi-chromosome build "
                         "(CHROME holds chr9 out as test too); every other "
                         "chromosome fetched here is a train/val candidate.")
    ap.add_argument("--resolution", type=int, default=RESOLUTION,
                    help=f"Hi-C band resolution in bp (default {RESOLUTION}). "
                         "docs/WORKPLAN_2026-08-25.md T1 / RESEARCH_PLAN B2: "
                         "confirm the resolution exists in the mcool's "
                         "resolutions/ group first (T1.1) before using this "
                         "for anything beyond a small tile.")
    ap.add_argument("--cell-line", default="GM12878", choices=sorted(CELL_LINES),
                    help="docs/RESEARCH_PLAN_2026-08-26.md B3: second cell "
                         "line. See CELL_LINES above for accessions and the "
                         "sequencing-depth caveat for K562/IMR90.")
    args = ap.parse_args()

    CHROM = args.chrom
    RESOLUTION = args.resolution
    FASTA_URL = fasta_url(CHROM)
    cl = CELL_LINES[args.cell_line]
    EXPSET = cl["expset"]
    FILES = cl["files"]
    DOWNLOAD = [acc for acc, (short, _) in FILES.items() if short != "mcool"]
    mcool_acc = next(acc for acc, (short, _) in FILES.items() if short == "mcool")

    RAW.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    man = Manifest(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config={
            "cell_line": args.cell_line, "experiment_set": EXPSET,
            "experiment_set_description": cl["description"],
            "chrom": CHROM,
            "resolution_bp": RESOLUTION,
            "band_bp": BAND_BP,
            "coarse_resolution_bp": COARSE_RESOLUTION,
            "genome_assembly": "GRCh38",
            "fasta_source": FASTA_URL,
            "gtf_source": GTF_URL,
            "chrom_naming_note":
                "Ensembl FASTA uses '9'; normalised to 'chr9' to match "
                "GENCODE and 4DN. Coordinates are unchanged (both GRCh38).",
        },
    )

    print(f"\n=== resolving 4DN accessions (set {EXPSET}, cell line "
          f"{args.cell_line}) ===")
    urls: dict[str, dict] = {}
    for acc, (short, desc) in FILES.items():
        rec = fdn_record(acc)
        urls[acc] = {
            "short": short,
            "desc": desc,
            "url": rec.get("open_data_url"),
            "size": rec.get("file_size"),
            "md5": rec.get("md5sum"),
            "assembly": rec.get("genome_assembly"),
        }
        size_gb = (rec.get("file_size") or 0) / 1e9
        print(f"  {acc}  {short:<16} {size_gb:9.3f} GB  {rec.get('genome_assembly')}")
        if not urls[acc]["url"]:
            print(f"    WARNING: no open_data_url for {acc}")

    if args.dry_run:
        print("\n--dry-run: stopping before any download")
        for acc, u in urls.items():
            print(f"  {acc}: {u['url']}")
        return

    print("\n=== small auxiliary files ===")
    for acc in DOWNLOAD:
        u = urls[acc]
        suffix = {"boundaries_bed": ".bed.gz",
                  "insulation_bw": ".bw",
                  "compartments_bw": ".bw"}[u["short"]]
        dest = RAW / f"{acc}_{u['short']}{suffix}"
        print(f"  {u['short']}")
        stream_download(u["url"], dest, expect_md5=u["md5"])
        man.add(path=str(dest.relative_to(REPO)), source=u["url"],
                accession=acc, role=u["desc"],
                bytes=dest.stat().st_size, md5=md5(dest))

    # Cell-line and resolution namespacing: GM12878 @ 5,000 bp keeps the
    # original bare filenames (every existing reference in this project
    # assumes them); anything else gets an explicit suffix so it cannot
    # silently collide with or shadow the pilot's files.
    cl_tag = "" if args.cell_line == "GM12878" else f"_{args.cell_line}"
    res_tag = "" if RESOLUTION == 5_000 else f"_{RESOLUTION}bp"

    print(f"\n=== Hi-C band, {CHROM} @ {RESOLUTION//1000} kb, {args.cell_line} ===")
    band = fetch_hic_band(urls[mcool_acc]["url"])
    band_path = INTERIM / f"hic_band_{CHROM}_{RESOLUTION}bp{cl_tag}.npz"
    np.savez_compressed(band_path, **band)
    man.add(path=str(band_path.relative_to(REPO)),
            source=urls[mcool_acc]["url"], accession=mcool_acc,
            role=f"near-diagonal band, +/-{BAND_BP} bp, balanced, upper triangle",
            entries=int(len(band["val"])), n_bins=int(band["n_bins"]),
            bytes=band_path.stat().st_size, md5=md5(band_path))

    print(f"\n=== Hi-C coarse matrix, {CHROM} @ {COARSE_RESOLUTION//1000} kb, "
          f"{args.cell_line} ===")
    coarse = fetch_coarse_matrix(urls[mcool_acc]["url"])
    coarse_path = INTERIM / f"hic_coarse_{CHROM}_{COARSE_RESOLUTION}bp{cl_tag}.npz"
    np.savez_compressed(coarse_path, **coarse)
    man.add(path=str(coarse_path.relative_to(REPO)),
            source=urls[mcool_acc]["url"], accession=mcool_acc,
            role="coarse balanced matrix for compartment calling",
            shape=list(coarse["matrix"].shape),
            bytes=coarse_path.stat().st_size, md5=md5(coarse_path))

    # Sequence and annotation are genome, not cell-line, data -- shared and
    # reused across cell lines exactly as they already are across
    # chromosomes' worth of a single cell line.
    print("\n=== reference sequence ===")
    fa = INTERIM / f"{CHROM}.fa"
    _, n_bp = fetch_sequence(fa)
    man.add(path=str(fa.relative_to(REPO)), source=FASTA_URL,
            role="GRCh38 sequence, header normalised (shared across cell lines)",
            bases=n_bp, bytes=fa.stat().st_size, md5=md5(fa))

    print("\n=== annotation ===")
    gtf = INTERIM / f"gencode.v47.{CHROM}.gtf"
    _, n_rows = fetch_annotation(gtf)
    man.add(path=str(gtf.relative_to(REPO)), source=GTF_URL,
            role="GENCODE v47 annotation filtered to chromosome (shared "
                 "across cell lines)",
            rows=n_rows, bytes=gtf.stat().st_size, md5=md5(gtf))

    manifest_name = ("pilot_manifest.json"
                      if CHROM == "chr9" and not cl_tag and not res_tag
                      else f"pilot_manifest_{CHROM}{cl_tag}{res_tag}.json")
    man.write(REPO / "data" / manifest_name)
    print("\nPhase 1 step 1 complete. Next: step 2 (build phi features), "
          "step 3 (visual validation -- you must eyeball a TAD and a loop).")


if __name__ == "__main__":
    sys.exit(main())
