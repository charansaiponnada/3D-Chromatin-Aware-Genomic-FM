#!/usr/bin/env python
"""C1 -- Genomic Benchmarks, the project's first EXTERNAL number.

docs/RESEARCH_PLAN_2026-08-26.md C1: "Wire ONE standard benchmark so the model
has a comparable number ... One benchmark number is worth more than a fifth
internal diagnostic." Genomic Benchmarks (Gresova et al. 2023, 8 classification
tasks) is chosen because Caduceus and HyenaDNA both report on it, and Phase F's
Caduceus comparison invites benchmark comparison the moment it is made.

WHAT THIS MEASURES, AND WHAT IT CANNOT BE COMPARED TO
-----------------------------------------------------
FROZEN-EMBEDDING LINEAR PROBE. The pretrained encoder is frozen, sequences are
mean-pooled over positions, and a logistic regression is fit on top. The
published Caduceus and HyenaDNA numbers on these tasks are FINE-TUNED end to
end. These are different protocols and the numbers are NOT interchangeable: a
frozen probe is a lower bound on what the same weights would reach fine-tuned.
Any table that puts them side by side must say so in the caption. What this
number legitimately supports is (a) a floor, and (b) the BASELINE-vs-STRUCTURAL
contrast, which is internally matched because both arms get the identical
protocol.

THE COMPOSITION FLOOR IS NOT OPTIONAL. Probe B's lesson (CLAUDE.md 4) was that
a model can look structurally sensitive while losing to local GC content. Every
task here is therefore also fit on k-mer frequencies alone (k = 1..3, 84
features). A model that does not clear its own composition floor has not earned
the word "representation", and on short genomic fragments k-mers are strong.

phi AT DOWNSTREAM TIME. These sequences have no Hi-C: they are 200-4,776 bp
fragments, and two tasks are not even human. The structural arm is therefore
run under the S0 convention (phi := 0, phi_valid all true), exactly as in
phase4_p1_swap.py. This is a REAL limitation of structure-conditioned
pretraining, not a shortcut -- at transfer time the structural channel is
unavailable, so the structural arm can only benefit through what conditioning
put into its sequence weights. That is precisely the claim worth testing.

CULLER SAFETY (CLAUDE.md 3). Parquet files, embeddings and results are each
cached to disk per (dataset, split) and per (run, dataset, split). A kill costs
one item, never the sweep. Re-running skips what is already on disk.

Run:
  ./3d-gen/bin/python -u scripts/c1_genomic_benchmarks.py --runs w32768
  ./3d-gen/bin/python -u scripts/c1_genomic_benchmarks.py --datasets human_enhancers_cohn
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from chromfm.model import BiMambaLM, ModelConfig, use_scan     # noqa: E402
from phase1_dataset import VOCAB, PAD                          # noqa: E402

CACHE = REPO / "data" / "interim" / "c1"
EMB = REPO / "results" / "c1" / "emb"
OUT = REPO / "results" / "c1_genomic_benchmarks.json"
HF = ("https://huggingface.co/api/datasets/katarinagresova/"
      "Genomic_Benchmarks_{ds}/parquet/default/{split}/0.parquet")

DATASETS = [
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "dummy_mouse_enhancers_ensembl",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
]
D_STRUCT_RAW = 8
# phi symmetry under reverse-complement (CLAUDE.md 4). Irrelevant while phi is
# zero, but passed so the code path is the same one training used.
SYMMETRY = torch.tensor([1., 1., 1., -1., 1., -1., 1., 1.])
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
VAL_FRAC = 0.1
PROBE_SEED = 20260831
MAX_LEN = 32_768          # the pilot checkpoints' training window


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def fetch(ds: str, split: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{ds}_{split}.parquet"
    if not f.exists():
        df = pd.read_parquet(HF.format(ds=ds, split=split))
        df.to_parquet(f)
    return pd.read_parquet(f)


def tokenize(seqs, max_len: int):
    """ACGT -> the model's own vocab. Anything else becomes N, as in training."""
    n = len(seqs)
    L = min(max(len(s) for s in seqs), max_len)
    out = np.full((n, L), PAD, dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s.upper()[:L]
        a = np.frombuffer(s.encode(), dtype="S1")
        t = np.full(len(a), VOCAB["N"], dtype=np.int64)
        for base, tok in VOCAB.items():
            t[a == base.encode()] = tok
        out[i, :len(t)] = t
        lens[i] = len(t)
    return out, lens


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

def kmer_features(seqs, k_max: int = 3):
    """Composition floor: k-mer frequencies, k = 1..3 (4+16+64 = 84 columns).

    Vectorised per sequence -- the naive triple loop is ~150M Python iterations
    on the 100k-sequence tasks and does not finish in reasonable time.
    Windows containing a non-ACGT base are dropped, not folded into a bucket.
    """
    lut = np.full(256, -1, dtype=np.int64)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i
    feats = np.zeros((len(seqs), sum(4 ** k for k in range(1, k_max + 1))),
                     dtype=np.float64)
    for r, s in enumerate(seqs):
        a = lut[np.frombuffer(s.upper().encode(), dtype=np.uint8)]
        col = 0
        for k in range(1, k_max + 1):
            size = 4 ** k
            m = len(a) - k + 1
            if m > 0:
                code = np.zeros(m, dtype=np.int64)
                ok = np.ones(m, dtype=bool)
                for j in range(k):
                    w = a[j:j + m]
                    code = code * 4 + np.maximum(w, 0)
                    ok &= w >= 0
                if ok.any():
                    v = np.bincount(code[ok], minlength=size).astype(np.float64)
                    feats[r, col:col + size] = v / ok.sum()
            col += size
    return feats


def logistic_probe(Xtr, ytr, Xte, yte, alphas=ALPHAS, seed=PROBE_SEED):
    """Multinomial logistic regression, L2 chosen on a held-out slice of train.

    Full-batch LBFGS on standardised features. No sklearn in this environment,
    and the project already fits its probes directly (phase5_positional_probe).
    """
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    n = len(Xtr)
    perm = rng.permutation(n)
    nval = max(int(VAL_FRAC * n), 1)
    vi, ti = perm[:nval], perm[nval:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ncls = int(max(ytr.max(), yte.max())) + 1

    def fit(X, y, alpha):
        X = torch.tensor(X, dtype=torch.float32, device=dev)
        y = torch.tensor(y, dtype=torch.long, device=dev)
        W = torch.zeros(X.shape[1], ncls, device=dev, requires_grad=True)
        b = torch.zeros(ncls, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200,
                                tolerance_grad=1e-7, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(X @ W + b, y)
            loss = loss + alpha * (W * W).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return W.detach(), b.detach()

    def acc(W, b, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=dev)
        pred = (X @ W + b).argmax(1).cpu().numpy()
        return float((pred == y).mean()), pred

    best = (-1.0, None)
    for a in alphas:
        W, b = fit(Xtr[ti], ytr[ti], a)
        va, _ = acc(W, b, Xtr[vi], ytr[vi])
        if va > best[0]:
            best = (va, a)
    W, b = fit(Xtr, ytr, best[1])
    a_te, pred = acc(W, b, Xte, yte)

    # Matthews correlation, the metric that survives class imbalance.
    if ncls == 2:
        tp = float(((pred == 1) & (yte == 1)).sum())
        tn = float(((pred == 0) & (yte == 0)).sum())
        fp = float(((pred == 1) & (yte == 0)).sum())
        fn = float(((pred == 0) & (yte == 1)).sum())
        den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn - fp*fn) / den if den > 0 else 0.0
    else:
        mcc = None
    return {"test_accuracy": a_te, "test_mcc": mcc,
            "alpha": best[1], "val_accuracy": best[0],
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "n_features": int(Xtr.shape[1]), "n_classes": ncls}


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

@torch.no_grad()
def embed(model, tokens, lens, device, structural, batch=8):
    """Mean-pooled final hidden states over real (non-PAD) positions."""
    out = np.zeros((len(tokens), model.c.d_model), dtype=np.float32)
    sym = SYMMETRY.to(device)
    for i in range(0, len(tokens), batch):
        tk = torch.tensor(tokens[i:i+batch], device=device)
        kw = {}
        if structural:
            # S0: phi present in shape, structurally silent. These sequences
            # have no Hi-C and two tasks are not human.
            kw = {"phi": torch.zeros(tk.shape[0], tk.shape[1], D_STRUCT_RAW,
                                     device=device),
                  "phi_valid": torch.ones(tk.shape, dtype=torch.bool,
                                          device=device),
                  "symmetry": sym}
        h = model.encode(tk, **kw)
        m = torch.zeros(tk.shape, device=device)
        for j, L in enumerate(lens[i:i+batch]):
            m[j, :L] = 1.0
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        out[i:i+batch] = pooled.float().cpu().numpy()
    return out


def load_model(ckpt: Path, structural: bool, device):
    model = BiMambaLM(ModelConfig(structural=structural)).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("step")


def discover_runs(spec: str):
    """spec 'w32768' -> the six pilot checkpoints, arms labelled."""
    runs = []
    for d in sorted((REPO / "results" / "baselines" / spec).glob("baseline_v2_seed*")):
        if (d / "checkpoint.pt").exists():
            runs.append((d.name, d / "checkpoint.pt", False))
    for d in sorted((REPO / "results" / "novel_model" / spec).glob("structural_seed*")):
        if (d / "checkpoint.pt").exists():
            runs.append((d.name, d / "checkpoint.pt", True))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="w32768")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-model", action="store_true",
                    help="composition floor only, no GPU")
    a = ap.parse_args()

    EMB.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        use_scan("triton")
    runs = [] if a.skip_model else discover_runs(a.runs)

    print("=" * 78)
    print("C1 -- Genomic Benchmarks, FROZEN-EMBEDDING LINEAR PROBE")
    print("=" * 78)
    print("  NOT comparable to published Caduceus/HyenaDNA numbers: those are")
    print("  fine-tuned end to end. This is a frozen probe, i.e. a lower bound.")
    print(f"  device {device}, runs {[r[0] for r in runs]}")
    print()

    results = {"protocol": "frozen encoder, mean-pool, logistic probe",
               "comparable_to_finetuned_published_numbers": False,
               "phi_at_downstream": "S0 (zeros) -- these sequences have no Hi-C",
               "runs": {}, "composition_floor": {}}
    if OUT.exists():
        results = json.loads(OUT.read_text())

    def save():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2) + "\n")

    for ds in a.datasets:
        print(f"--- {ds}")
        t0 = time.time()
        tr, te = fetch(ds, "train"), fetch(ds, "test")
        ytr = tr["label"].to_numpy()
        yte = te["label"].to_numpy()
        print(f"    train {len(tr):,}  test {len(te):,}  classes {len(set(ytr))}  "
              f"len {tr['seq'].str.len().min()}-{tr['seq'].str.len().max()}  "
              f"majority {max(np.bincount(ytr))/len(ytr):.4f}")

        if ds not in results["composition_floor"]:
            kf = CACHE / f"{ds}_kmer.npz"
            if kf.exists():
                z = np.load(kf)
                Ktr, Kte = z["tr"], z["te"]
            else:
                Ktr, Kte = kmer_features(tr["seq"]), kmer_features(te["seq"])
                np.savez_compressed(kf, tr=Ktr, te=Kte)
            r = logistic_probe(Ktr, ytr, Kte, yte)
            r["majority_class"] = float(max(np.bincount(ytr)) / len(ytr))
            results["composition_floor"][ds] = r
            save()
        cf = results["composition_floor"][ds]
        print(f"    k-mer floor (k<=3): acc {cf['test_accuracy']:.4f}  "
              f"majority {cf['majority_class']:.4f}")

        for name, ckpt, structural in runs:
            results["runs"].setdefault(name, {"structural": structural,
                                              "checkpoint": str(ckpt), "tasks": {}})
            if ds in results["runs"][name]["tasks"]:
                r = results["runs"][name]["tasks"][ds]
                print(f"    {name:24s} acc {r['test_accuracy']:.4f}  (cached)")
                continue
            ef = EMB / f"{name}__{ds}.npz"
            if ef.exists():
                z = np.load(ef)
                Etr, Ete = z["tr"], z["te"]
            else:
                model, step = load_model(ckpt, structural, device)
                Xtr, Ltr = tokenize(tr["seq"].tolist(), a.max_len)
                Xte, Lte = tokenize(te["seq"].tolist(), a.max_len)
                Etr = embed(model, Xtr, Ltr, device, structural, a.batch)
                Ete = embed(model, Xte, Lte, device, structural, a.batch)
                np.savez_compressed(ef, tr=Etr, te=Ete)
                del model
                torch.cuda.empty_cache()
            r = logistic_probe(Etr, ytr, Ete, yte)
            results["runs"][name]["tasks"][ds] = r
            save()
            print(f"    {name:24s} acc {r['test_accuracy']:.4f}  "
                  f"mcc {r['test_mcc'] if r['test_mcc'] is None else round(r['test_mcc'],4)}  "
                  f"vs floor {r['test_accuracy']-cf['test_accuracy']:+.4f}")
        print(f"    ({time.time()-t0:.0f}s)")

    save()
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
