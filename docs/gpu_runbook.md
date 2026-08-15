# GPU runbook — setting up on the L40S box

Written for a fresh clone on a Linux machine with 2× L40S. Follow it in order; step 3 must pass before step 4 is meaningful.

**The most important thing to know before you start:** `data/` is deliberately not in the repository. Cloning gives you the code and the manifest, not the 348 MB of pilot data. You rebuild it on the box in about ten minutes (step 2). That is by design — the manifest carries every accession, URL and checksum, so the data is reproducible rather than committed.

---

## 0. Clone and environment

```bash
git clone https://github.com/charansaiponnada/3D-Chromatin-Aware-Genomic-FM.git
cd 3D-Chromatin-Aware-Genomic-FM
python -m venv .venv && source .venv/bin/activate
```

Install PyTorch with the CUDA build matching the box's driver first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Confirm the GPUs and Triton are visible:

```bash
python -c "import torch, triton; print(torch.cuda.device_count(), torch.cuda.get_device_name(0), triton.__version__)"
```

Expect `2 NVIDIA L40S <version>`. If Triton is missing, the custom scan cannot run and step 3 will tell you so.

> Three packages fail to build on Windows and were worked around during development: `hic-straw` and `pyBigWig` need libcurl headers, and `cooltools` fails to link against a POSIX symbol. **All three install cleanly on Linux.** Nothing in the pipeline requires them — `pybigtools` covers the bigWig reads — but if you want `cooltools` for its calibrated boundary caller, the GPU box is where it will work.

## 1. Verify the code before touching data

```bash
python scripts/test_model.py
```

Expect `16/16 checks passed`. This runs on CPU at small sequence length and confirms parameter counts, that the structural pathway changes the output and receives gradient, and that the reverse-complement sign handling is correct.

## 2. Rebuild the pilot data

Three scripts, in order. Roughly ten minutes total, mostly the banded Hi-C fetch.

```bash
python scripts/phase1_acquire.py        # ~7 min, ~350 MB, reads the 27 GB cooler remotely
python scripts/phase1_features.py       # ~35 s, builds phi and validates against 4DN's tracks
python scripts/phase1_dataset.py        # ~1 min, tokenises and builds the window index
```

Checks that the run went right:

- `phase1_acquire.py` prints `md5 ... OK` for each auxiliary file and assembles **7,108,483** band entries.
- `phase1_features.py` reports insulation **r = +0.9969** and compartments **r = +0.9759** against 4D Nucleome's independently derived tracks. If either drops materially, stop — the pipeline is wrong, not the model.
- `phase1_dataset.py` reports **5,422 / 273 / 242** train/val/test windows.

Optional, regenerates the validation figure:

```bash
python scripts/phase1_validate_visual.py
```

## 3. Validate the custom kernel — do this before any training

```bash
python scripts/validate_kernel.py
```

**The Triton selective scan in `src/chromfm/scan_triton.py` has never been executed.** It was written on a CPU-only Windows machine with no CUDA device and no Triton install. The recurrence and its adjoint were derived by hand and the code is syntactically complete, but nothing is numerically confirmed.

The script compares the kernel against the reference implementation on forward values and every input gradient, with and without the permeability term, at single-chunk, multi-chunk and ragged-final-chunk lengths. It also asserts the gradient with respect to `p` is non-zero, because a kernel that silently ignored `p` would pass every other check.

Exit codes: `0` validated, `1` mismatch, `2` no GPU or no Triton.

**If it fails, do not train.** Fall back to `use_scan("ref")` for a small-scale smoke test, or drop the permeability term and use the stock `mamba_ssm` kernel, which needs no custom code because the Δ bias rides along in the `delta` argument. A scan with a subtly wrong gradient produces a smooth, believable loss curve while training the wrong model, and every number downstream of it is void.

Record the GPU, driver and Triton version alongside the result — they belong in the run config.

## 4. Phase 3 — not written yet

The training script does not exist. Phase 3 needs:

- a training loop with masked-language-model loss over nucleotides,
- every hyperparameter and seed logged to `results/baselines/run_config.yaml` and metrics to `results/baselines/metrics.json` before any structural model exists,
- the **trained memory-horizon measurement**, which gates Phase 4.

That last one matters more than it sounds. At initialisation the longest memory horizon is 994 tokens against a 5,000-token Hi-C bin, and roughly 15,900 tokens across all sixteen layers relayed — short of a 385 kb TAD. The timescale is learned and can grow, so this is not proof of failure, but it limits the sequence-only baseline exactly as much as the structural model. If trained horizons never approach TAD scale, conditioning on TAD structure is not expressible at this model size, and the mechanism should be re-scoped **before** the expensive phase rather than after.

> **⚠ Superseded 2026-08-15.** The paragraph above describes the architecture at `dt_min=1e-3`, which measurement showed was an exact 1,000-token cap on τ. It is fixed: `dt_min=1e-6`, `dt_floor=1e-7`, no parameter change. Three re-run Phase 3 seeds measure trained τ median **434.7** and **4.85e-02** of triples past 100 kb (was ~0), and the F4 gate passes. Note the scope limit: the training window is 32,768 bp, so this means *the state retains across the full window*, not that the model sees a 385 kb TAD. Details in `architecture_spec.md` §4.1.4.

`BiMambaLM.tau_stats()` returns the per-layer numbers; the training script needs to log it periodically.

## Two things that will confuse you if unflagged

**Splits are within one chromosome.** Train, validation and test are disjoint intervals of chr9 separated by 1 Mb buffers. CHROME holds out chr9 entirely and validates on chr8; we only have chr9, so this is weaker. Adjacent regions of one chromosome share compartment identity and replication timing in ways separate chromosomes do not, so held-out numbers here are optimistic. It belongs in the paper's limitations, not in a footnote.

**Training windows overlap by 50%.** 5,422 windows at stride 16,384 cover about 89 Mb of unique sequence, not 178 Mb. Quote unique coverage when reporting tokens seen.

## Where things are

| Path | What |
|---|---|
| `src/chromfm/model.py` | BiMamba backbone, both variants, reference scan, `tau_stats()` |
| `src/chromfm/scan_triton.py` | custom Triton scan carrying the permeability term |
| `scripts/phase1_*.py` | data acquisition, features, visual validation, dataset layer |
| `scripts/test_model.py` | 16 unit tests, CPU |
| `scripts/validate_kernel.py` | GPU kernel validation |
| `docs/related_work.md` | eleven papers, the gap, verification status per claim |
| `docs/architecture_spec.md` | the mechanism, controls, eight failure modes, decisions |
| `docs/data_card.md` | provenance, measured properties, limitations |
| `project-docs/project.tex` | the living LaTeX record, with the change log |
