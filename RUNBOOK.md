# Runbook

Every command, in order. Run them from the repository root.

Anything marked **laptop** needs no GPU. Anything marked **GPU** does.

---

## 0. Set up — 10 minutes, either machine

```bash
git clone https://github.com/charansaiponnada/3D-Chromatin-Aware-Genomic-FM.git
cd 3D-Chromatin-Aware-Genomic-FM
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

On the GPU box, install the CUDA build of torch **first**, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Then, before anything else, find out what the machine can do

```bash
python scripts/probe_env.py
```

Prints Python version, GPU model and VRAM, whether bf16 works, whether every
package imports, free disk, and the SLURM limits. Writes `env_probe.json`.
This one file answers most of the questions that otherwise cost days, and its
output is what the paper's compute statement needs anyway.

**If it reports Python 3.13 and a missing torch**, use a 3.11 environment
(`module load python/3.11`, conda, or a `pytorch/pytorch:2.x-cuda12.x`
container). `mamba_ssm` showing MISSING costs nothing — the model uses a
pure-PyTorch state-space layer.

---

## 1. Prove the code works — 1 minute, **laptop**, no downloads

```bash
python tests/test_config.py
python scripts/smoke_test.py
```

The smoke test fabricates a chromosome in the real on-disk format and runs
Phase 2, Phase 3 and Phase 5 over it. 34 checks. It verifies that the controls
change what they claim to, that held-out edges stay held out, that gradients
reach the contact-bias MLPs, and that removing the graph changes the output.

It does **not** tell you the model works on real chromatin. Nothing can, before
Phase 5.

---

## 2. Build the pilot dataset — a few hours, **laptop**

```bash
python scripts/build_data.py --pilot
```

Downloads hg38 (3.1 GB) and GM12878's `.mcool`, then builds chr20 and chr21 at
5 kb. Writes `data/processed/GM12878/chr20_5000/` and a `qc.json` per
chromosome.

**Check the QC before going further.** In each `qc.json`:

- `mean_oe_by_separation` should be **1.0 at every separation** — that is the
  detrending invariant, and if it is not 1.0 the distance correction is broken.
- `frac_usable` should be around 0.9. Much lower means ICE filtered most bins.
- `checksums` are the reproducibility gate — see step 4.

Disk: you need about **60 GB free**. Check first; `website/node_modules`
currently eats a chunk of the laptop's free space.

---

## 3. Verify the graphs and the controls — 1 minute, **laptop**

```bash
python scripts/build_graphs.py --chrom chr20 --all-arms
```

Prints edge counts, mean strength and mean separation for all seven arms, then
three sanity checks:

- **b3 keeps the edge count** — a control that also changed the count would
  confound *wrong structure* with *less structure*.
- **b3 keeps the strength distribution** — same marginals, destroyed pairing.
- **b2 flattens strength to a constant** — distance kept, contact removed.

If any of those fail, stop. Every downstream number would be meaningless.

---

## 4. Move to the GPU machine

```bash
git push                                    # from the laptop
# on the GPU box:
git clone ... && cd ... && pip install -e .
python scripts/probe_env.py
python scripts/smoke_test.py
python scripts/build_data.py --pilot
```

Then confirm the rebuild is byte-identical:

```bash
python -c "import json,pathlib; print(json.load(open('data/processed/GM12878/chr20_5000/qc.json'))['checksums'])"
```

Compare against the laptop's. **Different checksums mean something
environment-dependent got into the pipeline** — find it now, not after Phase 4.

---

## 5. First real training run — **GPU**

Smoke first, with a tiny model, to confirm the loop runs on this hardware:

```bash
python scripts/pretrain.py --arm full --seed 0 --smoke --steps 20
```

Then a timed run, to get the number that replaces every compute estimate:

```bash
time python scripts/pretrain.py --arm full --seed 0 --steps 100
```

Take `steps_per_sec` from `results/full_seed0/history.json` and multiply.
**That measurement supersedes every GPU-hour figure anyone has given you**,
including mine. Put it in the paper.

If it OOMs, lower `train.batch_size` to 1 and raise `train.grad_accum_steps` —
the effective batch is their product, so the maths is unchanged.

---

## 6. Full dataset — hours, **GPU box**, as a batch job

```bash
sbatch --time=12:00:00 --wrap "python scripts/build_data.py --full"
# no SLURM? nohup python scripts/build_data.py --full > build.log 2>&1 &
```

Do **not** run this in an interactive session. An idle culler will kill it
halfway and leave you with a partial dataset and no error message.

---

## 7. The screening sweep — one seed, every arm

```bash
for arm in b0_dna_only b2_distance_only b3_shuffled_hic b4_late_fusion b5_contrastive_only full; do
  python scripts/pretrain.py --arm $arm --seed 0
  python scripts/evaluate.py --run ${arm}_seed0
done
python scripts/evaluate.py --collect
```

Then **stop and look** at `results/final_comparison.json`, specifically
`full` against `b3_shuffled_hic` on `contact_r_long_hic_free_mean`.

If there is no separation at one seed, there is unlikely to be one at five.
That is the Phase 4 stop gate, and honouring it saves roughly 500 GPU-hours.
A clean negative here is a publishable result; an unexamined positive is not.

---

## 8. Confirmation — the four arms that carry the argument

```bash
for seed in 1 2 3 4; do
  for arm in b0_dna_only b2_distance_only b3_shuffled_hic full; do
    python scripts/pretrain.py --arm $arm --seed $seed
    python scripts/evaluate.py --run ${arm}_seed${seed}
  done
done
python scripts/evaluate.py --collect
```

Five seeds, not three: the effect will be small and three seeds may not
separate it from noise.

---

## 9. Held-out cell line

```bash
python scripts/build_data.py --cell-lines K562 --chroms chr3 chr13 chr17
python scripts/evaluate.py --run full_seed0 --held-out-cell-line K562
```

**Disclose the depth caveat.** GM12878's map comes from 72 experiments; K562's
from 6. A cross-cell-line difference is partly a sequencing-depth difference,
and the paper has to say so.

---

## Things that will bite you

**Resume is automatic.** An interrupted run picks up from
`results/<run>/checkpoint.pt`. Re-run the identical command; nothing is lost
beyond the last 500 steps.

**Never compare `with_hic` against a sequence-only baseline.** `metrics.json`
reports both settings. `hic_free` is the headline — every arm on sequence
alone, the only difference being what shaped their weights. `with_hic` is an
upper bound with privileged input.

**Results tables stay `??` until `results/final_comparison.json` exists.** The
decks and the website read that file. That is deliberate; do not type numbers
in by hand.

**Regenerate the decks and figures after results land:**

```bash
python scripts/export_diagrams.py
python scripts/build_decks.py
python docs/worked_example.py && cd docs && pdflatex chromgraphfm_explained.tex
```

---

## Measured on this machine

| | |
|---|---|
| Model, production config | **7.29M parameters** |
| Split | encoder 3.33M, blocks 3.16M, heads 0.73M |
| Conv tower | 5,000 bp → **157 positions** into the state-space layers |
| One sample | 128 windows × 5 kb = 640 kb |
| Local encoder, CPU | 267 ms for 4 windows → about 8.5 s per sample |

The CPU figure is there to be replaced. Run step 5 and use the real number.
