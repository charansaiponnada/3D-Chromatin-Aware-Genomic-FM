# Handover — the two jobs running as of 2026-08-31 23:45 IST

Everything here is a terminal command. Nothing needs Claude.

---

## 1. Phase D — the decision run (the important one)

**Started 23:21 IST 31-Aug.** 10 runs: 5 baseline + 5 structural-dual, two GPU
lanes in parallel.

### Restart it (safe at ANY time — this is also the culler recovery)
```bash
\
setsid nohup bash scripts/run_phase_d.sh >> results/phase_d/console.log 2>&1 < /dev/null &
```

**It resumes, it does not restart.** Two mechanisms:
- A seed whose `run_config.yaml` says `status: COMPLETED` is **skipped entirely**.
- A partial run resumes from `checkpoint.pt`, written every 120 steps, via
  `--resume`. Worst case loss is 120 steps ≈ **26 minutes**.

Run it as often as you like. If a copy is already running the `flock` makes the
second one exit immediately — it cannot double-start.

### Check progress
```bash
cd /home/jupyter-238w1a5447/3d-gen
grep -hE "^\[rank0\] (step|eval|EARLY)" results/phase_d/lane_*.log | tail -20
grep -l "status: COMPLETED" results/phase_d/*/run_config.yaml | wc -l   # 0..10
```

### Is it alive?
```bash
pgrep -af "train.py.*phase_d" | wc -l    # expect 2 (one per lane)
```
If this prints 0, the culler killed it → run the restart command above.

---

## 2. C1 — the benchmark sweep (secondary)

**4 of 8 tasks complete.** Results accumulate in
`results/c1_genomic_benchmarks.json`.

### Restart it
```bash
cd /home/jupyter-238w1a5447/3d-gen && \
nohup ./3d-gen/bin/python -u scripts/c1_genomic_benchmarks.py \
  --datasets human_ensembl_regulatory demo_coding_vs_intergenomic_seqs \
             demo_human_or_worm human_ocr_ensembl \
  >> results/c1_sweep.log 2>&1 &
```

**It resumes too**, at (model, dataset) granularity: every embedding is cached to
`results/c1/emb/`, every finished probe is already in the JSON and is skipped
with `(cached)`. A kill costs only the item in flight. Safe to re-run; it will
skip everything already done.

### Check progress
```bash
grep -vE "NVML|warnings.warn" results/c1_sweep.log | grep -E "^---|acc |floor" | tail -20
```

---

## Timing (projections — arithmetic shown, not measurements)

Measured **12.94 s/optimizer-step** on the lane that has a GPU to itself.
8,000-step cap → 28.8 h/seed → 5 seeds ≈ **144 h ≈ 6.0 days**.

| scenario | Phase D finishes |
|---|---|
| every run hits the 8,000 cap | **~23:00 IST 6-Sep** |
| convergence fires at ~6,000 steps | ~11:00 IST 5-Sep |
| convergence fires at ~4,000 steps | ~23:00 IST 3-Sep |

C1: **~05:00–07:00 IST 1-Sep**, wide error bars — it is sharing GPU 0 with the
baseline lane, which is why that lane currently reports 27.7 s/step against the
structural lane's 12.94. **The baseline lane speeds up when C1 finishes.**

---

## Rules that still apply

- **Do not read any Phase D number until that run's `run_config.yaml` says
  `status: COMPLETED`.** Standing rule.
- `metrics.json` never lags; the console log does (block buffering).
- **Never edit a running `.sh`** — bash reads it incrementally by byte offset.
- The idle culler kills the whole cgroup, supervisor included, after 10 minutes
  with no browser tab. `docs/ADMIN_REQUEST_idle_culler.md` is drafted and still
  unsent; over a six-day run it is worth more than any code change.
