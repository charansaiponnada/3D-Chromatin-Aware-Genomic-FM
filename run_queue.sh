#!/usr/bin/env bash
# Pre-registered confirmatory runs (docs/preregistration.md Section 3), one GPU.
#   bash run_queue.sh <gpu> <arm:seed> [<arm:seed> ...]
# Recipe v1 = configs/base.yaml defaults. Validation split only; test stays sealed.
set -u
cd ~/3D-Chromatin-Aware-Genomic-FM
source .venv/bin/activate
mkdir -p logs

GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU

for spec in "$@"; do
    arm=${spec%%:*}
    seed=${spec##*:}
    run="${arm}_seed${seed}"

    echo "=== $(date -Is) train $run"
    python -u scripts/pretrain.py --arm "$arm" --seed "$seed" >> "logs/${run}.train.log" 2>&1
    echo "TRAIN_EXIT=$? $run"

    echo "=== $(date -Is) evaluate $run (val)"
    python -u scripts/evaluate.py --run "$run" --split val >> "logs/${run}.eval.log" 2>&1
    echo "EVAL_EXIT=$? $run"

    echo "=== $(date -Is) probe $run (val)"
    python -u scripts/probe.py --run "$run" >> "logs/${run}.probe.log" 2>&1
    echo "PROBE_EXIT=$? $run"
done

echo "QUEUE_DONE gpu${GPU}"
