#!/bin/bash
# Phase 3 supervisor: run the three sequence-only baseline seeds to completion,
# surviving disconnects, crashes and preemption.
#
# Why this exists rather than a plain `for S in 0 1 2; do python train.py; done`:
#   * The loop must live at a stable path. A driver script under a temp
#     directory can have that directory removed underneath it, and bash reads a
#     script incrementally as it executes.
#   * A seed that dies (OOM because another process grabbed the GPUs, a machine
#     hiccup, a kill) must be retried, not skipped. train.py --resume picks up
#     from results/baselines/baseline_seed<S>/checkpoint.pt, which is written
#     atomically (torch.save to .tmp, then os.replace) every --ckpt-every steps.
#   * Completion is decided by the run's own record -- `status: COMPLETED` in
#     run_config.yaml, written by rank 0 only after the training loop exits
#     normally -- not by an exit code, so a half-finished run can never be
#     mistaken for a finished one.
#
# Re-running this script is safe at any time. A seed that is already COMPLETED
# is skipped; a seed that is partway through continues from its checkpoint.
# Caveat on resume: optimizer, scheduler, step count and RNG are restored, but
# the dataloader's position within the epoch is not, so a run that was
# interrupted sees a different window order than an uninterrupted one would.
#
# Usage:  setsid nohup bash scripts/run_phase3.sh >> results/baselines/train_console.log 2>&1 < /dev/null &

set -u

REPO=/home/jupyter-238w1a5447/3d-gen
PY="$REPO/3d-gen/bin/python"
OUT="$REPO/results/baselines"
SEEDS="0 1 2"
MAX_ATTEMPTS=20
RETRY_SLEEP=60

cd "$REPO" || exit 1
mkdir -p "$OUT"

completed() {  # $1 = run directory
    grep -q '^status: COMPLETED' "$1/run_config.yaml" 2>/dev/null
}

echo "===== supervisor start $(date -u +%F_%H:%M:%S) pid=$$ ====="

for S in $SEEDS; do
    RUN="$OUT/baseline_seed$S"

    if completed "$RUN"; then
        echo "########## SEED $S already COMPLETED, skipping ##########"
        continue
    fi

    attempt=0
    while : ; do
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            echo "########## SEED $S GAVE UP after $MAX_ATTEMPTS attempts ##########"
            break
        fi

        echo "########## SEED $S attempt $attempt start $(date -u +%F_%H:%M:%S) ##########"
        "$PY" -u scripts/train.py --seed "$S" --resume \
            --steps 2000 --batch-size 2 --grad-accum 2 \
            --warmup-steps 150 --eval-every 200 --tau-every 200 --ckpt-every 200 \
            --log-every 50 2>&1 | grep -v "Can't initialize NVML"
        rc=${PIPESTATUS[0]}
        echo "########## SEED $S attempt $attempt exit=$rc $(date -u +%F_%H:%M:%S) ##########"

        if completed "$RUN"; then
            echo "########## SEED $S COMPLETED ##########"
            break
        fi
        echo "seed $S did not reach COMPLETED; retrying in ${RETRY_SLEEP}s"
        sleep "$RETRY_SLEEP"
    done
done

echo "ALL SEEDS DONE $(date -u +%F_%H:%M:%S)"
