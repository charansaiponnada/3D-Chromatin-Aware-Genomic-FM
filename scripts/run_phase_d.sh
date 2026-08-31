#!/bin/bash
# PHASE D -- the decision run. Both arms, 5 paired seeds, leakage-safe split.
#
# Pre-registered in docs/PREREG_PHASE_D_2026-08-31.md. PI decisions recorded
# 2026-08-31:
#   D-a  --no-permeability. p imposes tau <= 1/p ~ 55 tokens on the STRUCTURAL
#        arm only; the baseline has no p term. Disabling it is the only option
#        that makes tau mean the same thing in both arms. D2 measured the gate
#        as never engaging, so nothing measured is given up.
#   D-b  --phi-granularity dual (T5c-dual). keep(phi)=0.1099 at 65,536 means a
#        per-position bias reaches ~11% of the structural variance BY
#        CONSTRUCTION; dual contains per-position as a strict subset and is the
#        only implemented arm with access to the rest.
#   D-c  5 paired seeds per arm. At 3v3 the exact permutation floor is 0.100 and
#        p<0.05 is unreachable BY DESIGN -- that is why Phase 4 could not have
#        produced a significant positive. At 5v5 the floor is 2/252 = 0.008.
#   D-d  window 65,536 (from the index; 131,072 fits but costs 2.41x/step).
#
# CONVERGENCE, not a fixed step count (prereg 3): --early-stop with the
# pre-registered delta 0.0010 over a 1,000-step window, 2 consecutive evals,
# and --steps 8000 as a HARD CAP. Both arms get the same criterion and the same
# cap; run_config.yaml records which one fired (stop_reason).
#
# WHY TWO LANES INSTEAD OF DDP
#   NCCL cannot initialise on this box (nvmlInit_v2, CLAUDE.md 3) so DDP falls
#   back to gloo, which measured 1.75x SLOWER per step at 32,768. A 65,536 run
#   with --grad-checkpoint peaks at 7.51 GiB (results/p5_memcheck.json), so two
#   independent single-GPU runs fit trivially in 2x44.39 GiB and give true 2x
#   throughput with no allreduce at all. Lane A runs the baselines, lane B the
#   structural arm.
#
# Usage:
#   setsid nohup bash scripts/run_phase_d.sh >> results/phase_d/console.log 2>&1 < /dev/null &

set -u

REPO=/home/jupyter-238w1a5447/3d-gen
PY="$REPO/3d-gen/bin/python"
OUT="$REPO/results/phase_d"
INDEX="${INDEX:-dataset_index_multichrom.npz}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-8000}"
RUN_TAG="${RUN_TAG:-phaseD_w65536_multichrom_dual}"
MAX_ATTEMPTS=40
RETRY_SLEEP=60

cd "$REPO" || exit 1
mkdir -p "$OUT"

exec 9> "$OUT/phase_d.lock"
if ! flock -n 9; then
    echo "phase_d: another instance holds the lock, exiting $(date -u +%F_%H:%M:%S)"
    exit 0
fi
echo $$ > "$OUT/phase_d.pid"
trap 'rm -f "$OUT/phase_d.pid"' EXIT

completed() { grep -q '^status: COMPLETED' "$1/run_config.yaml" 2>/dev/null; }

# lane: $1 = GPU id, $2 = arm ("baseline"|"structural"), $3 = log file
lane() {
    local GPU="$1" ARM="$2" LOG="$3"
    local EXTRA PREFIX
    if [ "$ARM" = "structural" ]; then
        PREFIX="structural_dual_d_seed"
        EXTRA="--structural --phi-granularity dual --no-permeability"
    else
        PREFIX="baseline_d_seed"
        EXTRA=""
    fi

    for S in $SEEDS; do
        local RUN="$OUT/$PREFIX$S"
        if completed "$RUN"; then
            echo "[$ARM] seed $S already COMPLETED, skipping" >> "$LOG"
            continue
        fi
        local attempt=0
        while : ; do
            attempt=$((attempt + 1))
            if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
                echo "[$ARM] seed $S GAVE UP after $MAX_ATTEMPTS attempts" >> "$LOG"
                break
            fi
            echo "[$ARM] seed $S attempt $attempt start $(date -u +%F_%H:%M:%S)" >> "$LOG"
            # Every hyperparameter is IDENTICAL across arms. The only
            # differences are $EXTRA. Matched steps and matched tokens are the
            # whole basis of the comparison.
            CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u scripts/train.py \
                --seed "$S" --resume --run-name "$PREFIX$S" --out-dir "$OUT" \
                --index "$INDEX" --gpus 1 --grad-checkpoint $EXTRA \
                --steps "$STEPS" --batch-size 2 --grad-accum 2 \
                --early-stop --early-stop-delta 0.0010 \
                --early-stop-window 1000 --early-stop-patience 2 \
                --warmup-steps 300 --eval-every 250 --tau-every 500 \
                --ckpt-every 120 --keep-every 1000 --log-every 50 \
                >> "$LOG" 2>&1
            local rc=$?
            echo "[$ARM] seed $S attempt $attempt exit=$rc $(date -u +%F_%H:%M:%S)" >> "$LOG"
            if completed "$RUN"; then
                echo "[$ARM] seed $S COMPLETED" >> "$LOG"
                break
            fi
            sleep "$RETRY_SLEEP"
        done
    done
    echo "[$ARM] LANE DONE ${RUN_TAG} $(date -u +%F_%H:%M:%S)" >> "$LOG"
}

echo "===== PHASE D start $(date -u +%F_%H:%M:%S) pid=$$ tag=$RUN_TAG ====="
echo "index=$INDEX seeds='$SEEDS' cap=$STEPS"

lane 0 baseline   "$OUT/lane_baseline.log"   &
LP0=$!
lane 1 structural "$OUT/lane_structural.log" &
LP1=$!
wait $LP0 $LP1

echo "===== PHASE D ALL LANES DONE ${RUN_TAG} $(date -u +%F_%H:%M:%S) ====="
