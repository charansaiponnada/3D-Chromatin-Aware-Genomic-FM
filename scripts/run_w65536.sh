#!/bin/bash
# The 65,536 bp decision run -- BOTH arms, seed 0, warm-started from 32,768.
#
# Pre-registered in docs/w65536_preregistration.md, committed BEFORE this
# script was launched. Read that file before reading any number this produces.
# It fixes the pass criterion (delta >= +0.0050 bits), the mechanism criterion
# (D1 >= 0.05) and the three-way decision rule, so that none of them can be
# chosen after the fact.
#
# WHAT THIS IS
#   n = 1 per arm. No seed variance, therefore no test and no p-value. This
#   run CANNOT produce a publishable claim and is not intended to. It decides
#   whether the ~150 GPU-hour publishable version is worth launching.
#
# WHY BOTH ARMS ARE HERE
#   run_phase4.sh runs the structural arm only, because Phase 3 had already
#   produced its baseline at the same window. 65,536 is a NEW dataset -- 2,713
#   train / 137 val / 122 test -- so no baseline exists at this width and the
#   32 kb one cannot be reused. Matched compute means both arms must be trained
#   here, back to back, on the same box, with identical hyperparameters.
#
# WARM START
#   --init-from loads weights only (not optimiser, schedule or step) from the
#   finished 32,768 runs. A Mamba SSM has no positional embeddings, so the
#   parameters are window-independent and transfer unchanged. This ADDRESSES
#   weakness #1 (not converged: 2,000 + 1,000 = 3,000 effective steps) and
#   INHERITS weakness #3 (the 32 kb checkpoints predate the paired-init fix).
#   Both are recorded in the pre-registration as accepted in advance.
#
# COMPLETION
#   Decided by `status: COMPLETED` in each run's own run_config.yaml. This
#   script deliberately does NOT grep a console log for a completion string --
#   that is exactly how phase4_guard.sh reported false completion when a stale
#   32 kb log was left in place (2026-08-18 change log entry).
#
# Usage:
#   setsid nohup bash scripts/run_w65536.sh \
#       >> results/novel_model/train_console_w65536.log 2>&1 < /dev/null &

set -u

REPO=/home/jupyter-238w1a5447/3d-gen
PY="$REPO/3d-gen/bin/python"
NOVEL="$REPO/results/novel_model"
BASE="$REPO/results/baselines"
SEED=0
STEPS=1000
MAX_ATTEMPTS=20
RETRY_SLEEP=60

cd "$REPO" || exit 1
mkdir -p "$NOVEL" "$BASE"

exec 9> "$NOVEL/w65536.lock"
if ! flock -n 9; then
    echo "w65536: another instance holds the lock, exiting $(date -u +%F_%H:%M:%S)"
    exit 0
fi
echo $$ > "$NOVEL/w65536.pid"
trap 'rm -f "$NOVEL/w65536.pid"' EXIT

completed() { grep -q '^status: COMPLETED' "$1/run_config.yaml" 2>/dev/null; }

# arm | out-dir | run-name | warm-start source | extra train.py args
run_arm() {
    local OUT="$1" NAME="$2" SRC="$3" EXTRA="$4"
    local RUN="$OUT/$NAME"

    if completed "$RUN"; then
        echo "########## $NAME already COMPLETED, skipping ##########"
        return 0
    fi
    if [ ! -f "$SRC" ]; then
        echo "########## $NAME ABORT: no warm-start checkpoint at $SRC ##########"
        return 1
    fi

    local attempt=0
    while : ; do
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            echo "########## $NAME GAVE UP after $MAX_ATTEMPTS attempts ##########"
            return 1
        fi
        echo "########## $NAME attempt $attempt start $(date -u +%F_%H:%M:%S) ##########"

        # Every hyperparameter below is IDENTICAL across the two arms. The only
        # differences are --structural, --out-dir and the warm-start source.
        # --grad-checkpoint is required at this width (and only works as of the
        # 2026-08-18 import fix). --ckpt-every 120 for the same reason as
        # Phase 3/4: the idle culler fires more often than a 200-step interval
        # can save.
        "$PY" -u scripts/train.py --seed "$SEED" --resume --run-name "$NAME" \
            --out-dir "$OUT" --init-from "$SRC" --grad-checkpoint $EXTRA \
            --steps "$STEPS" --batch-size 2 --grad-accum 2 \
            --warmup-steps 100 --eval-every 100 --tau-every 100 \
            --ckpt-every 120 --keep-every 250 \
            --log-every 25 2>&1 | grep --line-buffered -v "Can't initialize NVML"
        local rc=${PIPESTATUS[0]}
        echo "########## $NAME attempt $attempt exit=$rc $(date -u +%F_%H:%M:%S) ##########"

        if completed "$RUN"; then
            echo "########## $NAME COMPLETED ##########"
            return 0
        fi
        echo "$NAME did not reach COMPLETED; retrying in ${RETRY_SLEEP}s"
        sleep "$RETRY_SLEEP"
    done
}

echo "===== w65536 decision run start $(date -u +%F_%H:%M:%S) pid=$$ ====="
echo "pre-registration: docs/w65536_preregistration.md"

# Baseline first. If the box dies partway through, a finished baseline plus an
# unfinished structural arm is a recoverable state; the reverse tempts a
# comparison against the 32 kb baseline, which is a different dataset.
run_arm "$BASE"  "baseline_v2_w65536_seed$SEED" \
        "$BASE/w32768/baseline_v2_seed$SEED/checkpoint.pt" ""

run_arm "$NOVEL" "structural_w65536_seed$SEED" \
        "$NOVEL/w32768/structural_seed$SEED/checkpoint.pt" "--structural"

echo "===== w65536 decision run finished $(date -u +%F_%H:%M:%S) ====="
