#!/bin/bash
# Phase 4 supervisor: pretrain the structural arm to completion, surviving
# disconnects, crashes and the JupyterHub idle culler.
#
# Same structure and the same reasoning as scripts/run_phase3.sh -- read that
# file's header for why the loop lives at a stable path, why completion is
# decided by `status: COMPLETED` in the run's own run_config.yaml rather than an
# exit code, and why re-running this script at any time is safe.
#
# WHAT THIS RUNS, AND WHAT IT DELIBERATELY DOES NOT
# -------------------------------------------------
# architecture_spec.md 4.1.3 splits Phase 4 into two probes:
#
#   P1  swap phi at INFERENCE on the real-structure model. Cheap. THIS IS THE
#       GATE: it passes iff Delta_S1 > 0 with a 95% bootstrap CI over seeds
#       excluding 0, AND Delta_S1 >= 2*sigma_real = 0.0050 bits (sigma_real
#       0.0025, measured over the three Phase 3 v2 seeds).
#   P2  pretrain SEPARATE models under S1 and S3. Six more seeds, ~18 GPU-hours.
#
# This script runs the real-structure arm only -- the 3 seeds P1 needs, about 9
# GPU-hours. P2 is not launched here on purpose: if P1 shows the mechanism is
# inert (S1 ~= real), the spec says do not proceed, and 18 GPU-hours would have
# been spent answering a question that no longer matters. Run P2 by setting
# PHI_CONTROL below once P1 has passed.
#
# The baseline this is compared against is baseline_v2_seed*, NOT baseline_seed*.
# The latter were trained at dt_min=1e-3 and are a different architecture.
#
# Usage:
#   setsid nohup bash scripts/run_phase4.sh >> results/novel_model/train_console.log 2>&1 < /dev/null &

set -u

REPO=/home/jupyter-238w1a5447/3d-gen
# Must match phase4_guard.sh's RUN_TAG. The completion sentinel is scoped to
# this string so a console log from a different window/index cannot make a
# fresh launch exit immediately (that bug cost a full 65 kb launch cycle).
RUN_TAG="${RUN_TAG:-w65536_multichrom}"
PY="$REPO/3d-gen/bin/python"
OUT="$REPO/results/novel_model"
SEEDS="0 1 2"
MAX_ATTEMPTS=20
RETRY_SLEEP=60

# "none" = S0, real measured structure. Set to S1 or S3 to run the P2 arms.
PHI_CONTROL=none

if [ "$PHI_CONTROL" = "none" ]; then
    PREFIX=structural_seed
    CONTROL_ARGS=""
else
    PREFIX="structural_${PHI_CONTROL}_seed"
    CONTROL_ARGS="--phi-control $PHI_CONTROL"
fi

cd "$REPO" || exit 1
mkdir -p "$OUT"

exec 9> "$OUT/supervisor.lock"
if ! flock -n 9; then
    echo "supervisor: another instance holds the lock, exiting $(date -u +%F_%H:%M:%S)"
    exit 0
fi
echo $$ > "$OUT/supervisor.pid"
trap 'rm -f "$OUT/supervisor.pid"' EXIT

completed() {  # $1 = run directory
    grep -q '^status: COMPLETED' "$1/run_config.yaml" 2>/dev/null
}

echo "===== supervisor start $(date -u +%F_%H:%M:%S) pid=$$ prefix=$PREFIX ====="

for S in $SEEDS; do
    RUN="$OUT/$PREFIX$S"

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
        # Every training hyperparameter below is IDENTICAL to run_phase3.sh's,
        # because matched compute is the whole basis of the comparison. The only
        # differences are --structural, the optional control, and the output
        # directory. --ckpt-every 120 for the same reason as in Phase 3: the
        # idle culler fires every 15-20 minutes and a 200-step interval loses
        # more work than it saves.
        "$PY" -u scripts/train.py --seed "$S" --resume --run-name "$PREFIX$S" \
            --out-dir "$OUT" --structural --grad-checkpoint $CONTROL_ARGS \
            --steps 2000 --batch-size 2 --grad-accum 2 \
            --warmup-steps 150 --eval-every 200 --tau-every 200 --ckpt-every 120 \
            --log-every 50 2>&1 | grep --line-buffered -v "Can't initialize NVML"
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

echo "ALL SEEDS DONE ${RUN_TAG} $(date -u +%F_%H:%M:%S)"
