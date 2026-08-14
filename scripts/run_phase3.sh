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

# Run-directory prefix. baseline_v2_* is the re-run on the corrected Delta
# initialisation (dt_min 1e-6, architecture_spec.md 4.1.4 "F4 RESOLVED").
# The original baseline_* runs stay untouched: they are the measurement that
# justified the change, and overwriting them would destroy the evidence for it.
# They are NOT the Phase 4 baseline -- different architecture, and their
# sigma_real does not carry over.
PREFIX=baseline_v2_seed

cd "$REPO" || exit 1
mkdir -p "$OUT"

# Single-instance lock. Two supervisors running at once would have two training
# processes writing the same checkpoint.pt and fighting over the GPUs, so take an
# exclusive lock and exit quietly rather than start a second copy. The lock is
# held on fd 9 for the life of the process and released by the kernel when the
# process dies, however it dies -- a stale lock file cannot block a restart.
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

echo "===== supervisor start $(date -u +%F_%H:%M:%S) pid=$$ ====="

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
        # --ckpt-every 120, not 200 (PI's choice). The JupyterHub idle culler
        # (--timeout=600) kills the whole user cgroup, and by 2026-08-12 it was
        # firing every 15-20 minutes -- shorter than the 17.7 minutes a 200-step
        # checkpoint interval takes at 5.30 s/step. Every restart was therefore
        # recomputing work it never got to save, and seed 2 sat at step 1200
        # across three restarts making no net progress. At 120 steps a checkpoint
        # lands every ~10.6 minutes, inside the observed cull window. Checkpoint
        # frequency does not affect the training computation, only how much of it
        # survives. --eval-every and --tau-every stay at 200 so the logged metric
        # series remains directly comparable with seeds 0 and 1.
        #
        # grep needs --line-buffered: with stdout redirected to a file it block
        # buffers, which is why the console log used to lag tens of lines behind
        # the run.
        "$PY" -u scripts/train.py --seed "$S" --resume --run-name "$PREFIX$S" \
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

echo "ALL SEEDS DONE $(date -u +%F_%H:%M:%S)"
