#!/bin/bash
# One-shot status of the Phase 4 structural arm. Safe to run at any time; it
# only reads.
#
#   bash scripts/phase4_status.sh
#
# NOTE ON THE CONSOLE LOG: run_phase4.sh pipes train.py's output through grep to
# drop the NVML warning, and grep block-buffers when its stdout is a file, so
# results/novel_model/train_console.log can lag the run by many minutes.
# metrics.json is written atomically by train.py itself and never lags -- it is
# the source this script reads for progress.

REPO=/home/jupyter-238w1a5447/3d-gen
cd "$REPO" || exit 1

echo "=== $(TZ='Asia/Kolkata' date '+%F %H:%M:%S') IST  ($(date -u '+%H:%M:%S') UTC) ==="
echo

if grep -q "^ALL SEEDS DONE" results/novel_model/train_console.log 2>/dev/null; then
    echo "STATUS: all seeds finished."
elif ps -eo pid,cmd | grep -q "[s]cripts/train.py"; then
    echo "STATUS: training is running."
else
    echo "STATUS: NO TRAINER RUNNING and seeds are not done."
    echo "Restart it with:"
    echo "  cd $REPO && setsid nohup bash scripts/phase4_guard.sh >> results/novel_model/guard.log 2>&1 < /dev/null &"
fi
echo

echo "--- processes ---"
ps -eo pid,ppid,etime,cmd | grep -E "phase4_guard|run_phase4|scripts/train\.py" \
    | grep -v grep | cut -c1-95
echo

echo "--- evals logged (2000 steps per seed, 3 seeds) ---"
./3d-gen/bin/python scripts/phase3_progress.py 'results/novel_model/structural*seed*'

echo
echo "--- seed completion ---"
for S in 0 1 2; do
    cfg="results/novel_model/structural_seed$S/run_config.yaml"
    if [ -f "$cfg" ]; then
        printf "seed %s: %s\n" "$S" "$(grep -m1 '^status:' "$cfg")"
    else
        printf "seed %s: not started\n" "$S"
    fi
done

echo
echo "--- Phase 3 v2 baseline, for reference (different arm, same recipe) ---"
./3d-gen/bin/python scripts/phase3_progress.py 'results/baselines/baseline_v2_seed*' \
    | grep "step  2000"

echo
echo "--- GPU memory ---"
./3d-gen/bin/python -c "
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU{i} {torch.cuda.get_device_name(i)}  used {(total-free)/2**30:.2f} GiB / {total/2**30:.2f} GiB')
" 2>/dev/null
