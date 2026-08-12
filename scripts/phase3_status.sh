#!/bin/bash
# One-shot status of the Phase 3 baseline runs. Safe to run at any time; it only
# reads. Prints whether the supervisor chain is alive, every eval logged so far,
# and current GPU memory (read through the CUDA API, because nvidia-smi is broken
# on this box -- an NVML version mismatch that does not affect running kernels).
#
#   bash scripts/phase3_status.sh

REPO=/home/jupyter-238w1a5447/3d-gen
cd "$REPO" || exit 1

# Times are shown in IST, the PI's local zone, with UTC alongside. Timestamps
# written into result files stay UTC (train.py writes completed_at_utc), because
# a stored time should not depend on who reads it; only the display converts.
echo "=== $(TZ='Asia/Kolkata' date '+%F %H:%M:%S') IST  ($(date -u '+%H:%M:%S') UTC) ==="
echo

if grep -q "^ALL SEEDS DONE" results/baselines/train_console.log 2>/dev/null; then
    echo "STATUS: all seeds finished."
elif ps -eo pid,etime,cmd | grep -q "[s]cripts/train.py"; then
    echo "STATUS: training is running."
else
    echo "STATUS: NO TRAINER RUNNING and seeds are not done."
    echo "Restart it with:"
    echo "  cd $REPO && setsid nohup bash scripts/phase3_guard.sh >> results/baselines/guard.log 2>&1 < /dev/null &"
fi
echo

echo "--- processes ---"
ps -eo pid,ppid,etime,cmd | grep -E "phase3_guard|run_phase3|scripts/train\.py" \
    | grep -v grep | cut -c1-95
echo

echo "--- evals logged (2000 steps per seed, 3 seeds) ---"
./3d-gen/bin/python scripts/phase3_progress.py

echo
echo "--- seed completion ---"
for S in 0 1 2; do
    cfg="results/baselines/baseline_seed$S/run_config.yaml"
    if [ -f "$cfg" ]; then
        printf "seed %s: %s\n" "$S" "$(grep -m1 '^status:' "$cfg")"
    else
        printf "seed %s: not started\n" "$S"
    fi
done

echo
echo "--- GPU memory ---"
./3d-gen/bin/python -c "
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU{i} {torch.cuda.get_device_name(i)}  used {(total-free)/2**30:.2f} GiB / {total/2**30:.2f} GiB')
" 2>/dev/null
