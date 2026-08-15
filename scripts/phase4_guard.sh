#!/bin/bash
# Watchdog for scripts/run_phase4.sh. Identical in design to phase3_guard.sh --
# see that file for why liveness is checked through the supervisor's own pidfile
# rather than pgrep -f.
#
# After a reboot, or any time training has stopped, this one command restores
# everything: the guard starts the supervisor, the supervisor skips completed
# seeds and resumes partial ones from their checkpoints.
#
#   cd /home/jupyter-238w1a5447/3d-gen
#   setsid nohup bash scripts/phase4_guard.sh >> results/novel_model/guard.log 2>&1 < /dev/null &

set -u

REPO=/home/jupyter-238w1a5447/3d-gen
LOG="$REPO/results/novel_model/train_console.log"
PIDFILE="$REPO/results/novel_model/supervisor.pid"
POLL=120

cd "$REPO" || exit 1
mkdir -p "$REPO/results/novel_model"

supervisor_alive() {
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'run_phase4\.sh'
}

echo "===== guard start $(date -u +%F_%H:%M:%S) pid=$$ ====="

while : ; do
    if grep -q "^ALL SEEDS DONE" "$LOG" 2>/dev/null; then
        echo "guard: supervisor reported ALL SEEDS DONE $(date -u +%F_%H:%M:%S); exiting"
        exit 0
    fi

    if ! supervisor_alive; then
        echo "guard: no supervisor running, starting one $(date -u +%F_%H:%M:%S)"
        setsid nohup bash "$REPO/scripts/run_phase4.sh" >> "$LOG" 2>&1 < /dev/null &
        disown
        sleep 30
    fi

    sleep "$POLL"
done
