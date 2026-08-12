#!/bin/bash
# Start the Phase 3 guard if it is not already running. Idempotent and safe to
# call repeatedly.
#
# This exists because the JupyterHub idle culler on this box runs with
# --timeout=600: ten minutes after the last browser activity, the hub stops the
# single-user server, and systemd kills every process in the
# jupyter-<user>.service cgroup -- the whole user environment is one cgroup, so
# setsid/nohup/PPID=1 give no protection. A multi-hour training run therefore
# cannot survive an unattended night, and did not: the first Phase 3 launch was
# killed at step 1000.
#
# This does NOT try to keep the server alive -- that would defeat a resource
# policy the admins set deliberately. It does the opposite half of the problem:
# when the server legitimately starts again, training picks itself back up from
# its last checkpoint with no human command. Called from
# ~/.jupyter/jupyter_server_config.py, which the single-user server executes at
# startup.
#
# Manual use is identical:  bash scripts/phase3_autostart.sh

set -u

REPO=/home/jupyter-238w1a5447/3d-gen
OUT="$REPO/results/baselines"
LOG="$OUT/train_console.log"
GUARD_PID="$OUT/guard.pid"

cd "$REPO" || exit 1

stamp() { echo "$(TZ='Asia/Kolkata' date '+%F_%H:%M:%S')IST/$(date -u '+%H:%M:%S')UTC"; }

# Same liveness test the guard uses for the supervisor: a recorded pid is only
# believed if that pid's command line still names the script. A bare
# `pgrep -f phase3_guard.sh` is wrong here -- it also matches any shell whose
# command line merely mentions the script, including the one launching it, and
# a false "already running" is the failure that leaves training stopped.
guard_alive() {
    local pid
    pid=$(cat "$GUARD_PID" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'phase3_guard\.sh'
}

if grep -q "^ALL SEEDS DONE" "$LOG" 2>/dev/null; then
    echo "autostart $(stamp): all seeds already done, nothing to do"
    exit 0
fi

if guard_alive; then
    echo "autostart $(stamp): guard already running (pid $(cat "$GUARD_PID")), nothing to do"
    exit 0
fi

setsid nohup bash "$REPO/scripts/phase3_guard.sh" >> "$LOG" 2>&1 < /dev/null &
pid=$!
disown
echo "$pid" > "$GUARD_PID"
echo "autostart $(stamp): started guard pid $pid"
