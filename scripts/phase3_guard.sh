#!/bin/bash
# Watchdog for scripts/run_phase3.sh.
#
# The supervisor itself retries failed training runs, so this only covers the
# case where the supervisor process is killed outright (SIGKILL, session reaper,
# OOM killer). It polls every two minutes and restarts the supervisor if no copy
# is running, then exits once the supervisor reports all seeds done.
#
# There is no cron or systemd --user on this box (the user crontab directory is
# not writable and there is no session bus), so this is a plain detached poller.
# It does NOT survive a machine reboot -- after a reboot, relaunch by hand:
#
#   cd /home/jupyter-238w1a5447/3d-gen
#   setsid nohup bash scripts/phase3_guard.sh >> results/baselines/guard.log 2>&1 < /dev/null &
#
# That single command is enough to restore everything: the guard starts the
# supervisor, the supervisor skips completed seeds and resumes partial ones.

set -u

REPO=/home/jupyter-238w1a5447/3d-gen
LOG="$REPO/results/baselines/train_console.log"
PIDFILE="$REPO/results/baselines/supervisor.pid"
POLL=120

cd "$REPO" || exit 1

# Liveness is checked through the supervisor's own pidfile, confirmed against
# that pid's command line. Matching on `pgrep -f run_phase3.sh` instead is wrong:
# it also matches any unrelated shell whose command line merely mentions the
# script -- an editor, a grep, the command that launched this guard -- and a
# false "already running" would leave training stopped indefinitely.
supervisor_alive() {
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'run_phase3\.sh'
}

echo "===== guard start $(date -u +%F_%H:%M:%S) pid=$$ ====="

while : ; do
    if grep -q "^ALL SEEDS DONE" "$LOG" 2>/dev/null; then
        echo "guard: supervisor reported ALL SEEDS DONE $(date -u +%F_%H:%M:%S); exiting"
        exit 0
    fi

    if ! supervisor_alive; then
        echo "guard: no supervisor running, starting one $(date -u +%F_%H:%M:%S)"
        setsid nohup bash "$REPO/scripts/run_phase3.sh" >> "$LOG" 2>&1 < /dev/null &
        disown
        sleep 30
    fi

    sleep "$POLL"
done
