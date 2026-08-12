"""Template for ~/.jupyter/jupyter_server_config.py -- NOT active from here.

This file does nothing where it sits. It only takes effect once copied to
~/.jupyter/jupyter_server_config.py, which is first on the single-user server's
config search path and is executed every time that server starts:

    cp scripts/jupyter_server_config.py ~/.jupyter/jupyter_server_config.py

To disable it again, delete ~/.jupyter/jupyter_server_config.py. Nothing else
in the project depends on it.

Sole purpose: resume the Phase 3 training run automatically whenever the
single-user server starts.

Why this is needed. The hub on this box runs jupyterhub_idle_culler with
--timeout=600, and the entire user environment lives in one systemd cgroup
(/system.slice/jupyter-<user>.service). Ten minutes after the last browser
activity the hub stops that service and systemd kills every process in the
cgroup, so a detached training run is killed too -- setsid, nohup and PPID=1
protect against a hangup and against a dying parent, neither of which is what
happens here. A nine-hour job therefore cannot run unattended, and the first
Phase 3 launch was killed at step 1000 this way.

This does not try to keep the server alive or to defeat the culler; the idle
policy is the admins' to set. It solves the other half: when the server starts
again, training resumes from its last checkpoint without anyone typing a
command. Checkpoints are written every 200 steps, so a cull costs at most ~18
minutes of recomputation and nothing scientific.

Everything here is wrapped so that a failure can never stop the server from
starting -- being unable to open JupyterLab would be a far worse outcome than a
training run that needs a manual nudge.
"""

import subprocess
from pathlib import Path

_REPO = Path("/home/jupyter-238w1a5447/3d-gen")
_AUTOSTART = _REPO / "scripts" / "phase3_autostart.sh"
_LOG = _REPO / "results" / "baselines" / "autostart.log"


def _resume_phase3() -> None:
    if not _AUTOSTART.is_file():
        return  # script removed -> Phase 3 is over, nothing to do
    with open(_LOG, "a", encoding="utf-8") as log:
        # phase3_autostart.sh is idempotent: it exits immediately if a guard is
        # already running or if all seeds are finished. timeout is a backstop
        # only -- the script either returns at once or detaches and returns.
        subprocess.run(
            ["/bin/bash", str(_AUTOSTART)],
            cwd=str(_REPO), stdout=log, stderr=subprocess.STDOUT,
            timeout=60, check=False,
        )


try:
    _resume_phase3()
except Exception as exc:  # never block server startup
    try:
        with open(_LOG, "a", encoding="utf-8") as log:
            log.write(f"autostart: config hook failed: {exc!r}\n")
    except Exception:
        pass
