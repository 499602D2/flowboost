import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Optional

import psutil

from flowboost.manager.manager import Manager
from flowboost.openfoam.interface import FOAM


class Local(Manager):
    def __init__(self, wdir: Path | str, job_limit: int = 1) -> None:
        super().__init__(wdir, job_limit)
        self.shell: str = "bash"  # os.getenv("SHELL", "bash")

    @staticmethod
    def _is_available() -> bool:
        available = FOAM.in_env()
        if not available:
            logging.error("OpenFOAM not found in PATH")

        return available

    def _submit_job(
        self,
        job_name: str,
        submission_cwd: Path,
        script: Path,
        script_args: dict[str, Any] = {},
    ) -> Optional[str]:
        # Base command
        cmd = [self.shell, script]

        if script_args:
            cmd.extend(Manager._construct_script_arg_list(script_args))

        # Execute the script and get the PID
        process = subprocess.Popen(cmd, cwd=submission_cwd, start_new_session=True)
        pid = process.pid

        # Create and track the job
        return str(pid)

    def _cancel_job(self, job_id: str) -> bool:
        try:
            os.killpg(int(job_id), signal.SIGTERM)
        except OSError:
            return False

        return True

    def _job_has_finished(self, job_id: str) -> bool:
        try:
            process = psutil.Process(int(job_id))
            if not process.is_running():
                return True

            live_statuses = {
                getattr(psutil, status_name)
                for status_name in (
                    "STATUS_RUNNING",
                    "STATUS_SLEEPING",
                    "STATUS_DISK_SLEEP",
                    "STATUS_STOPPED",
                    "STATUS_TRACING_STOP",
                    "STATUS_WAKING",
                    "STATUS_IDLE",
                    "STATUS_LOCKED",
                    "STATUS_WAITING",
                    "STATUS_PARKED",
                )
                if hasattr(psutil, status_name)
            }
            if process.status() in live_statuses:
                return False

            finished_statuses = {psutil.STATUS_ZOMBIE}
            dead_status = getattr(psutil, "STATUS_DEAD", None)
            if dead_status is not None:
                finished_statuses.add(dead_status)

            return process.status() in finished_statuses
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            return False
        except psutil.ZombieProcess:
            return True

        return False
