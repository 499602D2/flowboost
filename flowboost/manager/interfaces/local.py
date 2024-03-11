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
            script_kv = Manager._construct_scipt_args(script_args, " ")
            cmd.extend(script_kv)

        # Execute the script and get the PID
        process = subprocess.Popen(cmd, cwd=submission_cwd, start_new_session=True)
        pid = process.pid

        # Create and track the job
        return str(pid)

    def _cancel_job(self, job_id: str) -> bool:
        try:
            os.kill(int(job_id), signal.SIGTERM)
        except OSError:
            return False

        return True

    def _job_has_finished(self, job_id: str) -> bool:
        try:
            # Check if the process is still running
            process = psutil.Process(int(job_id))
            # If the process is running or sleeping, it's not finished
            if process.status() in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]:
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return True

        return True
