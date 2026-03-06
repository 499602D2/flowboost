import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from flowboost.manager.manager import Manager

class Slurm(Manager):
    def __init__(self, wdir: Path | str, job_limit: int) -> None:
        super().__init__(wdir, job_limit)

    @staticmethod
    def _is_available() -> bool:
        if shutil.which("sbatch") and shutil.which("squeue"):
            return True

        logging.error("sbatch or squeue commands not found in PATH")
        return False

    def _submit_job(
        self,
        job_name: str,
        submission_cwd: Path,
        script: Path,
        script_args: dict[str, Any] = {},
    ) -> Optional[str]:
        # Base command with name
        cmd = ["sbatch", "--job-name", job_name]

        if script_args:
            # If Allrun accepts args, pass them
            script_kv = Manager._construct_scipt_args(script_args)
            cmd.extend(["-v", script_kv])

        cmd.append(str(script))
        # Run in case working directory
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=submission_cwd)

        if result.returncode != 0:
            logging.error(f"Error submitting job: out='{result.stderr.strip()}'")
            return None

        job_id = result.stdout.split()[-1]
        return job_id


    def _cancel_job(self, job_id: str) -> bool:
        cmd = ["scancel", job_id]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error cancelling jobs: {result.stderr.strip()}")
            return False

        return True

    def _job_has_finished(self, job_id: str) -> bool:
        """Check if a Slurm job has finished."""
        cmd = ["squeue", "-j", job_id, "-h"]  # -h suppresses header
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Job not in queue - likely finished or never existed
            if "Invalid job id specified" in result.stderr:
                return True

            logging.warning(
                f"Querying job finish status failed: {result.stderr.strip()}"
            )
            return False

        # If squeue returns successfully with output, job is still running/pending
        if result.stdout.strip():
            return False

        # Empty output means job finished
        return True

    def _get_job_info(self, job_id: str) -> str:
        """Fetch job details from Slurm using the given job_id."""
        cmd = ["scontrol", "show", "job", job_id]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout