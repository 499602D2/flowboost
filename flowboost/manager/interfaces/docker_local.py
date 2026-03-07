import logging
import os
import subprocess
from pathlib import Path
from subprocess import PIPE
from typing import Any, Optional

from flowboost.manager.manager import Manager
from flowboost.openfoam.runtime import FOAMRuntime, docker_image_name, get_runtime


class DockerLocal(Manager):
    """Run OpenFOAM simulations in per-job Docker containers.

    Each submitted case gets its own detached container. The case directory
    is bind-mounted into the container so simulation output appears directly
    on the host filesystem.
    """

    def __init__(self, wdir: Path | str, job_limit: int = 1) -> None:
        self._docker_image = docker_image_name()
        super().__init__(wdir, job_limit)

    @staticmethod
    def _is_available() -> bool:
        return FOAMRuntime._docker_available()

    def _submit_job(
        self,
        job_name: str,
        submission_cwd: Path,
        script: Path,
        script_args: dict[str, Any] = {},
    ) -> Optional[str]:
        get_runtime()._ensure_docker_image()

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            job_name,
        ]
        # Run as host user so bind-mounted files aren't owned by root
        if os.name != "nt":
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        cmd.extend(
            [
                "-v",
                f"{submission_cwd}:/work",
                "-w",
                "/work",
            ]
        )
        for key, value in script_args.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._docker_image, "bash", f"./{script.name}"])

        result = subprocess.run(cmd, stdout=PIPE, stderr=PIPE, text=True)
        if result.returncode != 0:
            logging.error(f"Docker run failed: {result.stderr}")
            return None

        return result.stdout.strip()

    def _cancel_job(self, job_id: str) -> bool:
        try:
            result = subprocess.run(
                ["docker", "stop", "-t", "10", job_id],
                stdout=PIPE,
                stderr=PIPE,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "kill", job_id],
                stdout=PIPE,
                stderr=PIPE,
                timeout=10,
            )
            result = None

        subprocess.run(["docker", "rm", job_id], stdout=PIPE, stderr=PIPE, timeout=10)
        return result is not None and result.returncode == 0

    def _job_has_finished(self, job_id: str) -> bool:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", job_id],
                stdout=PIPE,
                stderr=PIPE,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False

        if result.returncode != 0:
            return True  # Container gone = finished

        if result.stdout.strip() == "true":
            return False

        # Container stopped — clean it up
        subprocess.run(["docker", "rm", job_id], stdout=PIPE, stderr=PIPE, timeout=10)
        return True
