import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from flowboost.openfoam.runtime import FoamRuntime, get_runtime


def openfoam_in_env(func):
    def wrapper(*args, **kwargs):
        if not FOAM.in_env():
            raise ValueError("OpenFOAM not sourced")
        return func(*args, **kwargs)

    return wrapper


class FOAM:
    @staticmethod
    def source(path: str):
        if get_runtime().mode != FoamRuntime.Mode.NATIVE:
            return  # Container has its own environment

        command = f"source {path} && env"
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
        )

        if proc.stdout:
            for line in proc.stdout:
                (key, _, value) = line.decode().partition("=")
                os.environ[key] = value.strip()

        proc.communicate()

    @staticmethod
    @openfoam_in_env
    def tutorials() -> str | Path:
        runtime = get_runtime()
        if runtime.mode != FoamRuntime.Mode.NATIVE:
            return runtime.foam_tutorials_path()

        tutorials_path = os.environ.get("FOAM_TUTORIALS")
        if not tutorials_path:
            raise FileNotFoundError(f"OpenFOAM tutorials not found in {tutorials_path}")
        return Path(tutorials_path)

    @staticmethod
    def in_env(env_var: str = "FOAM_INST_DIR") -> bool:
        if os.getenv(env_var):
            return True

        # Check if Docker runtime is available and usable
        try:
            runtime = get_runtime()
            return runtime.is_available()
        except RuntimeError:
            return False

    @staticmethod
    @openfoam_in_env
    def tutorial(relative_path: str) -> str | Path:
        tutorials = FOAM.tutorials()
        runtime = get_runtime()

        if runtime.mode != FoamRuntime.Mode.NATIVE:
            # tutorials is a container-internal path string
            return f"{tutorials}/{relative_path}"

        tutorial_case = Path(tutorials) / relative_path
        if not tutorial_case.exists():
            raise FileNotFoundError(
                f"Tutorial case path does not exist: '{tutorial_case}'"
            )

        return tutorial_case.absolute()


def run_command(command: list[Any], cwd: Optional[Path] = None) -> str:
    """Executes a shell command, returning stdout. Routes FOAM commands
    through the runtime (native or Docker).

    Args:
        command (list[Any]): List of command arguments
        cwd (Optional[Path]): Directory to run command in. Defaults to None.

    Raises:
        ValueError: On non-zero return code.

    Returns:
        str: Command stdout
    """
    logging.debug(f"Executing command: {command}")
    result = get_runtime().run(command, cwd=cwd)

    if result.returncode != 0:
        raise ValueError(
            f"Command '{command}' failed (rc={result.returncode}): {result.stderr}"
        )

    logging.debug(f"Command output: {result.stdout}")
    return result.stdout


def run_foam_command(
    command: list, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Execute a command through the runtime, returning CompletedProcess
    for caller-side error handling."""
    return get_runtime().run(command, cwd=cwd)
