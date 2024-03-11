import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional


class FOAM:
    @staticmethod
    def source(path: str):
        command = f"source {path} && env"
        proc = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                shell=True,
                                executable='/bin/bash')

        if proc.stdout:
            for line in proc.stdout:
                (key, _, value) = line.decode().partition("=")
                os.environ[key] = value.strip()

        proc.communicate()

    @staticmethod
    def tutorials() -> Path:
        tutorials_path = os.environ.get("FOAM_TUTORIALS")
        if not tutorials_path:
            raise FileNotFoundError(
                f"OpenFOAM tutorials not found in {tutorials_path}")
        return Path(tutorials_path)

    @staticmethod
    def in_env(env_var: str = "FOAM_INST_DIR") -> bool:
        if os.getenv(env_var):
            return True

        return False


def run_command(command: list[Any], cwd: Optional[Path] = None) -> str:
    """ Executes a shell command in a given directory, returning the utf-8 decoded
    output.

    Args:
        command (list[Any]): List of command arguments
        cwd (Optional[Path], optional): Directory to run command in, if not current \
            directory. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        str: utf-8 decoded command output from stdout
    """
    logging.debug(f"Executing command: {command}")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True)

    if result.stderr:
        raise ValueError(
            f"Error executing command '{command}': {result.stderr}")

    logging.debug(f"Command output: {result.stdout}")
    return result.stdout
