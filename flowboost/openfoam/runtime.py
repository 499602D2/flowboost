import atexit
import logging
import os
import shlex
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from subprocess import PIPE

DOCKER_IMAGE = "flowboost/openfoam:13"
DOCKERFILE_DIR = Path(__file__).resolve().parent / "docker"


class FoamRuntime:
    """Decides how to execute OpenFOAM CLI commands: natively or via Docker.

    In Docker mode, a single persistent container is started on first use
    and reused for all subsequent commands via ``docker exec``. The container
    is removed on ``cleanup()``.
    """

    class Mode(Enum):
        NATIVE = "native"
        DOCKER = "docker"

    FOAM_COMMANDS = {
        "foamDictionary",
        "foamCloneCase",
        "foamCleanCase",
        "foamGet",
        "listTimes",
    }

    def __init__(self):
        self._docker_image = os.environ.get("FLOWBOOST_FOAM_IMAGE", DOCKER_IMAGE)
        self._cached_foam_tutorials: str | None = None
        self._mounts: list[tuple[Path, str]] = []
        self._container_id: str | None = None
        self.mode = self._detect_mode()

    def _detect_mode(self) -> "FoamRuntime.Mode":
        forced = os.environ.get("FLOWBOOST_FOAM_MODE", "auto")

        if forced == "native":
            return FoamRuntime.Mode.NATIVE
        if forced == "docker":
            if not self._docker_available():
                raise RuntimeError(
                    "FLOWBOOST_FOAM_MODE=docker but Docker is not available"
                )
            return FoamRuntime.Mode.DOCKER

        # Auto-detect: native → docker → error
        if os.environ.get("FOAM_INST_DIR"):
            return FoamRuntime.Mode.NATIVE

        if self._docker_available():
            return FoamRuntime.Mode.DOCKER

        raise RuntimeError(
            "OpenFOAM not available: no native install (FOAM_INST_DIR unset) "
            "and Docker is not available"
        )

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    @staticmethod
    def _docker_available() -> bool:
        try:
            result = subprocess.run(
                ["docker", "info"], stdout=PIPE, stderr=PIPE, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _docker_image_available(self) -> bool:
        """Check if the configured Docker image exists locally."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self._docker_image],
                stdout=PIPE, stderr=PIPE, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _ensure_docker_image(self):
        """Build the Docker image from the bundled Dockerfile if it doesn't exist."""
        if self._docker_image_available():
            return

        if not DOCKERFILE_DIR.is_dir():
            raise RuntimeError(
                f"Docker image '{self._docker_image}' not found and "
                f"Dockerfile directory not found at {DOCKERFILE_DIR}"
            )

        logging.info(
            f"Building Docker image '{self._docker_image}' — this is a "
            f"one-time operation..."
        )
        result = subprocess.run(
            ["docker", "build", "-t", self._docker_image, str(DOCKERFILE_DIR)],
            stderr=PIPE, text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build Docker image: {result.stderr}"
            )
        logging.info(f"Docker image '{self._docker_image}' built successfully")

    def is_available(self) -> bool:
        """Check if this runtime can actually execute FOAM commands."""
        if self.mode == FoamRuntime.Mode.NATIVE:
            return True
        if self.mode == FoamRuntime.Mode.DOCKER:
            # Available if image exists or can be built
            return self._docker_image_available() or DOCKERFILE_DIR.is_dir()
        return False

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _ensure_container(self):
        """Start the persistent container if not already running."""
        if self._container_id:
            return

        self._ensure_docker_image()

        create_cmd = ["docker", "create", "--rm", "-i"]
        for host_root, container_root in self._mounts:
            create_cmd.extend(["-v", f"{host_root}:{container_root}"])
        create_cmd.extend([self._docker_image, "sleep", "infinity"])

        result = subprocess.run(create_cmd, stdout=PIPE, stderr=PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")

        self._container_id = result.stdout.strip()

        result = subprocess.run(
            ["docker", "start", self._container_id],
            stdout=PIPE, stderr=PIPE, text=True,
        )
        if result.returncode != 0:
            self._container_id = None
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        logging.debug(f"Started persistent container {self._container_id[:12]}")
        atexit.register(self._stop_container)

    def _stop_container(self):
        """Stop and remove the persistent container."""
        if not self._container_id:
            return

        cid = self._container_id
        self._container_id = None

        try:
            subprocess.run(
                ["docker", "stop", "-t", "5", cid],
                stdout=PIPE, stderr=PIPE, timeout=15,
            )
        except subprocess.TimeoutExpired:
            # Force-kill if graceful stop hangs
            subprocess.run(
                ["docker", "kill", cid],
                stdout=PIPE, stderr=PIPE, timeout=10,
            )

        logging.debug(f"Stopped container {cid[:12]}")

    # ------------------------------------------------------------------
    # Mount management
    # ------------------------------------------------------------------

    def add_mount(self, host_path: Path, guest_path: str = "/work"):
        """Register a host directory to mount into the container.

        Must be called before the first FOAM command — mounts are set at
        container creation time.
        """
        if self._container_id:
            raise RuntimeError(
                "Cannot add mounts after the container has started. "
                "Call add_mount() before running any FOAM commands."
            )
        self._mounts.append((host_path.resolve(), guest_path))

    def cleanup(self):
        """Stop the persistent Docker container."""
        self._stop_container()

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def run(
        self, command: list, cwd: Path | None = None
    ) -> subprocess.CompletedProcess:
        """Execute a command, routing FOAM commands through Docker if needed."""
        cmd_name = Path(command[0]).name if command else ""

        if self.mode == FoamRuntime.Mode.NATIVE or cmd_name not in self.FOAM_COMMANDS:
            return subprocess.run(command, stdout=PIPE, stderr=PIPE, cwd=cwd, text=True)

        return self._docker_exec(command, cwd)

    def _docker_exec(
        self, command: list, cwd: Path | None
    ) -> subprocess.CompletedProcess:
        self._ensure_container()
        translated_cmd, translated_cwd = self._translate_command(command, cwd)

        # Build shell command with proper quoting
        shell_cmd = " ".join(shlex.quote(str(c)) for c in translated_cmd)
        if translated_cwd:
            shell_cmd = f"cd {shlex.quote(translated_cwd)} && {shell_cmd}"

        docker_cmd = [
            "docker", "exec", self._container_id,
            "bash", "-c", shell_cmd,
        ]

        logging.debug(f"Docker exec: {shell_cmd}")
        return subprocess.run(docker_cmd, stdout=PIPE, stderr=PIPE, text=True)

    # ------------------------------------------------------------------
    # OpenFOAM environment queries
    # ------------------------------------------------------------------

    def foam_tutorials_path(self) -> str:
        """Query the container for $FOAM_TUTORIALS. Returns a container-internal path.

        Only valid in Docker mode. Native mode should read FOAM_TUTORIALS
        from the environment directly (handled by FOAM.tutorials()).
        """
        assert self.mode != FoamRuntime.Mode.NATIVE, \
            "foam_tutorials_path() should not be called in native mode"

        if self._cached_foam_tutorials:
            return self._cached_foam_tutorials

        self._ensure_container()
        result = subprocess.run(
            [
                "docker", "exec", self._container_id,
                "bash", "-c", "echo $FOAM_TUTORIALS",
            ],
            stdout=PIPE, stderr=PIPE, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to query FOAM_TUTORIALS from container: {result.stderr}"
            )

        path = result.stdout.strip()
        if not path:
            raise RuntimeError("FOAM_TUTORIALS is empty inside the container")

        self._cached_foam_tutorials = path
        return path

    def transfer_file(self, remote_path: str, local_path: Path):
        """Copy a file from the container to the host."""
        if self.mode == FoamRuntime.Mode.NATIVE:
            shutil.copy2(remote_path, local_path)
            return

        self._ensure_container()
        result = subprocess.run(
            ["docker", "cp", f"{self._container_id}:{remote_path}", str(local_path)],
            stdout=PIPE, stderr=PIPE, text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to transfer {remote_path}: {result.stderr}"
            )

    # ------------------------------------------------------------------
    # Path translation
    # ------------------------------------------------------------------

    def _translate_path(self, host_path: str | Path) -> str:
        """Convert host absolute path to container path using registered mounts.

        If the path is absolute but not under any mount, assume it's already
        a container-internal path (e.g. tutorial paths) and pass through unchanged.
        """
        p = Path(host_path)
        if not p.is_absolute():
            return str(host_path)

        resolved = p.resolve()
        for host_root, guest_root in self._mounts:
            try:
                relative = resolved.relative_to(host_root)
                return str(Path(guest_root) / relative)
            except ValueError:
                continue

        # Not under any mount — assume container-internal path
        return str(host_path)

    def _translate_command(
        self, command: list, cwd: Path | None
    ) -> tuple[list[str], str | None]:
        """Translate command args and cwd for container execution."""
        translated_cmd = [str(command[0])]  # command name stays as-is
        for arg in command[1:]:
            translated_cmd.append(self._translate_path(arg))

        translated_cwd = self._translate_path(cwd) if cwd else None
        return translated_cmd, translated_cwd



_runtime: FoamRuntime | None = None


def get_runtime() -> FoamRuntime:
    global _runtime
    if _runtime is None:
        _runtime = FoamRuntime()
    return _runtime


def reset_runtime():
    """For testing — clear cached singleton."""
    global _runtime
    if _runtime is not None:
        _runtime.cleanup()
    _runtime = None
