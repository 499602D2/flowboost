import atexit
import logging
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from subprocess import PIPE

DOCKER_IMAGE = "flowboost/openfoam:13"
DOCKERFILE_DIR = Path(__file__).resolve().parent / "docker"


def docker_image_name() -> str:
    """Return the configured Docker image name (respects FLOWBOOST_FOAM_IMAGE)."""
    return os.environ.get("FLOWBOOST_FOAM_IMAGE", DOCKER_IMAGE)


class FOAMRuntime:
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
        "foamToVTK",
        "listTimes",
    }

    def __init__(self):
        self._docker_image = docker_image_name()
        self._cached_foam_tutorials: str | None = None
        self._mounts: list[tuple[Path, str]] = []
        self._container_id: str | None = None
        self._atexit_registered: bool = False
        self._in_container_block: bool = False
        self.mode = self._detect_mode()

    def _detect_mode(self) -> "FOAMRuntime.Mode":
        forced = os.environ.get("FLOWBOOST_FOAM_MODE", "auto")

        if forced == "native":
            return FOAMRuntime.Mode.NATIVE
        if forced == "docker":
            if not self._docker_available():
                raise RuntimeError(
                    "FLOWBOOST_FOAM_MODE=docker but Docker is not available"
                )
            return FOAMRuntime.Mode.DOCKER

        # Auto-detect: native → docker → error
        if os.environ.get("FOAM_INST_DIR"):
            return FOAMRuntime.Mode.NATIVE

        if self._docker_available():
            return FOAMRuntime.Mode.DOCKER

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
                ["docker", "info"],
                stdout=PIPE,
                stderr=PIPE,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _docker_image_available(self) -> bool:
        """Check if the configured Docker image exists locally."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self._docker_image],
                stdout=PIPE,
                stderr=PIPE,
                timeout=5,
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
            stderr=PIPE,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build Docker image: {result.stderr}")
        logging.info(f"Docker image '{self._docker_image}' built successfully")

    def is_available(self) -> bool:
        """Check if this runtime can actually execute FOAM commands."""
        if self.mode == FOAMRuntime.Mode.NATIVE:
            return True
        if self.mode == FOAMRuntime.Mode.DOCKER:
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
        # Run as host user so bind-mounted files aren't owned by root
        if os.name != "nt":
            create_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        for host_root, container_root in self._mounts:
            create_cmd.extend(["-v", f"{host_root}:{container_root}"])
        create_cmd.extend([self._docker_image, "sleep", "infinity"])

        result = subprocess.run(
            create_cmd, stdout=PIPE, stderr=PIPE, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")

        cid = result.stdout.strip()

        result = subprocess.run(
            ["docker", "start", cid],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # Clean up the created-but-not-started container
            subprocess.run(["docker", "rm", cid], stdout=PIPE, stderr=PIPE, timeout=10)
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self._container_id = cid
        logging.debug(f"Started persistent container {cid[:12]}")

        if not self._atexit_registered:
            atexit.register(self._stop_container)
            self._atexit_registered = True

    def _stop_container(self):
        """Stop and remove the persistent container. Never raises."""
        if not self._container_id:
            return

        cid = self._container_id
        self._container_id = None

        try:
            subprocess.run(
                ["docker", "stop", "-t", "5", cid],
                stdout=PIPE,
                stderr=PIPE,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    ["docker", "kill", cid],
                    stdout=PIPE,
                    stderr=PIPE,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, OSError):
                logging.warning(f"Failed to kill container {cid[:12]}")
        except OSError:
            logging.warning(f"Failed to stop container {cid[:12]}")

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
        existing_guests = {g for _, g in self._mounts}
        if guest_path in existing_guests:
            raise RuntimeError(
                f"Mount guest path '{guest_path}' is already in use. "
                "Use a different guest_path to avoid shadowing."
            )
        self._mounts.append((host_path.resolve(), guest_path))

    def _auto_mount(self, command: list, cwd: Path | None):
        """Auto-register mounts for host paths not covered by existing mounts.

        Collects absolute host paths from command args and cwd, finds their
        common ancestor, and registers it as a mount. Restarts the container
        if it's already running.
        """
        host_dirs = []
        if cwd:
            host_dirs.append(Path(cwd).resolve())
        for arg in command[1:]:
            p = Path(arg)
            if p.is_absolute() and (p.exists() or p.parent.exists()):
                # Use parent dir — the arg may be a file or non-existent target
                host_dirs.append(p.resolve().parent)

        # Filter to paths not already covered by a mount
        uncovered = [p for p in host_dirs if not self._is_mounted(p)]
        if not uncovered:
            return

        # Find common parent of all uncovered paths
        common = uncovered[0]
        for p in uncovered[1:]:
            while not p.is_relative_to(common):
                common = common.parent

        if len(common.parts) <= 2:
            raise RuntimeError(
                f"Auto-mount would mount '{common}' which is too broad. "
                f"Pre-register mounts with container() or add_mount() instead. "
                f"Uncovered paths: {uncovered}"
            )

        mount_index = len(self._mounts)
        guest_path = f"/work{mount_index}" if mount_index > 0 else "/work"

        if self._container_id:
            logging.debug(f"New mount needed for {common} — restarting container")
            self._stop_container()

        self._mounts.append((common, guest_path))
        logging.debug(f"Auto-mounted {common} → {guest_path}")

    def _is_mounted(self, path: Path) -> bool:
        """Check if a host path is covered by an existing mount."""
        resolved = path.resolve()
        return any(resolved.is_relative_to(host_root) for host_root, _ in self._mounts)

    @contextmanager
    def container(self, *mounts: Path):
        """Context manager that runs a Docker container for the block's duration.

        Not re-entrant — if a container is already running via ``container()``,
        inner functions should just call ``run()`` directly.

        Args:
            *mounts: Host directories to bind-mount. Pre-registering a parent
                directory (e.g. the workdir) avoids container restarts when
                iterating over subdirectories.
        """
        if self._in_container_block:
            raise RuntimeError(
                "container() is not re-entrant — a container is already running. "
                "Inner functions should use run() directly."
            )
        prev_mounts = self._mounts.copy()
        for m in mounts:
            mount_index = len(self._mounts)
            guest = f"/work{mount_index}" if mount_index > 0 else "/work"
            self.add_mount(m, guest)
        if self.mode == FOAMRuntime.Mode.DOCKER:
            self._ensure_container()
        self._in_container_block = True
        try:
            yield self
        finally:
            self._in_container_block = False
            self._stop_container()
            self._mounts = prev_mounts

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

        if self.mode == FOAMRuntime.Mode.NATIVE or cmd_name not in self.FOAM_COMMANDS:
            return subprocess.run(command, stdout=PIPE, stderr=PIPE, cwd=cwd, text=True)

        return self._docker_exec(command, cwd)

    def _docker_exec(
        self, command: list, cwd: Path | None
    ) -> subprocess.CompletedProcess:
        self._auto_mount(command, cwd)
        self._ensure_container()
        translated_cmd, translated_cwd = self._translate_command(command, cwd)

        # Build shell command with proper quoting
        shell_cmd = " ".join(shlex.quote(str(c)) for c in translated_cmd)
        if translated_cwd:
            shell_cmd = f"cd {shlex.quote(translated_cwd)} && {shell_cmd}"

        docker_cmd = [
            "docker",
            "exec",
            self._container_id,
            "bash",
            "-c",
            shell_cmd,
        ]

        logging.debug(f"Docker exec: {shell_cmd}")
        return subprocess.run(docker_cmd, stdout=PIPE, stderr=PIPE, text=True)

    # ------------------------------------------------------------------
    # OpenFOAM environment queries
    # ------------------------------------------------------------------

    def _foam_tutorials_path(self) -> str:
        """Query the container for $FOAM_TUTORIALS. Returns a container-internal path.

        Only valid in Docker mode. Native mode should read FOAM_TUTORIALS
        from the environment directly (handled by FOAM.tutorials()).
        """
        if self.mode == FOAMRuntime.Mode.NATIVE:
            raise RuntimeError(
                "_foam_tutorials_path() should not be called in native mode"
            )

        if self._cached_foam_tutorials:
            return self._cached_foam_tutorials

        self._ensure_container()
        result = subprocess.run(
            [
                "docker",
                "exec",
                self._container_id,
                "bash",
                "-c",
                "echo $FOAM_TUTORIALS",
            ],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
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
        if self.mode == FOAMRuntime.Mode.NATIVE:
            shutil.copy2(remote_path, local_path)
            return

        self._ensure_container()
        result = subprocess.run(
            ["docker", "cp", f"{self._container_id}:{remote_path}", str(local_path)],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to transfer {remote_path}: {result.stderr}")

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


_runtime: FOAMRuntime | None = None


def get_runtime() -> FOAMRuntime:
    global _runtime
    if _runtime is None:
        _runtime = FOAMRuntime()
    return _runtime


def reset_runtime():
    """For testing — clear cached singleton."""
    global _runtime
    if _runtime is not None:
        _runtime.cleanup()
    _runtime = None
