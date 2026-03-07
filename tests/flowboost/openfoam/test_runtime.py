import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from flowboost.openfoam.runtime import FOAMRuntime, get_runtime, reset_runtime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def native_runtime():
    """FOAMRuntime in native mode (no Docker needed)."""
    with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
        yield FOAMRuntime()


@pytest.fixture
def docker_runtime():
    """FOAMRuntime in docker mode (Docker availability faked)."""
    with (
        patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "docker"}, clear=False),
        patch.object(FOAMRuntime, "_docker_available", return_value=True),
    ):
        yield FOAMRuntime()


@pytest.fixture
def docker_runtime_no_lifecycle(docker_runtime):
    """Docker runtime with container lifecycle methods mocked out.

    Tests can assert on the mocks via ``runtime._ensure_container`` and
    ``runtime._stop_container`` (they are MagicMock instances while the
    fixture is active).
    """
    with (
        patch.object(docker_runtime, "_ensure_container"),
        patch.object(docker_runtime, "_stop_container"),
    ):
        yield docker_runtime


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


class TestDetectMode:
    def test_native_when_foam_inst_dir_set(self):
        with patch.dict(
            os.environ,
            {"FOAM_INST_DIR": "/opt/openfoam", "FLOWBOOST_FOAM_MODE": "auto"},
        ):
            runtime = FOAMRuntime()
            assert runtime.mode == FOAMRuntime.Mode.NATIVE

    def test_forced_native(self):
        with patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "native"}, clear=False):
            runtime = FOAMRuntime()
            assert runtime.mode == FOAMRuntime.Mode.NATIVE

    def test_forced_docker_without_docker_raises(self):
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "docker"}, clear=False),
            patch.object(FOAMRuntime, "_docker_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="Docker is not available"):
                FOAMRuntime()

    def test_forced_docker_with_docker(self, docker_runtime):
        assert docker_runtime.mode == FOAMRuntime.Mode.DOCKER

    def test_auto_falls_to_docker_when_no_foam(self):
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "auto"}, clear=False),
            patch.object(FOAMRuntime, "_docker_available", return_value=True),
        ):
            os.environ.pop("FOAM_INST_DIR", None)
            runtime = FOAMRuntime()
            assert runtime.mode == FOAMRuntime.Mode.DOCKER

    def test_auto_raises_when_nothing_available(self):
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "auto"}, clear=False),
            patch.object(FOAMRuntime, "_docker_available", return_value=False),
        ):
            os.environ.pop("FOAM_INST_DIR", None)
            with pytest.raises(RuntimeError, match="OpenFOAM not available"):
                FOAMRuntime()

    def test_auto_prefers_native_when_both_available(self):
        with (
            patch.dict(
                os.environ,
                {"FOAM_INST_DIR": "/opt/openfoam", "FLOWBOOST_FOAM_MODE": "auto"},
            ),
            patch.object(FOAMRuntime, "_docker_available", return_value=True),
        ):
            runtime = FOAMRuntime()
            assert runtime.mode == FOAMRuntime.Mode.NATIVE

    def test_custom_docker_image_from_env(self):
        with patch.dict(
            os.environ,
            {"FLOWBOOST_FOAM_IMAGE": "custom/foam:latest", "FOAM_INST_DIR": "/opt"},
        ):
            runtime = FOAMRuntime()
            assert runtime._docker_image == "custom/foam:latest"


# ---------------------------------------------------------------------------
# Path translation
# ---------------------------------------------------------------------------


class TestTranslatePath:
    def test_relative_path_unchanged(self, native_runtime):
        assert native_runtime._translate_path("constant/U") == "constant/U"

    def test_absolute_under_mount(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/cases"), "/work")
        result = native_runtime._translate_path("/tmp/cases/mycase/0/U")
        assert result == "/work/mycase/0/U"

    def test_absolute_not_under_mount_passes_through(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/cases"), "/work")
        result = native_runtime._translate_path("/opt/openfoam/tutorials/X/Y")
        assert result == "/opt/openfoam/tutorials/X/Y"

    def test_first_matching_mount_wins(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/cases"), "/work")
        native_runtime.add_mount(Path("/tmp/cases/sub"), "/work1")
        result = native_runtime._translate_path("/tmp/cases/sub/mycase/0/U")
        assert result == "/work/sub/mycase/0/U"

    def test_second_mount_matches_when_first_doesnt(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/a"), "/work")
        native_runtime.add_mount(Path("/tmp/b"), "/work1")
        result = native_runtime._translate_path("/tmp/b/mycase/0/U")
        assert result == "/work1/mycase/0/U"


class TestTranslateCommand:
    def test_foam_command_args_translated(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/cases"), "/work")
        cmd = ["foamDictionary", "/tmp/cases/mycase/constant/U", "-entry", "inlet"]
        translated_cmd, _ = native_runtime._translate_command(cmd, None)

        assert translated_cmd == [
            "foamDictionary", "/work/mycase/constant/U", "-entry", "inlet"
        ]

    def test_cwd_translated(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/cases"), "/work")
        _, translated_cwd = native_runtime._translate_command(
            ["listTimes"], Path("/tmp/cases/mycase")
        )
        assert translated_cwd == "/work/mycase"


# ---------------------------------------------------------------------------
# Run routing
# ---------------------------------------------------------------------------


class TestRunRouting:
    def test_non_foam_command_runs_natively(self, docker_runtime):
        result = docker_runtime.run(["echo", "hello"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_foam_command_detected(self):
        assert "foamDictionary" in FOAMRuntime.FOAM_COMMANDS
        assert "foamCloneCase" in FOAMRuntime.FOAM_COMMANDS
        assert "echo" not in FOAMRuntime.FOAM_COMMANDS

    def test_foam_command_via_full_path_detected(self, docker_runtime):
        with patch.object(docker_runtime, "_docker_exec") as mock_exec:
            mock_exec.return_value = subprocess.CompletedProcess([], 0, "", "")
            docker_runtime.run(["/usr/bin/foamDictionary", "constant/U"])
            mock_exec.assert_called_once()

    def test_foam_command_routes_to_docker_exec(self, docker_runtime):
        with patch.object(docker_runtime, "_docker_exec") as mock_exec:
            mock_exec.return_value = subprocess.CompletedProcess([], 0, "", "")
            docker_runtime.run(["foamDictionary", "constant/U"])
            mock_exec.assert_called_once_with(
                ["foamDictionary", "constant/U"], None
            )


# ---------------------------------------------------------------------------
# Mount management
# ---------------------------------------------------------------------------


class TestAddMount:
    def test_after_container_start_raises(self, native_runtime):
        native_runtime._container_id = "abc123"
        with pytest.raises(RuntimeError, match="Cannot add mounts"):
            native_runtime.add_mount(Path("/tmp/data"))

    def test_duplicate_guest_path_raises(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/a"), "/work")
        with pytest.raises(RuntimeError, match="already in use"):
            native_runtime.add_mount(Path("/tmp/b"), "/work")

    def test_different_guest_paths_allowed(self, native_runtime):
        native_runtime.add_mount(Path("/tmp/a"), "/work")
        native_runtime.add_mount(Path("/tmp/b"), "/work1")
        assert len(native_runtime._mounts) == 2


class TestIsMounted:
    def test_path_under_mount(self, native_runtime):
        native_runtime._mounts = [(Path("/tmp/cases").resolve(), "/work")]
        assert native_runtime._is_mounted(Path("/tmp/cases/sub/dir"))

    def test_path_not_under_any_mount(self, native_runtime):
        native_runtime._mounts = [(Path("/tmp/cases").resolve(), "/work")]
        assert not native_runtime._is_mounted(Path("/var/data"))

    def test_exact_mount_root_is_mounted(self, native_runtime):
        mount_root = Path("/tmp/cases").resolve()
        native_runtime._mounts = [(mount_root, "/work")]
        assert native_runtime._is_mounted(Path("/tmp/cases"))

    def test_symlinked_path_resolves(self, native_runtime, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real_dir)

        native_runtime._mounts = [(tmp_path.resolve(), "/work")]
        assert native_runtime._is_mounted(link)


# ---------------------------------------------------------------------------
# Auto-mount
# ---------------------------------------------------------------------------


class TestAutoMount:
    def test_discovers_host_path_from_command_arg(self, docker_runtime, tmp_path):
        case_dir = tmp_path / "cases" / "mycase"
        case_dir.mkdir(parents=True)
        u_file = case_dir / "0" / "U"
        u_file.parent.mkdir()
        u_file.touch()

        docker_runtime._auto_mount(["foamDictionary", str(u_file)], None)

        assert len(docker_runtime._mounts) == 1
        host_root, guest_path = docker_runtime._mounts[0]
        assert u_file.parent.resolve().is_relative_to(host_root)
        assert guest_path == "/work"

    def test_discovers_cwd(self, docker_runtime, tmp_path):
        case_dir = tmp_path / "cases" / "mycase"
        case_dir.mkdir(parents=True)

        docker_runtime._auto_mount(["foamCleanCase"], case_dir)

        assert len(docker_runtime._mounts) == 1
        host_root, _ = docker_runtime._mounts[0]
        assert case_dir.resolve().is_relative_to(host_root)

    def test_skips_nonexistent_host_paths(self, docker_runtime):
        docker_runtime._auto_mount(
            ["foamDictionary", "/opt/openfoam13/tutorials/cavity/0/U"], None
        )
        assert len(docker_runtime._mounts) == 0

    def test_skips_already_covered_paths(self, docker_runtime, tmp_path):
        docker_runtime._mounts = [(tmp_path.resolve(), "/work")]
        subdir = tmp_path / "case_0"
        subdir.mkdir()

        docker_runtime._auto_mount(["foamCleanCase"], subdir)
        assert len(docker_runtime._mounts) == 1

    def test_finds_common_ancestor(self, docker_runtime, tmp_path):
        case_a = tmp_path / "cases" / "a" / "0"
        case_b = tmp_path / "cases" / "b" / "0"
        case_a.mkdir(parents=True)
        case_b.mkdir(parents=True)
        (case_a / "U").touch()
        (case_b / "U").touch()

        docker_runtime._auto_mount(
            ["foamDictionary", str(case_a / "U"), str(case_b / "U")], None
        )

        assert len(docker_runtime._mounts) == 1
        host_root, _ = docker_runtime._mounts[0]
        assert case_a.resolve().is_relative_to(host_root)
        assert case_b.resolve().is_relative_to(host_root)

    def test_raises_when_common_ancestor_too_broad(self, docker_runtime):
        with patch.object(docker_runtime, "_is_mounted", return_value=False):
            with pytest.raises(RuntimeError, match="too broad"):
                docker_runtime._auto_mount(
                    ["foamDictionary", "/tmp/x", "/var/y"], None
                )

    def test_guest_path_increments(self, docker_runtime, tmp_path):
        docker_runtime._mounts = [(Path("/existing"), "/work")]
        case_dir = tmp_path / "cases"
        case_dir.mkdir()

        docker_runtime._auto_mount(["foamCleanCase"], case_dir)
        assert len(docker_runtime._mounts) == 2
        _, guest = docker_runtime._mounts[1]
        assert guest == "/work1"

    def test_restarts_container_for_new_mount(self, docker_runtime, tmp_path):
        docker_runtime._container_id = "existing_container"
        case_dir = tmp_path / "newcases"
        case_dir.mkdir()

        with patch.object(docker_runtime, "_stop_container") as mock_stop:
            docker_runtime._auto_mount(["foamCleanCase"], case_dir)
            mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# Docker exec
# ---------------------------------------------------------------------------


class TestDockerExec:
    def test_shell_command_properly_quoted(self, docker_runtime, tmp_path):
        docker_runtime._container_id = "test_container"
        docker_runtime._mounts = [(tmp_path.resolve(), "/work")]
        case_dir = tmp_path / "my case"
        case_dir.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
            docker_runtime._docker_exec(["foamCleanCase"], case_dir)

            args = mock_run.call_args[0][0]
            assert args[:3] == ["docker", "exec", "test_container"]
            assert args[3:5] == ["bash", "-c"]
            shell_cmd = args[5]
            assert "cd" in shell_cmd
            assert "foamCleanCase" in shell_cmd


# ---------------------------------------------------------------------------
# Container context manager
# ---------------------------------------------------------------------------


class TestContainer:
    def test_starts_and_stops(self, docker_runtime_no_lifecycle):
        rt = docker_runtime_no_lifecycle
        with rt.container():
            rt._ensure_container.assert_called_once()
        rt._stop_container.assert_called_once()

    def test_registers_mounts(self, docker_runtime_no_lifecycle):
        rt = docker_runtime_no_lifecycle
        mount_a = Path("/tmp/work_a")
        mount_b = Path("/tmp/work_b")

        with rt.container(mount_a, mount_b):
            assert len(rt._mounts) == 2
            host_roots = [m[0] for m in rt._mounts]
            assert mount_a.resolve() in host_roots
            assert mount_b.resolve() in host_roots

    def test_noop_in_native_mode(self, native_runtime):
        with (
            patch.object(native_runtime, "_ensure_container") as mock_ensure,
            patch.object(native_runtime, "_stop_container") as mock_stop,
        ):
            with native_runtime.container() as rt:
                assert rt is native_runtime
                mock_ensure.assert_not_called()
            mock_stop.assert_called_once()

    def test_stops_on_exception(self, docker_runtime_no_lifecycle):
        rt = docker_runtime_no_lifecycle
        with pytest.raises(ValueError):
            with rt.container():
                raise ValueError("boom")
        rt._stop_container.assert_called_once()

    def test_restores_mounts_on_exit(self, docker_runtime_no_lifecycle):
        rt = docker_runtime_no_lifecycle
        rt._mounts = [(Path("/existing"), "/work")]

        with rt.container(Path("/tmp/new")):
            assert len(rt._mounts) == 2
        assert len(rt._mounts) == 1
        assert rt._mounts[0] == (Path("/existing"), "/work")

    def test_restores_mounts_on_exception(self, docker_runtime_no_lifecycle):
        rt = docker_runtime_no_lifecycle
        rt._mounts = [(Path("/existing"), "/work")]

        with pytest.raises(ValueError):
            with rt.container(Path("/tmp/new")):
                assert len(rt._mounts) == 2
                raise ValueError("boom")
        assert len(rt._mounts) == 1
        assert rt._mounts[0] == (Path("/existing"), "/work")

    def test_nested_raises(self, docker_runtime_no_lifecycle):
        rt = docker_runtime_no_lifecycle
        with rt.container():
            with pytest.raises(RuntimeError, match="not re-entrant"):
                with rt.container():
                    pass

    def test_sequential_allowed(self, docker_runtime_no_lifecycle):
        rt = docker_runtime_no_lifecycle
        with rt.container():
            pass
        with rt.container():
            pass


# ---------------------------------------------------------------------------
# Container lifecycle internals
# ---------------------------------------------------------------------------


class TestEnsureContainer:
    def test_atexit_registered_only_once(self, docker_runtime):
        create_result = subprocess.CompletedProcess([], 0, "cid\n", "")
        start_result = subprocess.CompletedProcess([], 0, "", "")

        with (
            patch.object(docker_runtime, "_ensure_docker_image"),
            patch("subprocess.run") as mock_run,
            patch("atexit.register") as mock_atexit,
        ):
            mock_run.side_effect = [create_result, start_result]
            docker_runtime._ensure_container()
            assert mock_atexit.call_count == 1

            # Stop and recreate — atexit should not be called again
            docker_runtime._container_id = None
            mock_run.side_effect = [create_result, start_result]
            docker_runtime._ensure_container()
            assert mock_atexit.call_count == 1

    def test_cleans_up_on_start_failure(self, docker_runtime):
        create_result = subprocess.CompletedProcess([], 0, "container_id\n", "")
        start_result = subprocess.CompletedProcess([], 1, "", "start failed")
        rm_result = subprocess.CompletedProcess([], 0, "", "")

        with (
            patch.object(docker_runtime, "_ensure_docker_image"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [create_result, start_result, rm_result]
            with pytest.raises(RuntimeError, match="Failed to start"):
                docker_runtime._ensure_container()

            rm_call = mock_run.call_args_list[2]
            assert rm_call[0][0] == ["docker", "rm", "container_id"]
            assert docker_runtime._container_id is None

    def test_skips_if_already_running(self, docker_runtime):
        docker_runtime._container_id = "already_running"
        with patch.object(docker_runtime, "_ensure_docker_image") as mock_image:
            docker_runtime._ensure_container()
            mock_image.assert_not_called()


class TestStopContainer:
    def test_noop_when_no_container(self, docker_runtime):
        docker_runtime._stop_container()  # should not raise

    def test_clears_container_id(self, docker_runtime):
        docker_runtime._container_id = "test_id"
        with patch("subprocess.run"):
            docker_runtime._stop_container()
        assert docker_runtime._container_id is None

    def test_force_kills_on_stop_timeout(self, docker_runtime):
        docker_runtime._container_id = "test_id"
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.TimeoutExpired("docker stop", 15),
                subprocess.CompletedProcess([], 0, "", ""),
            ]
            docker_runtime._stop_container()

        assert docker_runtime._container_id is None
        kill_call = mock_run.call_args_list[1]
        assert "kill" in kill_call[0][0]

    def test_never_raises_on_double_timeout(self, docker_runtime):
        docker_runtime._container_id = "test_id"
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.TimeoutExpired("docker stop", 15),
                subprocess.TimeoutExpired("docker kill", 10),
            ]
            docker_runtime._stop_container()
        assert docker_runtime._container_id is None

    def test_never_raises_on_oserror(self, docker_runtime):
        docker_runtime._container_id = "test_id"
        with patch("subprocess.run", side_effect=OSError("docker not found")):
            docker_runtime._stop_container()
        assert docker_runtime._container_id is None


class TestEnsureDockerImage:
    def test_skips_build_when_image_exists(self, docker_runtime):
        with (
            patch.object(docker_runtime, "_docker_image_available", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            docker_runtime._ensure_docker_image()
            mock_run.assert_not_called()

    def test_raises_when_no_image_and_no_dockerfile(self, docker_runtime):
        with (
            patch.object(docker_runtime, "_docker_image_available", return_value=False),
            patch("flowboost.openfoam.runtime.DOCKERFILE_DIR", Path("/nonexistent")),
        ):
            with pytest.raises(RuntimeError, match="not found"):
                docker_runtime._ensure_docker_image()

    def test_raises_on_build_failure(self, docker_runtime, tmp_path):
        with (
            patch.object(docker_runtime, "_docker_image_available", return_value=False),
            patch("flowboost.openfoam.runtime.DOCKERFILE_DIR", tmp_path),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess(
                [], 1, "", "build error details"
            )
            with pytest.raises(RuntimeError, match="build error details"):
                docker_runtime._ensure_docker_image()


# ---------------------------------------------------------------------------
# File transfer & FOAM queries
# ---------------------------------------------------------------------------


class TestTransferFile:
    def test_native_mode_copies_file(self, native_runtime, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("hello")
        dst = tmp_path / "dest.txt"

        native_runtime.transfer_file(str(src), dst)
        assert dst.read_text() == "hello"

    def test_docker_mode_raises_on_failure(self, docker_runtime):
        docker_runtime._container_id = "test_container"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 1, "", "no such file"
            )
            with pytest.raises(RuntimeError, match="Failed to transfer"):
                docker_runtime.transfer_file("/container/path", Path("/local/path"))


class TestFoamTutorialsPath:
    def test_raises_in_native_mode(self, native_runtime):
        with pytest.raises(RuntimeError, match="native mode"):
            native_runtime._foam_tutorials_path()

    def test_caches_result(self, docker_runtime):
        docker_runtime._container_id = "test_container"
        result = subprocess.CompletedProcess([], 0, "/opt/openfoam/tutorials\n", "")

        with patch("subprocess.run", return_value=result) as mock_run:
            path1 = docker_runtime._foam_tutorials_path()
            path2 = docker_runtime._foam_tutorials_path()

        assert path1 == path2 == "/opt/openfoam/tutorials"
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_runtime_returns_same_instance(self):
        import flowboost.openfoam.runtime as rt

        saved = rt._runtime
        rt._runtime = None
        try:
            with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
                r1 = get_runtime()
                r2 = get_runtime()
                assert r1 is r2
        finally:
            rt._runtime = saved

    def test_reset_runtime_clears_and_cleans_up(self):
        import flowboost.openfoam.runtime as rt

        saved = rt._runtime
        rt._runtime = None
        try:
            with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
                r1 = get_runtime()
                with patch.object(r1, "cleanup") as mock_cleanup:
                    reset_runtime()
                    mock_cleanup.assert_called_once()
                assert rt._runtime is None
                r2 = get_runtime()
                assert r1 is not r2
        finally:
            rt._runtime = saved
