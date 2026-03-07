import os
from pathlib import Path
from unittest.mock import patch

import pytest

from flowboost.openfoam.runtime import FoamRuntime, get_runtime, reset_runtime


class TestDetectMode:
    def test_native_when_foam_inst_dir_set(self):
        with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam", "FLOWBOOST_FOAM_MODE": "auto"}):
            runtime = FoamRuntime()
            assert runtime.mode == FoamRuntime.Mode.NATIVE

    def test_forced_native(self):
        with patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "native"}, clear=False):
            runtime = FoamRuntime()
            assert runtime.mode == FoamRuntime.Mode.NATIVE

    def test_forced_docker_without_docker_raises(self):
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "docker"}, clear=False),
            patch.object(FoamRuntime, "_docker_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="Docker is not available"):
                FoamRuntime()

    def test_forced_docker_with_docker(self):
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "docker"}, clear=False),
            patch.object(FoamRuntime, "_docker_available", return_value=True),
        ):
            runtime = FoamRuntime()
            assert runtime.mode == FoamRuntime.Mode.DOCKER

    def test_auto_falls_to_docker_when_no_foam(self):
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "auto"}, clear=False),
            patch.object(FoamRuntime, "_docker_available", return_value=True),
        ):
            os.environ.pop("FOAM_INST_DIR", None)
            runtime = FoamRuntime()
            assert runtime.mode == FoamRuntime.Mode.DOCKER

    def test_auto_raises_when_nothing_available(self):
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "auto"}, clear=False),
            patch.object(FoamRuntime, "_docker_available", return_value=False),
        ):
            os.environ.pop("FOAM_INST_DIR", None)
            with pytest.raises(RuntimeError, match="OpenFOAM not available"):
                FoamRuntime()


class TestTranslatePath:
    def test_relative_path_unchanged(self):
        with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
            runtime = FoamRuntime()
        assert runtime._translate_path("constant/U") == "constant/U"

    def test_absolute_under_mount(self):
        with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
            runtime = FoamRuntime()
        runtime.add_mount(Path("/tmp/cases"), "/work")
        result = runtime._translate_path("/tmp/cases/mycase/0/U")
        assert result == "/work/mycase/0/U"

    def test_absolute_not_under_mount_passes_through(self):
        with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
            runtime = FoamRuntime()
        runtime.add_mount(Path("/tmp/cases"), "/work")
        result = runtime._translate_path("/opt/openfoam/tutorials/X/Y")
        assert result == "/opt/openfoam/tutorials/X/Y"


class TestTranslateCommand:
    def test_foam_command_args_translated(self):
        with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
            runtime = FoamRuntime()
        runtime.add_mount(Path("/tmp/cases"), "/work")

        cmd = ["foamDictionary", "/tmp/cases/mycase/constant/U", "-entry", "inlet"]
        translated_cmd, translated_cwd = runtime._translate_command(cmd, None)

        assert translated_cmd[0] == "foamDictionary"
        assert translated_cmd[1] == "/work/mycase/constant/U"
        assert translated_cmd[2] == "-entry"
        assert translated_cmd[3] == "inlet"

    def test_cwd_translated(self):
        with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
            runtime = FoamRuntime()
        runtime.add_mount(Path("/tmp/cases"), "/work")

        _, translated_cwd = runtime._translate_command(
            ["listTimes"], Path("/tmp/cases/mycase")
        )
        assert translated_cwd == "/work/mycase"


class TestRunRouting:
    def test_non_foam_command_runs_natively(self):
        """Non-FOAM commands should bypass Docker even in DOCKER mode."""
        with (
            patch.dict(os.environ, {"FLOWBOOST_FOAM_MODE": "docker"}, clear=False),
            patch.object(FoamRuntime, "_docker_available", return_value=True),
        ):
            runtime = FoamRuntime()

        result = runtime.run(["echo", "hello"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_foam_command_detected(self):
        """FOAM commands should be recognized by name."""
        assert "foamDictionary" in FoamRuntime.FOAM_COMMANDS
        assert "foamCloneCase" in FoamRuntime.FOAM_COMMANDS
        assert "echo" not in FoamRuntime.FOAM_COMMANDS


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

    def test_reset_runtime_clears(self):
        import flowboost.openfoam.runtime as rt
        saved = rt._runtime
        rt._runtime = None
        try:
            with patch.dict(os.environ, {"FOAM_INST_DIR": "/opt/openfoam"}):
                r1 = get_runtime()
                # Use module-level assignment instead of reset_runtime()
                # to avoid stopping a real container
                rt._runtime = None
                r2 = get_runtime()
                assert r1 is not r2
        finally:
            rt._runtime = saved
