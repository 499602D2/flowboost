import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flowboost.manager.interfaces.docker_local import DockerLocal
from flowboost.openfoam.runtime import FOAMRuntime


@pytest.fixture
def manager(tmp_path):
    with patch.object(FOAMRuntime, "_docker_available", return_value=True):
        return DockerLocal(wdir=tmp_path, job_limit=2)


class TestIsAvailable:
    def test_available_when_docker_present(self):
        with patch.object(FOAMRuntime, "_docker_available", return_value=True):
            assert DockerLocal._is_available()

    def test_unavailable_when_no_docker(self):
        with patch.object(FOAMRuntime, "_docker_available", return_value=False):
            assert not DockerLocal._is_available()


class TestSubmitJob:
    def test_submit_builds_correct_command(self, manager):
        mock_result = MagicMock(returncode=0, stdout="abc123containerid\n")
        script = Path("/tmp/cases/mycase/Allrun")
        cwd = Path("/tmp/cases/mycase")

        with (
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch("flowboost.manager.interfaces.docker_local.get_runtime") as mock_rt,
        ):
            job_id = manager._submit_job("flwbst_job01", cwd, script)

        assert job_id == "abc123containerid"
        mock_rt()._ensure_docker_image.assert_called_once()

        call_args = mock_run.call_args[0][0]
        assert call_args[:3] == ["docker", "run", "-d"]
        assert "--name" in call_args
        assert call_args[call_args.index("--name") + 1] == "flwbst_job01"
        assert f"{cwd}:/work" in call_args
        assert "Allrun" == call_args[-1]

    def test_submit_passes_script_args_as_env(self, manager):
        mock_result = MagicMock(returncode=0, stdout="containerid\n")

        with (
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch("flowboost.manager.interfaces.docker_local.get_runtime"),
        ):
            manager._submit_job(
                "job", Path("/work"), Path("/work/Allrun"),
                script_args={"NPROCS": "4", "SOLVER": "simpleFoam"},
            )

        call_args = mock_run.call_args[0][0]
        env_pairs = []
        for i, arg in enumerate(call_args):
            if arg == "-e":
                env_pairs.append(call_args[i + 1])
        assert "NPROCS=4" in env_pairs
        assert "SOLVER=simpleFoam" in env_pairs

    def test_submit_returns_none_on_failure(self, manager):
        mock_result = MagicMock(returncode=1, stderr="error")

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("flowboost.manager.interfaces.docker_local.get_runtime"),
        ):
            result = manager._submit_job("job", Path("/work"), Path("/work/Allrun"))

        assert result is None


class TestJobHasFinished:
    def test_running_container(self, manager):
        mock_result = MagicMock(returncode=0, stdout="true\n")

        with patch("subprocess.run", return_value=mock_result):
            assert not manager._job_has_finished("abc123")

    def test_stopped_container_is_cleaned_up(self, manager):
        mock_result = MagicMock(returncode=0, stdout="false\n")

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert manager._job_has_finished("abc123")

        # Should have called docker rm
        rm_call = mock_run.call_args_list[-1]
        assert rm_call[0][0][:2] == ["docker", "rm"]

    def test_missing_container_is_finished(self, manager):
        mock_result = MagicMock(returncode=1)

        with patch("subprocess.run", return_value=mock_result):
            assert manager._job_has_finished("gone123")

    def test_timeout_returns_not_finished(self, manager):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="", timeout=10)):
            assert not manager._job_has_finished("slow123")


class TestCancelJob:
    def test_cancel_stops_and_removes(self, manager):
        mock_result = MagicMock(returncode=0)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert manager._cancel_job("abc123")

        cmds = [call[0][0][:2] for call in mock_run.call_args_list]
        assert ["docker", "stop"] in cmds
        assert ["docker", "rm"] in cmds

    def test_cancel_force_kills_on_timeout(self, manager):
        def side_effect(cmd, **kwargs):
            if cmd[1] == "stop":
                raise subprocess.TimeoutExpired(cmd="", timeout=20)
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=side_effect) as mock_run:
            result = manager._cancel_job("stuck123")

        assert not result
        cmds = [call[0][0][:2] for call in mock_run.call_args_list]
        assert ["docker", "kill"] in cmds
