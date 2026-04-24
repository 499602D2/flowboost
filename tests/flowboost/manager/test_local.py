import os
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest

from flowboost.manager.interfaces.local import Local


def _make_manager(tmp_path: Path) -> Local:
    with patch("flowboost.openfoam.interface.FOAM.in_env", return_value=True):
        return Local(wdir=tmp_path, job_limit=1)


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)

    return False


def test_submit_job_passes_script_args_as_complete_argv_entries(tmp_path):
    script = tmp_path / "Allrun"
    process = MagicMock(pid=12345)

    with (
        patch("flowboost.openfoam.interface.FOAM.in_env", return_value=True),
        patch("subprocess.Popen", return_value=process) as mock_popen,
    ):
        manager = Local(wdir=tmp_path, job_limit=1)
        job_id = manager._submit_job(
            job_name="flwbst_job01",
            submission_cwd=tmp_path,
            script=script,
            script_args={"NPROCS": 4, "SOLVER": "simpleFoam"},
        )

    assert job_id == "12345"
    assert mock_popen.call_args[0][0] == [
        "bash",
        Path(script),
        'NPROCS="4"',
        'SOLVER="simpleFoam"',
    ]


def test_cancel_job_terminates_entire_process_group(tmp_path):
    script = tmp_path / "Allrun"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -eu",
                "sleep 30 &",
                "child=$!",
                'echo "$child" > child.pid',
                'wait "$child"',
            ]
        )
        + "\n"
    )

    manager = _make_manager(tmp_path)
    real_popen = subprocess.Popen
    process = None

    def recording_popen(*args, **kwargs):
        nonlocal process
        process = real_popen(*args, **kwargs)
        return process

    child_pid_path = tmp_path / "child.pid"
    child_pid = None

    try:
        with patch("subprocess.Popen", side_effect=recording_popen):
            job_id = manager._submit_job(
                job_name="flwbst_job01",
                submission_cwd=tmp_path,
                script=script,
            )

        assert process is not None
        assert job_id is not None
        assert _wait_for(child_pid_path.exists), "Child PID file was not written"

        child_pid = int(child_pid_path.read_text().strip())
        assert psutil.pid_exists(child_pid)

        assert manager._cancel_job(job_id)
        process.wait(timeout=5)
        assert _wait_for(lambda: not psutil.pid_exists(child_pid))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        if child_pid is not None and psutil.pid_exists(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    "status",
    [
        getattr(psutil, status_name)
        for status_name in (
            "STATUS_RUNNING",
            "STATUS_SLEEPING",
            "STATUS_DISK_SLEEP",
            "STATUS_STOPPED",
        )
        if hasattr(psutil, status_name)
    ],
)
def test_job_has_finished_treats_live_process_states_as_not_finished(tmp_path, status):
    manager = _make_manager(tmp_path)
    process = MagicMock()
    process.status.return_value = status

    with patch("psutil.Process", return_value=process):
        assert manager._job_has_finished("123") is False
