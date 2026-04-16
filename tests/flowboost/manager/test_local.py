from pathlib import Path
from unittest.mock import MagicMock, patch

from flowboost.manager.interfaces.local import Local


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
