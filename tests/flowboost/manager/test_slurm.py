from pathlib import Path
from unittest.mock import MagicMock, patch

from flowboost.manager.interfaces.slurm import Slurm


def test_is_available_requires_all_slurm_commands():
    commands = {
        "sbatch": "/usr/bin/sbatch",
        "squeue": "/usr/bin/squeue",
        "scancel": "/usr/bin/scancel",
        "scontrol": "/usr/bin/scontrol",
    }

    with patch("shutil.which", side_effect=commands.get):
        assert Slurm._is_available()


def test_is_available_false_when_command_missing():
    commands = {
        "sbatch": "/usr/bin/sbatch",
        "squeue": "/usr/bin/squeue",
        "scancel": None,
        "scontrol": "/usr/bin/scontrol",
    }

    with patch("shutil.which", side_effect=commands.get):
        assert not Slurm._is_available()


def test_submit_job_passes_script_args_as_sbatch_exports(tmp_path):
    mock_result = MagicMock(returncode=0, stdout="Submitted batch job 123\n")
    script = Path("/tmp/cases/mycase/Allrun")

    with (
        patch.object(Slurm, "_is_available", return_value=True),
        patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        manager = Slurm(wdir=tmp_path, job_limit=1)
        job_id = manager._submit_job(
            job_name="flwbst_job01",
            submission_cwd=tmp_path,
            script=script,
            script_args={"NPROCS": 4, "SOLVER": "simpleFoam"},
        )

    assert job_id == "123"
    assert mock_run.call_args[0][0] == [
        "sbatch",
        "--job-name",
        "flwbst_job01",
        "--export",
        'ALL,NPROCS="4",SOLVER="simpleFoam"',
        str(script),
    ]


def test_job_has_finished_returns_true_when_job_leaves_queue(tmp_path):
    invalid_job = MagicMock(
        returncode=1,
        stderr="slurm_load_jobs error: Invalid job id specified\n",
    )

    with (
        patch.object(Slurm, "_is_available", return_value=True),
        patch("subprocess.run", return_value=invalid_job),
    ):
        manager = Slurm(wdir=tmp_path, job_limit=1)
        assert manager._job_has_finished("123")
