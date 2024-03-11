import json
import logging
from pathlib import Path
from typing import Any

import pytest

from flowboost.manager.manager import JobV2, Manager
from flowboost.openfoam.case import Case


class MockManager(Manager):
    @staticmethod
    def _is_available() -> bool:
        return True

    def _submit_job(
        self,
        job_name: str,
        submission_cwd: Path,
        script: Path,
        script_args: dict[str, Any] = {},
    ) -> str | None:
        return "123"

    def _cancel_job(self, job_id: str) -> bool:
        return True

    def submit_case(self, case: Case, script_args: dict = {}) -> bool:
        return True

    def cancel_job(self, job: JobV2) -> bool:
        return True

    def get_running_jobs(self) -> list[str]:
        return []

    def _job_has_finished(self, job_id: str) -> bool:
        # For testing, let's assume jobs with "finished" in their name are finished
        for j in self.job_pool:
            if job_id == j.id and "finished" in j.name:
                return True

        return False


@pytest.fixture
def mock_job():
    return JobV2(
        id="123",
        name="test_job",
        wdir=Path("/tmp"),
    )


@pytest.fixture
def manager(tmp_path):
    return MockManager(wdir=tmp_path, job_limit=5)


def test_job_state(mock_job: JobV2):
    state = mock_job.to_dict()
    assert state["id"] == "123"
    assert state["name"] == "test_job"
    logging.info("Job state:" + json.dumps(state, indent=4))


def test_manager_save_state(manager: Manager, mock_job: JobV2):
    manager.job_pool.add(mock_job)
    manager._save_state()

    filename = Path(manager.wdir, f"job_tracking_{manager.type}.json")
    assert filename.exists()

    with open(filename, "r") as f:
        state = json.load(f)
        assert state["type"] == "MockManager"
        assert len(state["job_pool"]) == 1
        logging.info("Manager state:" + json.dumps(state, indent=4))


def test_manager_restore_state(tmp_path, mock_job):
    # Pre-create a state file
    manager = MockManager(wdir=tmp_path, job_limit=5)
    manager.job_pool.add(mock_job)
    manager._save_state()

    # Attempt to restore state
    manager._restore_state(wdir=manager.wdir)
    assert len(manager.job_pool) == 1
    restored_job = manager.job_pool.pop()

    assert restored_job.id == mock_job.id
    assert restored_job.name == mock_job.name
    # Further assertions to validate the restored state matches


def test_status_print(manager: Manager):
    job_running = JobV2(
        id="124",
        name="running_job",
        wdir=Path("/tmp"),
    )
    job_finished = JobV2(
        id="125",
        name="finished_job",
        wdir=Path("/tmp"),
    )

    manager.job_pool.add(job_running)
    manager.job_pool.add(job_finished)

    status_output = manager._status_print()
    print(status_output)
    assert job_running.id in status_output
    assert job_finished.id in status_output
    logging.info("Manager status print:\n" + status_output)
