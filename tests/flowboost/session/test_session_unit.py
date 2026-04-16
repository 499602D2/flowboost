"""Unit tests for Session logic that doesn't require OpenFOAM."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flowboost.manager.manager import JobV2
from flowboost.openfoam.case import Case
from flowboost.optimizer.backend import OptimizationComplete
from flowboost.optimizer.objectives import Objective
from flowboost.session.session import Session


def _make_foam_dir(path: Path) -> Path:
    """Create a minimal directory that passes path_is_foam_dir."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "constant").mkdir(exist_ok=True)
    (path / "system").mkdir(exist_ok=True)
    return path


class RecordingJobManager:
    def __init__(self, do_monitoring_result=(1, [], False), move_result=True):
        self.type = "RecordingJobManager"
        self.job_limit = 1
        self._do_monitoring_result = do_monitoring_result
        self._move_result = move_result
        self.submitted_cases: list[Case] = []
        self.finalized_jobs: list[JobV2] = []

    def do_monitoring(self):
        return self._do_monitoring_result

    def submit_case(self, case: Case, script_name: str | None = None) -> bool:
        self.submitted_cases.append(case)
        return True

    def free_slots(self) -> int:
        return self.job_limit

    def move_data_for_job(self, job: JobV2, dest: Path) -> bool:
        if self._move_result:
            _make_foam_dir(dest)
        return self._move_result

    def finalize_job(self, job: JobV2) -> bool:
        self.finalized_jobs.append(job)
        return True


class TestGetNextJobNumber:
    def test_no_cases_returns_1(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session")
        assert session._get_next_job_number() == 1

    def test_sequential_numbering(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session")
        # Place cases in archival dir
        _make_foam_dir(session.archival_dir / "job_00001_abc")
        _make_foam_dir(session.archival_dir / "job_00003_def")
        assert session._get_next_job_number() == 4

    def test_skips_non_job_names(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session")
        _make_foam_dir(session.archival_dir / "manual_run_001")
        _make_foam_dir(session.archival_dir / "job_00005_abc")
        assert session._get_next_job_number() == 6

    def test_skips_unparseable_numbers(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session")
        _make_foam_dir(session.archival_dir / "job_notanumber_xyz")
        _make_foam_dir(session.archival_dir / "job_00002_abc")
        assert session._get_next_job_number() == 3


class TestCheckTerminationCriteria:
    def _make_session_with_cases(self, tmp_path, n_cases, **session_kwargs):
        session = Session(name="test", data_dir=tmp_path / "session", **session_kwargs)
        obj = Objective("score", minimize=True, objective_function=lambda c: 1.0)
        session.backend.set_objectives([obj])

        for i in range(n_cases):
            case_dir = _make_foam_dir(session.archival_dir / f"job_{i:05d}_abc")
            case = Case(case_dir)
            case.success = True
            case.persist_to_file()
            case.update_metadata(
                {"score": {"value": float(i), "minimize": True}},
                entry_header="objective-outputs",
            )
            case.update_metadata(
                {"score": float(i)},
                entry_header="objective-values-raw",
            )

        return session

    def test_max_evaluations_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=3, max_evaluations=3)
        assert session._check_termination_criteria() is True

    def test_max_evaluations_not_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=2, max_evaluations=5)
        assert session._check_termination_criteria() is False

    def test_no_criteria_returns_false(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=3)
        assert session._check_termination_criteria() is False

    def test_target_value_minimize_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=3, target_value=1.5)
        # Case 0 has value=0.0 which is <= 1.5
        assert session._check_termination_criteria() is True

    def test_target_value_minimize_not_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=1, target_value=-1.0)
        # Case 0 has value=0.0 which is > -1.0
        assert session._check_termination_criteria() is False

    def test_target_value_maximize(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session", target_value=1.5)
        # batch_process re-evaluates, so the function must return the value
        # we want the termination check to see
        obj = Objective("score", minimize=False, objective_function=lambda c: 2.0)
        session.backend.set_objectives([obj])

        case_dir = _make_foam_dir(session.archival_dir / "job_00000_abc")
        case = Case(case_dir)
        case.success = True
        case.persist_to_file()
        case.update_metadata(
            {"score": {"value": 2.0, "minimize": False}},
            entry_header="objective-outputs",
        )
        case.update_metadata({"score": 2.0}, entry_header="objective-values-raw")

        # value=2.0 >= target=1.5
        assert session._check_termination_criteria() is True

    def test_missing_target_objective_returns_false(self, tmp_path):
        session = self._make_session_with_cases(
            tmp_path,
            n_cases=1,
            target_value=0.5,
            target_objective="nonexistent",
        )
        assert session._check_termination_criteria() is False


class TestPersistentOptimizationFailureHandling:
    def test_submission_failure_raises_and_leaves_case_pending(
        self, tmp_path, monkeypatch
    ):
        session = Session(name="test", data_dir=tmp_path / "session")
        case = Case(_make_foam_dir(session.pending_dir / "job_00001_abc"))
        manager = RecordingJobManager()
        manager.submit_case = MagicMock(return_value=False)
        session.job_manager = manager

        monkeypatch.setattr(session, "print_top_designs", lambda n=5: None)
        monkeypatch.setattr(session, "_check_termination_criteria", lambda: False)
        monkeypatch.setattr(
            session, "loop_optimizer_once", lambda num_new_cases: [case]
        )

        with pytest.raises(RuntimeError, match="Job submission failed"):
            session.persistent_optimization()

        assert case.path.exists()
        manager.submit_case.assert_called_once_with(
            case, script_name=session.submission_script_name
        )

    def test_move_failure_does_not_finalize_job(self, tmp_path, monkeypatch):
        session = Session(name="test", data_dir=tmp_path / "session")
        finished_job = JobV2(
            id="123",
            name="flwbst_job_00001_abc",
            wdir=session.pending_dir / "job_00001_abc",
        )
        manager = RecordingJobManager(
            do_monitoring_result=(1, [finished_job], False), move_result=False
        )
        session.job_manager = manager

        monkeypatch.setattr(session, "print_top_designs", lambda n=5: None)
        monkeypatch.setattr(session, "_check_termination_criteria", lambda: False)
        monkeypatch.setattr(
            session,
            "loop_optimizer_once",
            lambda num_new_cases: (_ for _ in ()).throw(OptimizationComplete("done")),
        )
        monkeypatch.setattr(session, "_write_designs_log", lambda: None)

        session.persistent_optimization()

        assert manager.finalized_jobs == []

    def test_successful_move_finalizes_job_after_post_evaluation_update(
        self, tmp_path, monkeypatch
    ):
        session = Session(name="test", data_dir=tmp_path / "session")
        finished_job = JobV2(
            id="123",
            name="flwbst_job_00001_abc",
            wdir=session.pending_dir / "job_00001_abc",
        )
        manager = RecordingJobManager(
            do_monitoring_result=(1, [finished_job], False), move_result=True
        )
        session.job_manager = manager
        post_updates: list[dict] = []

        monkeypatch.setattr(session, "print_top_designs", lambda n=5: None)
        monkeypatch.setattr(session, "_check_termination_criteria", lambda: False)
        monkeypatch.setattr(
            session,
            "loop_optimizer_once",
            lambda num_new_cases: (_ for _ in ()).throw(OptimizationComplete("done")),
        )
        monkeypatch.setattr(session, "_write_designs_log", lambda: None)
        monkeypatch.setattr(
            Case,
            "post_evaluation_update",
            lambda self, data: post_updates.append(data),
        )

        session.persistent_optimization()

        assert post_updates == [finished_job.to_dict()]
        assert manager.finalized_jobs == [finished_job]
