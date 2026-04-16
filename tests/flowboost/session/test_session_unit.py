"""Unit tests for Session logic that doesn't require OpenFOAM."""

from pathlib import Path

import pytest

from flowboost.openfoam.case import Case
from flowboost.optimizer.objectives import Objective
from flowboost.session.session import Session


def _make_foam_dir(path: Path) -> Path:
    """Create a minimal directory that passes path_is_foam_dir."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "constant").mkdir(exist_ok=True)
    (path / "system").mkdir(exist_ok=True)
    return path


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
        session = self._make_session_with_cases(
            tmp_path, n_cases=3, max_evaluations=3
        )
        assert session._check_termination_criteria() is True

    def test_max_evaluations_not_reached(self, tmp_path):
        session = self._make_session_with_cases(
            tmp_path, n_cases=2, max_evaluations=5
        )
        assert session._check_termination_criteria() is False

    def test_no_criteria_returns_false(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=3)
        assert session._check_termination_criteria() is False

    def test_target_value_minimize_reached(self, tmp_path):
        session = self._make_session_with_cases(
            tmp_path, n_cases=3, target_value=1.5
        )
        # Case 0 has value=0.0 which is <= 1.5
        assert session._check_termination_criteria() is True

    def test_target_value_minimize_not_reached(self, tmp_path):
        session = self._make_session_with_cases(
            tmp_path, n_cases=1, target_value=-1.0
        )
        # Case 0 has value=0.0 which is > -1.0
        assert session._check_termination_criteria() is False

    def test_target_value_maximize(self, tmp_path):
        session = Session(
            name="test", data_dir=tmp_path / "session", target_value=1.5
        )
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
