"""Unit tests for Session logic that doesn't require OpenFOAM."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flowboost.manager.manager import JobV2
from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.backend import OptimizationComplete
from flowboost.optimizer.objectives import Constraint, Objective, ScalarizedObjective
from flowboost.optimizer.search_space import Dimension
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


class TestSessionOptimizationConfiguration:
    def test_configure_optimization_delegates_to_backend(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session")
        objective = Objective("score", minimize=True, objective_function=lambda c: 1.0)
        constraint = Constraint("pressure", objective_function=lambda c: 0.5, gte=0.0)
        dimension = Dimension.range(
            name="x",
            link=DictionaryLink("constant/setup").entry("x"),
            lower=0.0,
            upper=1.0,
        )

        configured = session.configure_optimization(
            objectives=objective,
            constraints=constraint,
            search_space=dimension,
        )

        assert configured is session
        assert session.backend.objectives == [objective]
        assert session.backend.constraints == [constraint]
        assert session.backend.dimensions == [dimension]


class TestCheckTerminationCriteria:
    def _make_session_with_cases(self, tmp_path, n_cases, **session_kwargs):
        session = Session(name="test", data_dir=tmp_path / "session", **session_kwargs)
        obj = Objective(
            "score",
            minimize=True,
            objective_function=lambda c: float(
                c.read_metadata()["optimizer-suggestion"]["score"]["value"]
            ),
        )
        session.backend.set_objectives([obj])

        for i in range(n_cases):
            case_dir = _make_foam_dir(session.archival_dir / f"job_{i:05d}_abc")
            case = Case(case_dir)
            case.success = True
            case.persist_to_file()
            case.update_metadata(
                {"score": {"value": float(i)}},
                entry_header="optimizer-suggestion",
            )

        return session

    def test_max_evaluations_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=3, max_evaluations=3)
        assert session._check_termination_criteria() is True

    def test_max_evaluations_not_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=2, max_evaluations=5)
        assert session._check_termination_criteria() is False

    def test_max_evaluations_counts_failed_cases(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=1, max_evaluations=2)

        failed_case = Case(_make_foam_dir(session.archival_dir / "job_00001_failed"))
        failed_case.success = False
        failed_case.persist_to_file()

        assert session._check_termination_criteria() is True

    def test_no_criteria_returns_false(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=3)
        assert session._check_termination_criteria() is False

    def test_target_value_minimize_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=3, target_value=1.5)
        # Case 0 has value=0.0 which is <= 1.5
        assert session._check_termination_criteria() is True

    def test_target_value_minimize_not_reached(self, tmp_path):
        session = self._make_session_with_cases(tmp_path, n_cases=1, target_value=-1.0)
        # Case 0 re-evaluates to value=0.0, which is > -1.0
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


class TestWriteDesignsLog:
    def test_preserves_bounded_objective_metadata(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session")
        session.backend.set_objectives(
            [
                Objective(
                    "drag",
                    minimize=True,
                    objective_function=lambda c: 0.04,
                    gte=0.01,
                    lte=0.05,
                )
            ]
        )

        case = Case(_make_foam_dir(session.archival_dir / "job_00000_abc"))
        case.success = True
        case.persist_to_file()

        session._write_designs_log()

        log = json.loads((session.data_dir / "designs.json").read_text())
        objective = log["designs"][0]["objectives"]["drag"]
        assert objective["value"] == pytest.approx(0.04)
        assert objective["minimize"] is True
        assert objective["gte"] == pytest.approx(0.01)
        assert objective["lte"] == pytest.approx(0.05)

    def test_preserves_scalarized_component_bounds(self, tmp_path):
        session = Session(name="test", data_dir=tmp_path / "session")
        drag = Objective(
            "drag",
            minimize=True,
            objective_function=lambda c: 0.04,
            lte=0.05,
        )
        mass = Objective("mass", minimize=True, objective_function=lambda c: 1.0)
        session.backend.set_objectives(
            ScalarizedObjective(
                "ratio",
                minimize=True,
                objectives=[drag, mass],
                weights=[1.0, 1.0],
            )
        )

        case = Case(_make_foam_dir(session.archival_dir / "job_00000_abc"))
        case.success = True
        case.persist_to_file()

        session._write_designs_log()

        log = json.loads((session.data_dir / "designs.json").read_text())
        ratio = log["designs"][0]["objectives"]["ratio"]
        assert ratio["is_scalarized"] is True
        assert ratio["value"] == pytest.approx(1.04)
        assert ratio["component_bounds"] == {"drag": {"lte": 0.05}}
        assert ratio["components"]["drag"]["value"] == pytest.approx(0.04)
        assert ratio["components"]["drag"]["minimize"] is True
        assert ratio["components"]["mass"]["value"] == pytest.approx(1.0)
        assert ratio["components"]["mass"]["minimize"] is True


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

    def test_finished_job_preserves_pending_case_metadata(self, tmp_path, monkeypatch):
        session = Session(name="test", data_dir=tmp_path / "session")
        finished_job = JobV2(
            id="123",
            name="flwbst_job_00001_abc",
            wdir=session.pending_dir / "job_00001_abc",
        )
        case = Case(_make_foam_dir(finished_job.wdir))
        case._generation_index = "00001.01"
        case._based_on_case = tmp_path / "template"
        case._execution_environment = "unit-test"
        case._model_predictions_by_objective = {"drag": {"mean": 1.23, "sem": 0.1}}
        case.persist_to_file()
        case.update_metadata(
            {"x": {"value": 0.5}},
            entry_header="optimizer-suggestion",
        )

        manager = RecordingJobManager(
            do_monitoring_result=(1, [finished_job], False), move_result=True
        )
        session.job_manager = manager

        def move_data_for_job(job: JobV2, dest: Path) -> bool:
            shutil.move(str(job.wdir), str(dest))
            return True

        manager.move_data_for_job = move_data_for_job
        monkeypatch.setattr(session, "print_top_designs", lambda n=5: None)
        monkeypatch.setattr(session, "_check_termination_criteria", lambda: False)
        monkeypatch.setattr(
            session,
            "loop_optimizer_once",
            lambda num_new_cases: (_ for _ in ()).throw(OptimizationComplete("done")),
        )
        monkeypatch.setattr(session, "_write_designs_log", lambda: None)

        session.persistent_optimization()

        archived = Case.try_restoring(session.archival_dir / finished_job.wdir.name)
        assert archived._generation_index == "00001.01"
        assert archived._based_on_case == tmp_path / "template"
        assert archived._execution_environment == "unit-test"
        assert archived._model_predictions_by_objective == {
            "drag": {"mean": 1.23, "sem": 0.1}
        }

        metadata = archived.read_metadata()
        assert metadata["generation_index"] == "00001.01"
        assert metadata["optimizer-suggestion"] == {"x": {"value": 0.5}}

    def test_caps_new_cases_to_remaining_max_evaluations(self, tmp_path, monkeypatch):
        session = Session(name="test", data_dir=tmp_path / "session", max_evaluations=3)
        for i in range(2):
            case = Case(_make_foam_dir(session.archival_dir / f"job_0000{i}_abc"))
            case.success = True
            case.persist_to_file()

        manager = RecordingJobManager()
        manager.job_limit = 2
        session.job_manager = manager

        monitoring_calls = 0

        def do_monitoring():
            nonlocal monitoring_calls
            monitoring_calls += 1
            if monitoring_calls == 1:
                return (2, [], False)

            pending = session.get_pending_cases()
            assert len(pending) == 1
            return (
                1,
                [
                    JobV2(
                        id="456",
                        name=f"flwbst_{pending[0].name}",
                        wdir=pending[0].path,
                    )
                ],
                False,
            )

        def move_data_for_job(job: JobV2, dest: Path) -> bool:
            shutil.move(str(job.wdir), str(dest))
            return True

        requested_new_cases: list[int] = []

        def loop_optimizer_once(num_new_cases: int):
            requested_new_cases.append(num_new_cases)
            assert num_new_cases == 1
            return [Case(_make_foam_dir(session.pending_dir / "job_00002_abc"))]

        manager.do_monitoring = do_monitoring
        manager.move_data_for_job = move_data_for_job
        monkeypatch.setattr(session, "print_top_designs", lambda n=5: None)
        monkeypatch.setattr(session, "_write_designs_log", lambda: None)
        monkeypatch.setattr(session, "loop_optimizer_once", loop_optimizer_once)

        session.persistent_optimization()

        assert requested_new_cases == [1]
        assert [case.name for case in manager.submitted_cases] == ["job_00002_abc"]
        assert len(session.get_finished_cases(include_failed=True)) == 3

    def test_waits_when_max_evaluation_budget_is_pending(self, tmp_path, monkeypatch):
        session = Session(name="test", data_dir=tmp_path / "session", max_evaluations=3)
        for i in range(2):
            case = Case(_make_foam_dir(session.archival_dir / f"job_0000{i}_abc"))
            case.success = True
            case.persist_to_file()

        pending = Case(_make_foam_dir(session.pending_dir / "job_00002_abc"))
        pending.persist_to_file()

        manager = RecordingJobManager()
        manager.job_limit = 2
        session.job_manager = manager

        monitoring_calls = 0

        def do_monitoring():
            nonlocal monitoring_calls
            monitoring_calls += 1
            if monitoring_calls == 1:
                return (1, [], False)

            return (
                1,
                [
                    JobV2(
                        id="456",
                        name=f"flwbst_{pending.name}",
                        wdir=pending.path,
                    )
                ],
                False,
            )

        def move_data_for_job(job: JobV2, dest: Path) -> bool:
            shutil.move(str(job.wdir), str(dest))
            return True

        manager.do_monitoring = do_monitoring
        manager.move_data_for_job = move_data_for_job
        monkeypatch.setattr(session, "print_top_designs", lambda n=5: None)
        monkeypatch.setattr(session, "_write_designs_log", lambda: None)
        monkeypatch.setattr(
            session,
            "loop_optimizer_once",
            lambda num_new_cases: pytest.fail(
                f"unexpected acquisition request for {num_new_cases} cases"
            ),
        )

        session.persistent_optimization()

        assert manager.submitted_cases == []
        assert len(session.get_finished_cases(include_failed=True)) == 3

    def test_failed_cases_consume_remaining_evaluation_budget(
        self, tmp_path, monkeypatch
    ):
        session = Session(name="test", data_dir=tmp_path / "session", max_evaluations=2)

        successful = Case(_make_foam_dir(session.archival_dir / "job_00000_abc"))
        successful.success = True
        successful.persist_to_file()

        failed = Case(_make_foam_dir(session.archival_dir / "job_00001_failed"))
        failed.success = False
        failed.persist_to_file()

        manager = RecordingJobManager()
        manager.job_limit = 2
        session.job_manager = manager

        monkeypatch.setattr(session, "print_top_designs", lambda n=5: None)
        monkeypatch.setattr(session, "_write_designs_log", lambda: None)
        monkeypatch.setattr(
            session,
            "loop_optimizer_once",
            lambda num_new_cases: pytest.fail(
                f"unexpected acquisition request for {num_new_cases} cases"
            ),
        )

        session.persistent_optimization()

        assert manager.submitted_cases == []
