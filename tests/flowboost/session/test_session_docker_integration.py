import os
import time
from pathlib import Path

import pytest

from flowboost import Case, Dictionary, Dimension, Manager, Objective, Session
from flowboost.openfoam.runtime import FOAMRuntime, get_runtime, reset_runtime

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def docker_foam_runtime(tmp_path_factory):
    previous_mode = os.environ.get("FLOWBOOST_FOAM_MODE")
    os.environ["FLOWBOOST_FOAM_MODE"] = "docker"
    reset_runtime()

    try:
        runtime = get_runtime()
    except RuntimeError:
        _restore_foam_mode(previous_mode)
        pytest.skip("Docker-based OpenFOAM runtime not available")

    if runtime.mode != FOAMRuntime.Mode.DOCKER:
        runtime.cleanup()
        reset_runtime()
        _restore_foam_mode(previous_mode)
        pytest.skip("Docker-based OpenFOAM runtime not active")

    if not runtime.is_available():
        runtime.cleanup()
        reset_runtime()
        _restore_foam_mode(previous_mode)
        pytest.skip("Docker-based OpenFOAM runtime not usable")

    runtime.add_mount(tmp_path_factory.getbasetemp(), "/work")

    try:
        yield runtime
    finally:
        runtime.cleanup()
        reset_runtime()
        _restore_foam_mode(previous_mode)


def _restore_foam_mode(previous_mode: str | None) -> None:
    if previous_mode is None:
        os.environ.pop("FLOWBOOST_FOAM_MODE", None)
    else:
        os.environ["FLOWBOOST_FOAM_MODE"] = previous_mode


def _pressure_drop_objective(case: Case) -> float | None:
    fo_name = "patchAverage(patch=inlet,fields=(pU))"
    df = case.data.simple_function_object_reader(fo_name, backend="pandas")

    if df is None or df.empty:
        return None

    return float(df["areaAverage(p)"].iloc[-1])


def _configure_pitzdaily_case(case_dir: Path) -> Case:
    case = Case.from_tutorial("incompressibleFluid/pitzDaily", case_dir)
    case.foam_get("patchAverage")

    patch_average = case.dictionary("system/patchAverage")
    patch_average.set("patch", "inlet")
    patch_average.set("fields", "(p U)")

    control_dict = case.dictionary("system/controlDict")
    control_dict.entry("endTime").set("0.01")

    return case


def _wait_for_finished_job(manager: Manager, timeout_seconds: int = 180):
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        finished_jobs = {
            job for job in manager.job_pool if manager._job_has_finished(job.id)
        }
        if finished_jobs:
            manager.job_pool.difference_update(finished_jobs)
            manager._save_state()
            return list(finished_jobs)

        time.sleep(1)

    tracked = ", ".join(sorted(job.name for job in manager.job_pool)) or "none"
    raise TimeoutError(f"Timed out waiting for DockerLocal job(s): {tracked}")


def _build_pitzdaily_session(tmp_path: Path, max_evaluations: int) -> Session:
    session = Session(
        name="pitzDaily-docker-test",
        data_dir=tmp_path / "session_data",
        max_evaluations=max_evaluations,
    )

    template = _configure_pitzdaily_case(session.data_dir / "pitzDaily_template")
    session.attach_template_case(case=template)

    objective = Objective(
        name="inlet_pressure",
        minimize=True,
        objective_function=_pressure_drop_objective,
        normalization_step="min-max",
    )
    session.backend.set_objectives([objective])

    inlet_k = Dictionary.link("0/k").entry("boundaryField/inlet/value")
    session.backend.set_search_space(
        [
            Dimension.range(
                name="inlet_k",
                link=inlet_k,
                lower=0.1,
                upper=1.5,
                log_scale=True,
            )
        ]
    )

    session.job_manager = Manager.create(
        scheduler="dockerlocal", wdir=session.data_dir, job_limit=1
    )
    session.job_manager.monitoring_interval = 1
    session.backend.initialization_trials = 1
    session.clean_pending_cases()
    return session


def _submit_and_archive_case(session: Session, case: Case) -> Case:
    manager = session.job_manager
    assert manager is not None

    assert manager.submit_case(case)

    finished_job = _wait_for_finished_job(manager)[0]
    archived_path = session.archival_dir / finished_job.wdir.name

    assert manager.move_data_for_job(finished_job, archived_path)

    archived_case = Case(archived_path)
    archived_case.post_evaluation_update(finished_job.to_dict())
    return archived_case


def test_pitzdaily_dockerlocal_example_smoke(docker_foam_runtime, tmp_path):
    manager = Manager.create(scheduler="dockerlocal", wdir=tmp_path, job_limit=1)
    case = _configure_pitzdaily_case(tmp_path / "pitzDaily_case")

    assert manager.submit_case(case)

    finished_jobs = _wait_for_finished_job(manager)
    assert len(finished_jobs) == 1
    assert _pressure_drop_objective(case) is not None

    function_object_dir = (
        case.path / "postProcessing" / "patchAverage(patch=inlet,fields=(pU))"
    )
    assert function_object_dir.exists()


def test_session_loop_optimizer_cycle_with_dockerlocal(docker_foam_runtime, tmp_path):
    session = _build_pitzdaily_session(tmp_path, max_evaluations=2)
    manager = session.job_manager
    assert manager is not None

    session.backend.initialize()

    first_cases = session.loop_optimizer_once(num_new_cases=1)
    assert len(first_cases) == 1
    first_case = first_cases[0]
    assert manager.submit_case(first_case)

    first_job = _wait_for_finished_job(manager)[0]
    first_dest = session.archival_dir / first_job.wdir.name
    assert manager.move_data_for_job(first_job, first_dest)
    archived_first = Case(first_dest)
    archived_first.post_evaluation_update(first_job.to_dict())

    second_cases = session.loop_optimizer_once(num_new_cases=1)
    assert len(second_cases) == 1

    first_metadata = archived_first.read_metadata()
    assert first_metadata is not None
    assert type(float(first_metadata["objective-values-raw"]["inlet_pressure"])) is float
    assert type(float(first_metadata["objective-outputs"]["inlet_pressure"]["value"])) is float

    second_case = second_cases[0]
    assert manager.submit_case(second_case)

    second_job = _wait_for_finished_job(manager)[0]
    second_dest = session.archival_dir / second_job.wdir.name
    assert manager.move_data_for_job(second_job, second_dest)
    Case(second_dest).post_evaluation_update(second_job.to_dict())

    finished_cases = session.get_finished_cases(include_failed=False, batch_process=True)
    assert len(finished_cases) == 2
    for case in finished_cases:
        outputs = case.objective_function_outputs(session.backend.objectives)
        assert type(outputs["inlet_pressure"]) is float

    assert session._check_termination_criteria()


def test_session_tell_allows_duplicate_parameterizations_with_dockerlocal(
    docker_foam_runtime, tmp_path
):
    session = _build_pitzdaily_session(tmp_path, max_evaluations=3)
    session.backend.initialize()

    inlet_k = session.backend.dimensions[0]
    duplicate_suggestion = {inlet_k: 0.5}

    first_case = session._process_optimizer_suggestion([duplicate_suggestion])[0]
    second_case = session._process_optimizer_suggestion([duplicate_suggestion])[0]

    archived_first = _submit_and_archive_case(session, first_case)
    archived_second = _submit_and_archive_case(session, second_case)

    assert archived_first.name != archived_second.name
    assert archived_first.parametrize_configuration(session.backend.dimensions) == {
        "inlet_k": 0.5
    }
    assert archived_second.parametrize_configuration(session.backend.dimensions) == {
        "inlet_k": 0.5
    }

    finished_cases = sorted(
        session.get_finished_cases(include_failed=False, batch_process=True),
        key=lambda case: case.name,
    )
    assert len(finished_cases) == 2

    session.backend.tell(finished_cases)

    attached_trials = {
        case.name: session.backend.client.experiment.trials[
            session.backend._trial_index_case_mapping[case]
        ]
        for case in finished_cases
    }

    assert set(attached_trials) == {archived_first.name, archived_second.name}
    assert len({trial.index for trial in attached_trials.values()}) == 2
    assert all(trial.status.is_completed for trial in attached_trials.values())
    assert all(trial.arm is not None for trial in attached_trials.values())
    assert {trial.arm.parameters["inlet_k"] for trial in attached_trials.values()} == {
        0.5
    }
    assert {trial.arm.name for trial in attached_trials.values()} == {
        finished_cases[0].name
    }
    assert list(session.backend.client.experiment.arms_by_name) == [
        finished_cases[0].name
    ]
