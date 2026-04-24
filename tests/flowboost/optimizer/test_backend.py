import json
import logging
from typing import Any, cast

import pytest
from ax.core.arm import Arm
from ax.core.trial import Trial

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.backend import OptimizationComplete
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import AggregateObjective, Objective
from flowboost.optimizer.search_space import Dimension


def _trial_for_case(backend: AxBackend, case: Case) -> Trial:
    trial = backend.client.experiment.trials[backend._trial_index_case_mapping[case]]
    assert isinstance(trial, Trial)
    return trial


def _trial_arm(trial: Trial) -> Arm:
    assert trial.arm is not None
    return trial.arm


@pytest.fixture
def Ax_backend() -> AxBackend:
    # Test that AxBackend initializes correctly
    backend = AxBackend()

    # Add objective
    objective = Objective(
        name="test_objective", minimize=True, objective_function=lambda x: 1
    )

    # Define something to modify
    dict_link = DictionaryLink("constant/chemistryProperties").entry(
        "tabulation/tolerance"
    )

    # Add dimension for the dictionary entry
    dim = Dimension.range(
        name="test_dim", link=dict_link, lower=1e-5, upper=1e-1, log_scale=True
    )

    # Define something to modify
    dict_link = DictionaryLink("constant/cloudProperties").entry(
        "subModels/injectionModels/model1/SOI"
    )

    # Add dimension for the dictionary entry
    dim = Dimension.range(
        name="test_dim", link=dict_link, lower=1e-5, upper=1e-1, log_scale=True
    )

    # Set search space + objectives
    backend.set_search_space([dim])
    backend.set_objectives([objective])
    return backend


def test_initialization(Ax_backend):
    Ax_backend.initialize()


def test_ask_before_initialize_raises_clear_error(Ax_backend):
    """Calling ask() on a fresh backend used to surface a deep Ax
    AssertionError; it should now raise a FlowBoost-level guidance message."""
    with pytest.raises(RuntimeError, match="called before initialize"):
        Ax_backend.ask(max_cases=1)


def test_ask_raises_optimization_complete_instead_of_sys_exit(Ax_backend, monkeypatch):
    """The backend used to call sys.exit when Ax reported the optimization
    finished, which would kill the host process. It should now raise a
    dedicated exception the caller can catch."""
    Ax_backend.initialize()

    def fake_get_next_trials(*args, **kwargs):
        return ({}, True)  # empty parametrizations, finished=True

    monkeypatch.setattr(Ax_backend.client, "get_next_trials", fake_get_next_trials)

    with pytest.raises(OptimizationComplete):
        Ax_backend.ask(max_cases=1)


def test_ask_with_no_initialization_and_no_data_raises_clear_error(Ax_backend):
    """init_trials=0 + no attached trials used to produce an opaque Ax
    DataRequiredError deep in the transform pipeline. It should now surface
    as a FlowBoost-level ValueError pointing at the fix."""
    Ax_backend.initialization_trials = 0
    Ax_backend.initialize()

    with pytest.raises(ValueError, match="no observations to fit a surrogate"):
        Ax_backend.ask(max_cases=1)


def test_ask_with_no_initialization_but_cold_start_trials_works(tmp_path):
    """init_trials=0 is legitimate when the caller attaches cold-start trials
    via tell() before ask(). This path must keep working."""
    first = _make_case(tmp_path, "cold-a", value=0.25)
    second = _make_case(tmp_path, "cold-b", value=0.75)
    backend, objective = _make_normalized_backend(first)
    backend.initialization_trials = 0
    # Re-initialize so the init_trials=0 setting takes effect.
    backend._initialized = False
    backend.client = backend.client.__class__()
    backend.initialize()

    _evaluate_objective_batch([first, second], objective)
    backend.tell([first, second])

    # With two cold-start trials attached, BO should be able to generate.
    suggestion = backend.ask(max_cases=1)
    assert len(suggestion) == 1


def test_ask_returns_empty_when_backend_yields_no_trials(Ax_backend, monkeypatch):
    """An empty generator response (e.g. parallelism cap reached mid-run) is
    a legitimate state; ask() should return an empty list, not raise."""
    Ax_backend.initialize()

    def fake_get_next_trials(*args, **kwargs):
        return ({}, False)  # empty, not finished

    monkeypatch.setattr(Ax_backend.client, "get_next_trials", fake_get_next_trials)

    assert Ax_backend.ask(max_cases=1) == []


def test_tell_on_unevaluated_case_raises_clear_error(tmp_path):
    """Calling tell() with a case that hasn't been batch-processed used to
    blow up with 'output None for case ...' from deep inside Case; it should
    now point the caller at batch_process / get_finished_cases(batch_process=True)."""
    unevaluated = _make_case(tmp_path, "unevaluated", value=0.5)
    backend, _ = _make_normalized_backend(unevaluated)
    other = _make_case(tmp_path, "other", value=0.25)

    with pytest.raises(ValueError, match="has not been evaluated"):
        backend.tell([other])


@pytest.mark.slow
def test_tell(Ax_backend, test_case, foam_in_env):
    # Evaluate an objective
    obj = Ax_backend.objectives[0]

    # Run evaluation for objective
    outputs = obj.batch_evaluate(cases=[test_case])
    out = obj.batch_post_process(cases=[test_case], outputs=outputs)
    logging.info(f"Batch-processed: {out}")

    # Initialize backend
    Ax_backend.initialize()

    logging.info("Running tell()")
    Ax_backend.tell([test_case])


def _make_case(tmp_path, name: str, value: float = 0.5) -> Case:
    return _make_case_with_params(tmp_path, name, {"test_dim": value})


def _make_case_with_params(
    tmp_path, name: str, params: dict[str, int | float | bool | str]
) -> Case:
    case_dir = tmp_path / name
    case_dir.mkdir()
    case = Case(case_dir)
    case.update_metadata(
        {key: {"value": value} for key, value in params.items()},
        entry_header="optimizer-suggestion",
    )
    return case


def _evaluate_objective(case: Case, objective: Objective) -> None:
    outputs = objective.batch_evaluate([case])
    objective.batch_post_process([case], outputs)


def _evaluate_objective_batch(cases: list[Case], objective: Objective) -> None:
    outputs = objective.batch_evaluate(cases)
    objective.batch_post_process(cases, outputs)


def _make_normalized_backend(case: Case) -> tuple[AxBackend, Objective]:
    objective = Objective(
        name="normalized_objective",
        minimize=True,
        objective_function=lambda _: 1.0,
        normalization_step="min-max",
    )
    _evaluate_objective(case, objective)

    backend = AxBackend()
    backend.set_search_space(
        [
            Dimension.range(
                name="test_dim",
                link=DictionaryLink("constant/chemistryProperties").entry(
                    "tabulation/tolerance"
                ),
                lower=0.0,
                upper=1.0,
            )
        ]
    )
    backend.set_objectives([objective])
    return backend, objective


def _make_issue_style_backend(
    *, random_seed: int | None = None, should_deduplicate: bool = True
) -> tuple[AxBackend, Objective]:
    backend = AxBackend()
    backend.initialization_trials = 1
    backend.random_seed = random_seed
    backend.should_deduplicate = should_deduplicate
    backend.set_search_space(
        [
            Dimension.range(
                name="heatSource",
                link=DictionaryLink("constant/energy").entry("heatSource"),
                lower=500.0,
                upper=2000.0,
            ),
            Dimension.choice(
                name="position",
                link=DictionaryLink("constant/setup").entry("position"),
                choices=[1, 3, 5, 7],
                dtype=int,
                is_ordered=True,
            ),
        ]
    )
    objective = Objective(
        name="score",
        minimize=True,
        objective_function=lambda case: (
            float(case.read_metadata()["optimizer-suggestion"]["heatSource"]["value"])
            + float(case.read_metadata()["optimizer-suggestion"]["position"]["value"])
        ),
    )
    backend.set_objectives([objective])
    return backend, objective


def _collect_issue_style_suggestions(
    tmp_path, backend: AxBackend, objective: Objective, limit: int
) -> tuple[list[dict[str, int | float]], int | None, dict[str, int | float] | None]:
    backend.initialize()

    finished_cases: list[Case] = []
    seen: list[dict[str, int | float]] = []

    for i in range(1, limit + 1):
        suggestion = backend.ask(1)[0]
        params = {dim.name: value for dim, value in suggestion.items()}

        if params in seen:
            return seen, i, params

        case = _make_case_with_params(tmp_path, f"issue-case-{i:02d}", params)
        finished_cases.append(case)
        _evaluate_objective_batch(finished_cases, objective)
        seen.append(params)
        backend.tell(finished_cases)

    return seen, None, None


def test_tell_accepts_normalized_scalar_like_outputs(tmp_path):
    case = _make_case(tmp_path, "normalized-case")
    backend, _ = _make_normalized_backend(case)

    backend.tell([case])

    trial_index = backend._trial_index_case_mapping[case]
    trial = backend.client.experiment.trials[trial_index]
    assert trial.status.is_completed


def test_batch_process_and_tell_support_aggregate_objective(tmp_path):
    first_case = _make_case(tmp_path, "aggregate-a", value=0.25)
    second_case = _make_case(tmp_path, "aggregate-b", value=0.75)
    for case in (first_case, second_case):
        case.success = True

    objective_a = Objective(
        name="component_a",
        minimize=True,
        objective_function=lambda case: float(
            case.read_metadata()["optimizer-suggestion"]["test_dim"]["value"]
        ),
    )
    objective_b = Objective(
        name="component_b",
        minimize=True,
        objective_function=lambda case: 2.0,
    )
    aggregate = AggregateObjective(
        name="aggregate",
        minimize=True,
        objectives=[objective_a, objective_b],
        threshold=0.0,
        weights=[0.5, 0.5],
    )

    backend = AxBackend()
    backend.set_search_space(
        [
            Dimension.range(
                name="test_dim",
                link=DictionaryLink("constant/chemistryProperties").entry(
                    "tabulation/tolerance"
                ),
                lower=0.0,
                upper=1.0,
            )
        ]
    )
    backend.set_objectives([aggregate])

    outputs = backend.batch_process([first_case, second_case])
    assert outputs == [[1.125, 1.375]]
    assert aggregate.data_for_case(first_case) == pytest.approx(1.125)
    assert aggregate.data_for_case(second_case) == pytest.approx(1.375)

    backend.tell([first_case, second_case])

    for case in (first_case, second_case):
        trial_index = backend._trial_index_case_mapping[case]
        trial = backend.client.experiment.trials[trial_index]
        assert trial.status.is_completed


def test_tell_reuses_existing_arm_for_duplicate_parameterizations(tmp_path):
    first_case = _make_case(tmp_path, "duplicate-a", value=0.5)
    second_case = _make_case(tmp_path, "duplicate-b", value=0.5)
    backend, objective = _make_normalized_backend(first_case)
    _evaluate_objective_batch([first_case, second_case], objective)

    backend.tell([first_case, second_case])

    first_trial = _trial_for_case(backend, first_case)
    second_trial = _trial_for_case(backend, second_case)

    assert first_trial.index != second_trial.index
    assert first_trial.status.is_completed
    assert second_trial.status.is_completed
    assert first_trial.arm is not None
    assert second_trial.arm is not None
    assert first_trial.arm.parameters == {"test_dim": 0.5}
    assert second_trial.arm.parameters == {"test_dim": 0.5}
    assert first_trial.arm.name == second_trial.arm.name == first_case.name
    assert list(backend.client.experiment.arms_by_name) == [first_case.name]


def test_tell_collapses_boundary_float_noise_onto_one_arm(tmp_path):
    """Regression for #18: BO converging on a box boundary can emit
    numerically-indistinguishable floats (e.g. ``500.0`` and
    ``500.0000000000001``). Without range-level rounding, each hashes to a
    distinct ``Arm.signature``, slipping past Ax's dedup and our
    ``_arm_name_for_attachment`` lookup, and surfacing as "duplicate" top
    designs. With the default ``digits`` applied to the dimension, the
    values collapse onto the same arm before ever reaching Ax."""
    exact = _make_case(tmp_path, "boundary-exact", value=0.5)
    noisy_up = _make_case(tmp_path, "boundary-noisy-up", value=0.5 + 1e-14)
    noisy_down = _make_case(tmp_path, "boundary-noisy-down", value=0.5 - 1e-14)

    backend, objective = _make_normalized_backend(exact)
    _evaluate_objective_batch([exact, noisy_up, noisy_down], objective)

    # Sanity-check the precondition: default digits is set for a float range.
    (dim,) = backend.dimensions
    assert dim.digits is not None and dim.digits >= 11

    backend.tell([exact, noisy_up, noisy_down])

    for case in (exact, noisy_up, noisy_down):
        assert case in backend._trial_index_case_mapping

    arm_names = {
        _trial_arm(_trial_for_case(backend, c)).name
        for c in (exact, noisy_up, noisy_down)
    }
    assert arm_names == {exact.name}
    assert list(backend.client.experiment.arms_by_name) == [exact.name]


def test_prepare_for_acquisition_offload_serializes_normalized_outputs(tmp_path):
    case = _make_case(tmp_path, "offload-case")
    backend, _ = _make_normalized_backend(case)
    backend.offload_acquisition = True
    backend.initialize()

    _, data_snapshot = backend.prepare_for_acquisition_offload([case], [], tmp_path)
    snapshot_data = json.loads(data_snapshot.read_text())
    objective_value = snapshot_data["finished_cases"][case.name]["objectives"][
        "normalized_objective"
    ]

    assert type(objective_value) is float
    assert objective_value == 0.0


def test_prepare_for_acquisition_offload_rejects_duplicate_pending_case_names(
    tmp_path,
):
    first_dir = tmp_path / "group_a" / "shared_case"
    second_dir = tmp_path / "group_b" / "shared_case"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)

    first_case = Case(first_dir)
    second_case = Case(second_dir)
    for case, value in ((first_case, 0.25), (second_case, 0.75)):
        case.update_metadata(
            {"test_dim": {"value": value}},
            entry_header="optimizer-suggestion",
        )

    backend, _ = _make_normalized_backend(first_case)
    backend.offload_acquisition = True
    backend.initialize()

    with pytest.raises(ValueError, match="unique across finished and pending cases"):
        backend.prepare_for_acquisition_offload([], [first_case, second_case], tmp_path)


def test_offloaded_acquisition_round_trip(tmp_path):
    case = _make_case(tmp_path, "roundtrip-case")
    backend, _ = _make_normalized_backend(case)
    backend.offload_acquisition = True
    backend.initialize()

    model_snapshot, data_snapshot = backend.prepare_for_acquisition_offload(
        [case], [], tmp_path
    )
    output_path = tmp_path / "acquisition_result.json"

    AxBackend.offloaded_acquisition(
        model_snapshot=model_snapshot,
        data_snapshot=data_snapshot,
        num_trials=1,
        output_path=output_path,
    )

    result = json.loads(output_path.read_text())
    assert result["optimizer"] == "AxBackend"
    assert result["status_finished"] is False
    assert len(result["parametrizations"]) == 1


def test_offloaded_acquisition_accepts_duplicate_parameterizations(tmp_path):
    first_case = _make_case(tmp_path, "roundtrip-duplicate-a", value=0.5)
    second_case = _make_case(tmp_path, "roundtrip-duplicate-b", value=0.5)
    backend, objective = _make_normalized_backend(first_case)
    _evaluate_objective_batch([first_case, second_case], objective)
    backend.offload_acquisition = True
    backend.initialize()

    model_snapshot, data_snapshot = backend.prepare_for_acquisition_offload(
        [first_case, second_case], [], tmp_path
    )
    output_path = tmp_path / "duplicate_acquisition_result.json"

    AxBackend.offloaded_acquisition(
        model_snapshot=model_snapshot,
        data_snapshot=data_snapshot,
        num_trials=1,
        output_path=output_path,
    )

    result = json.loads(output_path.read_text())
    assert result["optimizer"] == "AxBackend"
    assert result["status_finished"] is False
    assert len(result["parametrizations"]) == 1


def test_issue_style_search_space_encoding_matches_ax_schema():
    backend, _ = _make_issue_style_backend()

    assert backend._get_ax_search_space() == [
        {
            "name": "heatSource",
            "type": "range",
            "value_type": "float",
            "bounds": [500.0, 2000.0],
            "log_scale": False,
            "digits": 8,
        },
        {
            "name": "position",
            "type": "choice",
            "value_type": "int",
            "values": [1, 3, 5, 7],
            "is_ordered": True,
        },
    ]


def test_issue_style_generation_passes_disabled_deduplication_to_ax(monkeypatch):
    # Behaviorally forcing Ax to return a duplicate is fragile: the Sobol
    # engine is quasi-random and the BO step's rejection sampler rejects
    # repeats independently of `should_deduplicate`, so the outcome hinges
    # on the exact Ax version. FlowBoost owns passing the flag into Ax's
    # generation-strategy configuration; Ax owns how it applies that setting
    # internally across strategy steps / nodes.
    backend, _ = _make_issue_style_backend(
        random_seed=0,
        should_deduplicate=False,
    )
    captured: dict[str, object] = {}

    def fake_create_experiment(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(backend.client, "create_experiment", fake_create_experiment)

    backend.initialize()

    gs_kwargs = cast(dict[str, Any], captured["choose_generation_strategy_kwargs"])
    assert gs_kwargs["should_deduplicate"] is False


def test_issue_style_generation_avoids_repeats_with_deduplication(tmp_path):
    backend, objective = _make_issue_style_backend(
        random_seed=0,
        should_deduplicate=True,
    )

    suggestions, duplicate_at, duplicate = _collect_issue_style_suggestions(
        tmp_path,
        backend,
        objective,
        limit=10,
    )

    assert len(suggestions) == 10
    assert duplicate_at is None
    assert duplicate is None


def test_ax_backend_deduplicates_by_default():
    backend = AxBackend()

    assert backend.should_deduplicate is True
