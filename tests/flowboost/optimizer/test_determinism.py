"""
Verify determinism of the stateless replay cycle used by the async
close/reopen workflow: a user shuts down FlowBoost while cluster jobs run,
reopens later, and the optimizer replays all finished cases into a fresh
AxClient before generating the next suggestion.

The guarantees tested here:

1. **Replay-vs-replay**: two independent replayed sessions with the same
   seed and data produce identical suggestions — both in the Sobol
   (initialization) phase and the BO (model-based) phase.

2. **Continuous-vs-replayed**: a single backend kept alive across cycles
   produces the same suggestions as fresh-backend replay.

3. **Normalization stability**: the above guarantees hold when user-side
   normalization (e.g. MinMaxScaler) is active and the scale shifts as
   new cases arrive.
"""

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import Objective
from flowboost.optimizer.search_space import Dimension

SEED = 42
N_CYCLES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(seed: int) -> tuple[AxBackend, Objective]:
    backend = AxBackend()
    backend.random_seed = seed
    backend.initialization_trials = 3

    objective = Objective(
        name="score",
        minimize=True,
        objective_function=lambda case: (
            float(case.read_metadata()["optimizer-suggestion"]["x"]["value"]) ** 2
            + float(case.read_metadata()["optimizer-suggestion"]["y"]["value"]) ** 2
        ),
    )

    dims = [
        Dimension.range(
            name="x",
            link=DictionaryLink("constant/setup").entry("x"),
            lower=-5.0,
            upper=5.0,
        ),
        Dimension.range(
            name="y",
            link=DictionaryLink("constant/setup").entry("y"),
            lower=-5.0,
            upper=5.0,
        ),
    ]

    backend.set_search_space(dims)
    backend.set_objectives([objective])
    return backend, objective


def _make_1d_normalized_backend(seed: int) -> tuple[AxBackend, Objective]:
    backend = AxBackend()
    backend.random_seed = seed
    backend.initialization_trials = 3

    objective = Objective(
        name="score",
        minimize=True,
        objective_function=lambda case: (
            float(case.read_metadata()["optimizer-suggestion"]["x"]["value"]) ** 2
        ),
        normalization_step="min-max",
    )
    dims = [
        Dimension.range(
            name="x",
            link=DictionaryLink("constant/setup").entry("x"),
            lower=-5.0,
            upper=5.0,
        ),
    ]
    backend.set_search_space(dims)
    backend.set_objectives([objective])
    return backend, objective


def _make_case(tmp_path, name: str, params: dict[str, float]) -> Case:
    case_dir = tmp_path / name
    case_dir.mkdir(parents=True, exist_ok=True)
    case = Case(case_dir)
    case.update_metadata(
        {key: {"value": value} for key, value in params.items()},
        entry_header="optimizer-suggestion",
    )
    return case


def _evaluate_and_tell(
    backend: AxBackend,
    objective: Objective,
    cases: list[Case],
) -> None:
    """Run batch evaluation and tell the backend about all cases."""
    outputs = objective.batch_evaluate(cases)
    objective.batch_post_process(cases, outputs)
    backend.tell(cases)


def _run_replayed(
    tmp_path,
    prefix: str,
    make_backend_fn,
    seed: int = SEED,
    n_cycles: int = N_CYCLES,
) -> list[dict[str, float]]:
    """Fresh backend each cycle, replaying all history from scratch.
    This is the actual async close/reopen workflow."""
    all_cases: list[Case] = []
    suggestions: list[dict[str, float]] = []

    for cycle in range(n_cycles):
        backend, objective = make_backend_fn(seed)
        backend.initialize()

        if all_cases:
            _evaluate_and_tell(backend, objective, all_cases)

        suggestion = backend.ask(max_cases=1)[0]
        params = {dim.name: value for dim, value in suggestion.items()}
        suggestions.append(params)

        case = _make_case(tmp_path, f"{prefix}-{cycle:02d}", params)
        all_cases.append(case)
        _evaluate_and_tell(backend, objective, all_cases)

    return suggestions


def _run_continuous(
    tmp_path,
    prefix: str,
    make_backend_fn,
    seed: int = SEED,
    n_cycles: int = N_CYCLES,
) -> list[dict[str, float]]:
    """Single backend kept alive across all tell/ask cycles."""
    backend, objective = make_backend_fn(seed)
    backend.initialize()

    all_cases: list[Case] = []
    suggestions: list[dict[str, float]] = []

    for cycle in range(n_cycles):
        suggestion = backend.ask(max_cases=1)[0]
        params = {dim.name: value for dim, value in suggestion.items()}
        suggestions.append(params)

        case = _make_case(tmp_path, f"{prefix}-{cycle:02d}", params)
        all_cases.append(case)
        _evaluate_and_tell(backend, objective, all_cases)

    return suggestions


def _assert_suggestions_match(
    label_a: str,
    label_b: str,
    suggestions_a: list[dict[str, float]],
    suggestions_b: list[dict[str, float]],
    *,
    abs_tol: float = 1e-12,
) -> None:
    for cycle, (a, b) in enumerate(zip(suggestions_a, suggestions_b)):
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=abs_tol), (
                f"Cycle {cycle}, param '{key}': {label_a}={a[key]}, {label_b}={b[key]}"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_replay_vs_replay_determinism(tmp_path):
    """Two independent replayed runs with the same seed must produce
    identical suggestions — including the BO phase."""
    run_a = _run_replayed(tmp_path / "a", "a", _make_backend)
    run_b = _run_replayed(tmp_path / "b", "b", _make_backend)

    _assert_suggestions_match("run_a", "run_b", run_a, run_b)


def test_replay_vs_replay_determinism_with_normalization(tmp_path):
    """Same guarantee with min-max normalization active. The normalizer
    re-fits on every batch_post_process call, so the scale shifts as new
    cases arrive. Both replayed runs must still agree."""
    run_a = _run_replayed(tmp_path / "a", "a", _make_1d_normalized_backend)
    run_b = _run_replayed(tmp_path / "b", "b", _make_1d_normalized_backend)

    _assert_suggestions_match("run_a", "run_b", run_a, run_b)


def test_continuous_vs_replayed_determinism(tmp_path):
    """A single long-lived backend must produce the same suggestions as
    fresh-backend replay. Both paths go through _re_initialize_client on
    every tell(), so the AxClient state at each ask() should be identical."""
    continuous = _run_continuous(tmp_path / "cont", "cont", _make_backend)
    replayed = _run_replayed(tmp_path / "replay", "replay", _make_backend)

    _assert_suggestions_match("continuous", "replayed", continuous, replayed)


def test_different_seeds_diverge(tmp_path):
    """Sanity check: different seeds produce different suggestions,
    confirming the seed actually controls generation."""
    backend_a, _ = _make_backend(seed=1)
    backend_b, _ = _make_backend(seed=99)
    backend_a.initialize()
    backend_b.initialize()

    suggestion_a = backend_a.ask(max_cases=1)[0]
    suggestion_b = backend_b.ask(max_cases=1)[0]

    params_a = {dim.name: value for dim, value in suggestion_a.items()}
    params_b = {dim.name: value for dim, value in suggestion_b.items()}

    assert params_a != params_b, (
        f"Seed 1 and seed 99 produced identical first suggestions: {params_a}"
    )
