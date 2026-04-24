"""Aggregate MOBO acceptance tests.

These tests intentionally live at the optimizer layer, not the session layer.
They check whether Ax-backed MOBO adds value over a Sobol-only baseline on a
cheap analytic multi-objective problem. Session tests should prove FlowBoost's
orchestration contracts; this file is the slower canary for optimizer quality.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Callable

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import Objective
from flowboost.optimizer.search_space import Dimension


SEEDS = (7, 13, 29)
BUDGET = 14
N_INIT = 5
REFERENCE_POINT = (0.70, 0.70)


def _read(case: Case, key: str) -> float:
    metadata = case.read_metadata()
    assert metadata is not None
    return float(metadata["optimizer-suggestion"][key]["value"])


def _pareto_ribbon_objectives() -> tuple[Objective, Objective]:
    """Two minimized objectives with known Pareto ribbon y=0.5, x in [0.25, 0.75]."""

    f_left = Objective(
        name="f_left",
        minimize=True,
        threshold=REFERENCE_POINT[0],
        objective_function=lambda case: (
            (_read(case, "x") - 0.25) ** 2 + 0.10 * (_read(case, "y") - 0.50) ** 2
        ),
    )
    f_right = Objective(
        name="f_right",
        minimize=True,
        threshold=REFERENCE_POINT[1],
        objective_function=lambda case: (
            (_read(case, "x") - 0.75) ** 2 + 0.10 * (_read(case, "y") - 0.50) ** 2
        ),
    )
    return f_left, f_right


def _make_backend(
    seed: int, initialization_trials: int
) -> tuple[AxBackend, list[Objective]]:
    backend = AxBackend()
    backend.random_seed = seed
    backend.initialization_trials = initialization_trials
    objectives = list(_pareto_ribbon_objectives())
    backend.set_objectives(objectives)
    backend.set_search_space(
        [
            Dimension.range(
                name="x",
                link=DictionaryLink("constant/setup").entry("x"),
                lower=0.0,
                upper=1.0,
            ),
            Dimension.range(
                name="y",
                link=DictionaryLink("constant/setup").entry("y"),
                lower=0.0,
                upper=1.0,
            ),
        ]
    )
    backend.initialize()
    return backend, objectives


def _non_dominated(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    non_dominated = []
    for i, candidate in enumerate(points):
        c_left, c_right = candidate
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            o_left, o_right = other
            if (
                o_left <= c_left
                and o_right <= c_right
                and (o_left < c_left or o_right < c_right)
            ):
                dominated = True
                break
        if not dominated:
            non_dominated.append(candidate)
    return non_dominated


def _hypervolume_2d_minimize(
    points: list[tuple[float, float]], reference: tuple[float, float]
) -> float:
    """Exact dominated hypervolume for 2D minimization."""

    bounded = [
        (left, right)
        for left, right in _non_dominated(points)
        if left < reference[0] and right < reference[1]
    ]
    hv = 0.0
    best_right = reference[1]
    for left, right in sorted(bounded):
        if right < best_right:
            hv += (reference[0] - left) * (best_right - right)
            best_right = right
    return hv


def _run_campaign(
    tmp_path: Path,
    make_suggestion_case: Callable[[Path, str, dict], Case],
    *,
    seed: int,
    initialization_trials: int,
    label: str,
) -> float:
    backend, objectives = _make_backend(seed, initialization_trials)
    cases: list[Case] = []
    outputs: list[tuple[float, float]] = []
    case_parent = tmp_path / label / f"seed-{seed}"

    for cycle in range(BUDGET):
        suggestion = backend.ask(max_cases=1)[0]
        params = {dim.name: value for dim, value in suggestion.items()}
        case = make_suggestion_case(case_parent, f"trial-{cycle:02d}", params)
        cases.append(case)

        for objective in objectives:
            objective.batch_evaluate(cases)
        backend.tell(cases)

        left = objectives[0].data_for_case(case)
        right = objectives[1].data_for_case(case)
        assert left is not None and right is not None
        outputs.append((left, right))

    return _hypervolume_2d_minimize(outputs, REFERENCE_POINT)


@pytest.mark.slow
def test_mobo_beats_sobol_baseline_on_pareto_ribbon(tmp_path, make_suggestion_case):
    """MOBO should beat a same-budget Sobol-only baseline in aggregate.

    This is deliberately statistical rather than seed-exact. A single Ax/BoTorch
    seed may move around after dependency updates; the median across seeds should
    still show that post-Sobol acquisitions use the observations productively.
    """

    mobo_hv = [
        _run_campaign(
            tmp_path,
            make_suggestion_case,
            seed=seed,
            initialization_trials=N_INIT,
            label="mobo",
        )
        for seed in SEEDS
    ]
    sobol_hv = [
        _run_campaign(
            tmp_path,
            make_suggestion_case,
            seed=seed,
            initialization_trials=BUDGET,
            label="sobol",
        )
        for seed in SEEDS
    ]

    median_mobo = median(mobo_hv)
    median_sobol = median(sobol_hv)
    assert median_mobo > median_sobol, (
        "MOBO did not improve median dominated hypervolume over a Sobol-only "
        f"baseline. mobo={mobo_hv}, sobol={sobol_hv}"
    )
