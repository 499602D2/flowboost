"""Synthetic OpenFOAM campaign tests.

These tests intentionally build a tiny OpenFOAM-shaped world instead of
mocking the FlowBoost stack. A test-local ``foamDictionary`` executable edits
JSON-backed "dictionaries", and a case-local ``Allrun`` script acts as a
deterministic synthetic solver that writes OpenFOAM-style postProcessing data.

The goal is to validate the contracts that matter during long campaigns:
dictionary links are resolved and written, cases are cloned, jobs are submitted
through the real Manager API, results are read through Case.data, objectives
and constraints are replayed into Ax, metadata survives designs.json export,
and a session can be closed with pending jobs and restored later.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
from itertools import count
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from flowboost.manager.manager import Manager
from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import Dictionary
from flowboost.openfoam.runtime import reset_runtime
from flowboost.optimizer.objectives import Constraint, Objective, ScalarizedObjective
from flowboost.optimizer.search_space import Dimension
from flowboost.session.session import Session

pytestmark = pytest.mark.slow

MAX_EVALUATIONS = 8
JOB_LIMIT = 1
RANDOM_SEED = 17
INITIALIZATION_TRIALS = 4
RESTORE_INTERRUPT_AFTER_COMPLETED = INITIALIZATION_TRIALS + 1
PARETO_MAX_EVALUATIONS = 18
PARETO_INITIALIZATION_TRIALS = 5
PARETO_SEED = 29


def _install_synthetic_foam_dictionary(tmp_path: Path, monkeypatch) -> None:
    """Install a small foamDictionary-compatible CLI on PATH.

    The script supports the subset FlowBoost needs in this campaign:
    ``-keywords``, ``-keywords -entry ...``, ``-entry ... -value``, and
    ``-entry ... -set ...``.
    """
    bin_dir = tmp_path / "synthetic_bin"
    bin_dir.mkdir()
    foam_dictionary = bin_dir / "foamDictionary"
    foam_dictionary.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def parse_scalar(raw):
    raw = str(raw).strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def format_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve(data, entry_path):
    node = data
    if not entry_path:
        return node
    for part in entry_path.split("/"):
        node = node[part]
    return node


def assign(data, entry_path, value):
    parts = entry_path.split("/")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


args = sys.argv[1:]
if not args:
    print("missing dictionary path", file=sys.stderr)
    sys.exit(2)

path = Path(args[0])
data = json.loads(path.read_text())

try:
    if "-keywords" in args:
        entry = args[args.index("-entry") + 1] if "-entry" in args else ""
        node = resolve(data, entry)
        if isinstance(node, dict):
            print("\\n".join(node.keys()))
            sys.exit(0)
        print(f"'{entry}' is a leaf entry", file=sys.stderr)
        sys.exit(0)

    if "-value" in args:
        entry = args[args.index("-entry") + 1]
        print(format_scalar(resolve(data, entry)))
        sys.exit(0)

    if "-set" in args:
        entry = args[args.index("-entry") + 1]
        value = parse_scalar(args[args.index("-set") + 1])
        assign(data, entry, value)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n")
        sys.exit(0)
except (KeyError, IndexError) as exc:
    print(f"foamDictionary synthetic error: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"unsupported foamDictionary invocation: {args}", file=sys.stderr)
sys.exit(2)
""",
    )
    foam_dictionary.chmod(0o755)

    monkeypatch.setenv("FOAM_INST_DIR", str(tmp_path / "synthetic-openfoam"))
    monkeypatch.setenv("FLOWBOOST_FOAM_MODE", "native")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    reset_runtime()


def _reset_session_case_ids(monkeypatch) -> None:
    ids = count(1)
    monkeypatch.setattr(
        "flowboost.session.session.unique_id",
        lambda hashable=None: f"{next(ids):08x}",
    )


def _disable_progress_tables(session: Session) -> None:
    # Progress table rendering is not part of these campaign contracts.
    session.print_top_designs = lambda n=5, by_objective=None: None


def _create_synthetic_template(root: Path) -> Case:
    template = root / "synthetic_airfoil_template"
    (template / "constant").mkdir(parents=True)
    (template / "system").mkdir()
    (template / "constant" / "design").write_text(
        json.dumps(
            {
                "alpha": 0.0,
                "camber": 0.04,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (template / "system" / "controlDict").write_text("{}\n")

    python = shlex.quote(sys.executable)
    (template / "Allrun").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

{python} - <<'PY'
import json
from pathlib import Path

case = Path.cwd()
design = json.loads((case / "constant" / "design").read_text())
alpha = float(design["alpha"])
camber = float(design["camber"])

# A small analytic airfoil surrogate with a known feasible basin. The numbers
# are not physical constants; they are shaped to create realistic trade-offs:
# lift likes alpha/camber, drag penalizes moving away from a sweet spot, and
# pressure recovery/stability bound the useful region.
separation = max(0.0, alpha - (4.8 + 18.0 * camber))
lift = (
    1.02
    + 0.105 * alpha
    + 2.6 * camber
    - 0.010 * (alpha - 3.0) ** 2
    - 0.16 * separation**2
)
drag = (
    0.032
    + 0.0045 * (alpha - 2.6) ** 2
    + 5.5 * (camber - 0.055) ** 2
    + 0.020 * separation**2
)
pressure_recovery = 0.78 - 0.018 * (alpha - 1.8) ** 2 - 1.4 * (camber - 0.045) ** 2
stability_margin = 0.145 + 0.95 * camber - 0.018 * alpha

out_dir = case / "postProcessing" / "syntheticAero" / "0"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "coefficients.dat").write_text(
    "# Time\\tlift\\tdrag\\tpressure_recovery\\tstability_margin\\tseparation\\n"
    f"0\\t{{lift:.12f}}\\t{{drag:.12f}}\\t{{pressure_recovery:.12f}}\\t"
    f"{{stability_margin:.12f}}\\t{{separation:.12f}}\\n"
    f"1\\t{{lift:.12f}}\\t{{drag:.12f}}\\t{{pressure_recovery:.12f}}\\t"
    f"{{stability_margin:.12f}}\\t{{separation:.12f}}\\n"
)
PY
""",
    )

    return Case(template)


def _create_pareto_ribbon_template(root: Path) -> Case:
    template = root / "synthetic_pareto_template"
    (template / "constant").mkdir(parents=True)
    (template / "system").mkdir()
    (template / "constant" / "design").write_text(
        json.dumps({"x": 0.5, "y": 0.5}, indent=2, sort_keys=True) + "\n"
    )
    (template / "system" / "controlDict").write_text("{}\n")

    python = shlex.quote(sys.executable)
    (template / "Allrun").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

{python} - <<'PY'
import json
from pathlib import Path

case = Path.cwd()
design = json.loads((case / "constant" / "design").read_text())
x = float(design["x"])
y = float(design["y"])

# True MOO target: two minimized objectives with an analytic Pareto ribbon:
# x in [0.25, 0.75], y = 0.50. Any point off y=0.50 is dominated by its
# projection onto the ribbon; moving along x trades f_left against f_right.
f_left = (x - 0.25) ** 2 + 0.10 * (y - 0.50) ** 2
f_right = (x - 0.75) ** 2 + 0.10 * (y - 0.50) ** 2

# Alternate landscape used only to prove the post-Sobol BO phase consumes
# observations. Same search space and same solver output shape, different
# objective functions.
alt_left = (x - 0.12) ** 2 + (y - 0.88) ** 2
alt_right = (x - 0.88) ** 2 + (y - 0.12) ** 2

stability = 1.0 - 4.0 * (y - 0.50) ** 2

out_dir = case / "postProcessing" / "paretoRibbon" / "0"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "objectives.dat").write_text(
    "# Time\\tf_left\\tf_right\\talt_left\\talt_right\\tstability\\n"
    f"0\\t{{f_left:.12f}}\\t{{f_right:.12f}}\\t{{alt_left:.12f}}\\t"
    f"{{alt_right:.12f}}\\t{{stability:.12f}}\\n"
    f"1\\t{{f_left:.12f}}\\t{{f_right:.12f}}\\t{{alt_left:.12f}}\\t"
    f"{{alt_right:.12f}}\\t{{stability:.12f}}\\n"
)
PY
""",
    )

    return Case(template)


def _last_metric_from(
    case: Case, function_object_name: str, metric_name: str
) -> float | None:
    frame = case.data.simple_function_object_reader(function_object_name)
    if frame is None or frame.is_empty():
        return None
    assert isinstance(frame, pl.DataFrame)
    return float(frame.select(pl.last(metric_name)).item())


def _last_metric(case: Case, metric_name: str) -> float | None:
    return _last_metric_from(case, "syntheticAero", metric_name)


def _last_pareto_metric(case: Case, metric_name: str) -> float | None:
    return _last_metric_from(case, "paretoRibbon", metric_name)


def _configure_synthetic_session(
    session: Session, template: Case | None = None
) -> None:
    if template is not None:
        session.attach_template_case(template)

    lift = Objective(
        name="lift",
        minimize=False,
        objective_function=lambda case: _last_metric(case, "lift"),
    )
    drag = Objective(
        name="drag",
        minimize=True,
        objective_function=lambda case: _last_metric(case, "drag"),
        lte=0.080,
    )
    score = ScalarizedObjective(
        name="aero_score",
        minimize=False,
        objectives=[lift, drag],
        weights=[0.75, -0.25],
    )
    pressure = Constraint(
        name="pressure_recovery",
        objective_function=lambda case: _last_metric(case, "pressure_recovery"),
        gte=0.58,
    )
    stability = Constraint(
        name="stability_margin",
        objective_function=lambda case: _last_metric(case, "stability_margin"),
        gte=0.08,
    )

    session.backend.random_seed = RANDOM_SEED
    session.backend.initialization_trials = INITIALIZATION_TRIALS
    session.configure_optimization(
        objectives=score,
        constraints=[pressure, stability],
        search_space=[
            Dimension.range(
                name="alpha",
                link=Dictionary.link("constant/design").entry("alpha"),
                lower=-2.0,
                upper=6.0,
            ),
            Dimension.range(
                name="camber",
                link=Dictionary.link("constant/design").entry("camber"),
                lower=0.00,
                upper=0.10,
            ),
        ],
    )

    if session.job_manager is None:
        session.job_manager = Manager.create(
            scheduler="Local",
            wdir=session.data_dir,
            job_limit=JOB_LIMIT,
        )
    session.job_manager.monitoring_interval = 0.01
    _disable_progress_tables(session)
    session.persist()


def _configure_pareto_session(
    session: Session,
    template: Case | None = None,
    *,
    objective_variant: str = "normal",
) -> None:
    if template is not None:
        session.attach_template_case(template)

    if objective_variant == "normal":
        left_metric = "f_left"
        right_metric = "f_right"
    elif objective_variant == "alternate":
        left_metric = "alt_left"
        right_metric = "alt_right"
    else:
        raise ValueError(f"Unknown Pareto objective variant: {objective_variant}")

    f_left = Objective(
        name="f_left",
        minimize=True,
        threshold=0.70,
        objective_function=lambda case: _last_pareto_metric(case, left_metric),
    )
    f_right = Objective(
        name="f_right",
        minimize=True,
        threshold=0.70,
        objective_function=lambda case: _last_pareto_metric(case, right_metric),
    )
    stability = Constraint(
        name="stability",
        objective_function=lambda case: _last_pareto_metric(case, "stability"),
        gte=0.85,
    )

    session.backend.random_seed = PARETO_SEED
    session.backend.initialization_trials = PARETO_INITIALIZATION_TRIALS
    session.configure_optimization(
        objectives=[f_left, f_right],
        constraints=[stability],
        search_space=[
            Dimension.range(
                name="x",
                link=Dictionary.link("constant/design").entry("x"),
                lower=0.0,
                upper=1.0,
            ),
            Dimension.range(
                name="y",
                link=Dictionary.link("constant/design").entry("y"),
                lower=0.0,
                upper=1.0,
            ),
        ],
    )

    if session.job_manager is None:
        session.job_manager = Manager.create(
            scheduler="Local",
            wdir=session.data_dir,
            job_limit=JOB_LIMIT,
        )
    session.job_manager.monitoring_interval = 0.01
    _disable_progress_tables(session)
    session.persist()


def _design_parameters(designs: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "alpha": float(design["parameters"]["alpha"]),
            "camber": float(design["parameters"]["camber"]),
        }
        for design in designs
    ]


def _xy_parameters(designs: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "x": float(design["parameters"]["x"]),
            "y": float(design["parameters"]["y"]),
        }
        for design in designs
    ]


def _load_designs(session_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((session_dir / "designs.json").read_text())
    assert payload["num_designs"] == len(payload["designs"])
    return payload["designs"]


def _finite_value(metric: dict[str, Any], label: str) -> float:
    assert "value" in metric, f"{label} missing value"
    assert metric["value"] is not None, f"{label} value is None"
    value = float(metric["value"])
    assert math.isfinite(value), f"{label} value is not finite: {value}"
    return value


def _objective_value(design: dict[str, Any], name: str) -> float:
    return _finite_value(
        design["objectives"][name], f"{design['name']} objective {name}"
    )


def _constraint_value(design: dict[str, Any], name: str) -> float:
    return _finite_value(
        design["constraints"][name], f"{design['name']} constraint {name}"
    )


def _component_value(design: dict[str, Any], objective: str, component: str) -> float:
    metric = design["objectives"][objective]["components"][component]
    return _finite_value(
        metric, f"{design['name']} objective {objective} component {component}"
    )


def _case_parameters(case: Case, names: tuple[str, ...]) -> dict[str, float]:
    metadata = case.read_metadata()
    assert metadata is not None
    suggestion = metadata["optimizer-suggestion"]
    return {name: float(suggestion[name]["value"]) for name in names}


def _param_delta(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(abs(left[name] - right[name]) for name in left)


def _assert_campaign_outputs(
    session_dir: Path, *, expected_count: int = MAX_EVALUATIONS
) -> list[dict[str, Any]]:
    designs = _load_designs(session_dir)
    assert len(designs) == expected_count

    for design in designs:
        assert design["generation_index"] is not None
        assert set(design["parameters"]) == {"alpha", "camber"}
        score = design["objectives"]["aero_score"]
        score_value = _objective_value(design, "aero_score")
        assert 0.25 < score_value < 1.50
        assert score["is_scalarized"] is True
        assert score["component_bounds"] == {"drag": {"lte": 0.08}}
        assert set(score["components"]) == {"lift", "drag"}
        assert score["components"]["lift"]["minimize"] is False
        assert score["components"]["drag"]["minimize"] is True
        lift = _component_value(design, "aero_score", "lift")
        drag = _component_value(design, "aero_score", "drag")
        assert 0.30 < lift < 2.00
        assert 0.00 < drag < 0.30
        pressure = _constraint_value(design, "pressure_recovery")
        stability = _constraint_value(design, "stability_margin")
        assert -0.10 < pressure < 0.90
        assert -0.10 < stability < 0.40
        assert design["constraints"]["pressure_recovery"]["gte"] == 0.58
        assert design["constraints"]["stability_margin"]["gte"] == 0.08

    completed = session_dir / "cases_completed"
    pending = session_dir / "cases_pending"
    assert len([p for p in completed.iterdir() if p.is_dir()]) == len(designs)
    assert [p for p in pending.iterdir() if p.is_dir()] == []

    return designs


def _assert_pareto_campaign_outputs(session_dir: Path) -> list[dict[str, Any]]:
    designs = _load_designs(session_dir)
    assert len(designs) == PARETO_MAX_EVALUATIONS

    for design in designs:
        assert design["generation_index"] is not None
        assert set(design["parameters"]) == {"x", "y"}
        assert set(design["objectives"]) == {"f_left", "f_right"}
        assert design["objectives"]["f_left"]["minimize"] is True
        assert design["objectives"]["f_right"]["minimize"] is True
        f_left = _objective_value(design, "f_left")
        f_right = _objective_value(design, "f_right")
        stability = _constraint_value(design, "stability")
        assert 0.00 <= f_left < 2.00
        assert 0.00 <= f_right < 2.00
        assert 0.00 <= stability <= 1.00
        assert design["constraints"]["stability"]["gte"] == 0.85

    completed = session_dir / "cases_completed"
    pending = session_dir / "cases_pending"
    assert len([p for p in completed.iterdir() if p.is_dir()]) == len(designs)
    assert [p for p in pending.iterdir() if p.is_dir()] == []

    return designs


def _assert_design_sequences_match(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    assert len(actual) == len(expected)
    for i, (actual_design, expected_design) in enumerate(zip(actual, expected)):
        assert actual_design["generation_index"] == expected_design["generation_index"]
        actual_params = _design_parameters([actual_design])[0]
        expected_params = _design_parameters([expected_design])[0]
        assert actual_params["alpha"] == pytest.approx(
            expected_params["alpha"], abs=1e-6
        ), f"alpha diverged at design {i}"
        assert actual_params["camber"] == pytest.approx(
            expected_params["camber"], abs=1e-6
        ), f"camber diverged at design {i}"
        assert _objective_value(actual_design, "aero_score") == pytest.approx(
            _objective_value(expected_design, "aero_score"), abs=1e-6
        ), f"aero_score diverged at design {i}"
        assert _constraint_value(actual_design, "pressure_recovery") == pytest.approx(
            _constraint_value(expected_design, "pressure_recovery"), abs=1e-6
        ), f"pressure_recovery diverged at design {i}"
        assert _constraint_value(actual_design, "stability_margin") == pytest.approx(
            _constraint_value(expected_design, "stability_margin"), abs=1e-6
        ), f"stability_margin diverged at design {i}"


def _assert_sobol_prefix_matches(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    actual_params = _design_parameters(actual[:INITIALIZATION_TRIALS])
    expected_params = _design_parameters(expected[:INITIALIZATION_TRIALS])
    for i, (actual_param, expected_param) in enumerate(
        zip(actual_params, expected_params)
    ):
        assert actual_param["alpha"] == pytest.approx(
            expected_param["alpha"], abs=1e-9
        ), f"Sobol alpha diverged at design {i}"
        assert actual_param["camber"] == pytest.approx(
            expected_param["camber"], abs=1e-9
        ), f"Sobol camber diverged at design {i}"


def _assert_xy_prefix_matches(
    actual: list[dict[str, float]],
    expected: list[dict[str, float]],
    *,
    n: int,
) -> None:
    assert len(actual) >= n
    assert len(expected) >= n
    for i, (actual_params, expected_params) in enumerate(zip(actual[:n], expected[:n])):
        assert actual_params["x"] == pytest.approx(expected_params["x"], abs=1e-9), (
            f"x diverged at Sobol design {i}"
        )
        assert actual_params["y"] == pytest.approx(expected_params["y"], abs=1e-9), (
            f"y diverged at Sobol design {i}"
        )


def _submit_one_pending_bo_case(session: Session) -> tuple[str, dict[str, float]]:
    # Drive exactly one optimizer cycle so the test can simulate a process
    # dying after submission but before the next monitoring pass.
    session._verify_search_space_in_template()

    cases = session.loop_optimizer_once(num_new_cases=JOB_LIMIT)
    assert len(cases) == JOB_LIMIT
    case = cases[0]
    params = _case_parameters(case, ("alpha", "camber"))

    assert session.job_manager is not None
    assert session.job_manager.submit_case(case)
    session.persist()
    return case.name, params


def _run_uninterrupted_campaign(tmp_path: Path, monkeypatch) -> list[dict[str, Any]]:
    _reset_session_case_ids(monkeypatch)
    session_dir = tmp_path / "uninterrupted"
    template = _create_synthetic_template(tmp_path / "template_a")
    session = Session(
        name="synthetic-airfoil-uninterrupted",
        data_dir=session_dir,
        clone_method="copy",
        random_seed=RANDOM_SEED,
        max_evaluations=MAX_EVALUATIONS,
    )
    _configure_synthetic_session(session, template)

    session.start()

    return _assert_campaign_outputs(session_dir)


def _run_cold_first_suggestion(tmp_path: Path, monkeypatch) -> dict[str, float]:
    _reset_session_case_ids(monkeypatch)
    session_dir = tmp_path / "cold-start"
    template = _create_synthetic_template(tmp_path / "template_cold")
    session = Session(
        name="synthetic-airfoil-cold-start",
        data_dir=session_dir,
        clone_method="copy",
        random_seed=RANDOM_SEED,
        max_evaluations=1,
    )
    _configure_synthetic_session(session, template)
    session._verify_search_space_in_template()
    session.backend.initialize()

    cases = session.loop_optimizer_once(num_new_cases=1)
    assert len(cases) == 1
    return _case_parameters(cases[0], ("alpha", "camber"))


def _run_pareto_campaign(
    tmp_path: Path,
    monkeypatch,
    *,
    subdir: str,
    objective_variant: str,
    max_evaluations: int = PARETO_MAX_EVALUATIONS,
) -> list[dict[str, Any]]:
    _reset_session_case_ids(monkeypatch)
    session_dir = tmp_path / subdir
    template = _create_pareto_ribbon_template(tmp_path / f"{subdir}_template")
    session = Session(
        name=f"synthetic-pareto-{subdir}",
        data_dir=session_dir,
        clone_method="copy",
        random_seed=PARETO_SEED,
        max_evaluations=max_evaluations,
    )
    _configure_pareto_session(
        session,
        template,
        objective_variant=objective_variant,
    )

    session.start()

    if max_evaluations == PARETO_MAX_EVALUATIONS:
        return _assert_pareto_campaign_outputs(session_dir)
    designs = _load_designs(session_dir)
    assert len(designs) == max_evaluations
    return designs


def _run_restored_campaign(
    tmp_path: Path, monkeypatch
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    _reset_session_case_ids(monkeypatch)
    session_dir = tmp_path / "restored"
    template = _create_synthetic_template(tmp_path / "template_b")
    session = Session(
        name="synthetic-airfoil-restored",
        data_dir=session_dir,
        clone_method="copy",
        random_seed=RANDOM_SEED,
        max_evaluations=RESTORE_INTERRUPT_AFTER_COMPLETED,
    )
    _configure_synthetic_session(session, template)

    # Finish the Sobol seed plus one BO trial, then interrupt with the next BO
    # case pending. A cold restart from the same seed would now produce Sobol
    # again, so this distinguishes continuation from accidental replay.
    session.start()
    before_interrupt = _assert_campaign_outputs(
        session_dir, expected_count=RESTORE_INTERRUPT_AFTER_COMPLETED
    )
    assert before_interrupt[-1]["generation_index"] == "00005.01"

    session.max_evaluations = MAX_EVALUATIONS
    pending_bo_name, pending_bo_params = _submit_one_pending_bo_case(session)

    restored = Session(name="ignored", data_dir=session_dir)
    _configure_synthetic_session(restored)
    restored.backend.initialize()
    restored.persistent_optimization()

    restored_designs = _assert_campaign_outputs(session_dir)
    assert restored_designs[5]["name"] == pending_bo_name
    assert (
        _param_delta(_design_parameters([restored_designs[5]])[0], pending_bo_params)
        < 1e-9
    )
    return restored_designs, pending_bo_params


def test_synthetic_campaign_restores_pending_jobs_without_changing_designs(
    tmp_path,
    monkeypatch,
):
    try:
        _install_synthetic_foam_dictionary(tmp_path, monkeypatch)
        uninterrupted = _run_uninterrupted_campaign(tmp_path, monkeypatch)
        restored, pending_bo_params = _run_restored_campaign(tmp_path, monkeypatch)
        cold_start_params = _run_cold_first_suggestion(tmp_path, monkeypatch)
    finally:
        reset_runtime()

    assert _param_delta(pending_bo_params, cold_start_params) > 1e-3, (
        "The interrupted case matched a cold Sobol restart. The restore path "
        "may have lost completed BO state instead of continuing the campaign."
    )
    _assert_sobol_prefix_matches(restored, uninterrupted)
    _assert_design_sequences_match(restored, uninterrupted)


def test_synthetic_pareto_ribbon_exercises_sobol_handoff_and_mobo_convergence(
    tmp_path,
    monkeypatch,
):
    try:
        _install_synthetic_foam_dictionary(tmp_path, monkeypatch)
        normal = _run_pareto_campaign(
            tmp_path,
            monkeypatch,
            subdir="pareto-normal",
            objective_variant="normal",
        )
        alternate = _run_pareto_campaign(
            tmp_path,
            monkeypatch,
            subdir="pareto-alternate",
            objective_variant="alternate",
            max_evaluations=PARETO_INITIALIZATION_TRIALS + 1,
        )
    finally:
        reset_runtime()

    normal_params = _xy_parameters(normal)
    alternate_params = _xy_parameters(alternate)

    _assert_xy_prefix_matches(
        normal_params,
        alternate_params,
        n=PARETO_INITIALIZATION_TRIALS,
    )
    first_bo_normal = normal_params[PARETO_INITIALIZATION_TRIALS]
    first_bo_alternate = alternate_params[PARETO_INITIALIZATION_TRIALS]
    first_bo_delta = abs(first_bo_normal["x"] - first_bo_alternate["x"]) + abs(
        first_bo_normal["y"] - first_bo_alternate["y"]
    )
    assert first_bo_delta > 1e-3, (
        "First post-Sobol suggestion did not change when objective observations "
        "changed. That means the campaign may still be sampling instead of "
        "using BO model state."
    )

    rounded_params = [
        (round(params["x"], 12), round(params["y"], 12)) for params in normal_params
    ]
    assert len(rounded_params) == len(set(rounded_params))
    assert len({design["name"] for design in normal}) == len(normal)
    assert any(_constraint_value(design, "stability") >= 0.85 for design in normal)
