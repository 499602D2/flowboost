"""Tests for the Constraint primitive.

Covers construction-time validation, evaluation reuse from the shared
_CallableMetric base, and the end-to-end Ax integration that fixes the
silent-failure bug previously triggered by the old `set_outcome_constraints`
string DSL when a constraint metric wasn't registered as an objective.
"""

import json
import logging

import pytest

from flowboost.openfoam.case import Case
from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.interfaces.Ax import AxBackend
from flowboost.optimizer.objectives import Constraint, Objective, ScalarizedObjective
from flowboost.optimizer.search_space import Dimension


class TestConstraintConstruction:
    def test_requires_at_least_one_bound(self):
        with pytest.raises(ValueError, match="at least one of"):
            Constraint(name="c", objective_function=lambda c: 1.0)

    def test_accepts_gte_only(self):
        c = Constraint(name="c", objective_function=lambda c: 1.0, gte=0.5)
        assert c.gte == 0.5
        assert c.lte is None

    def test_accepts_lte_only(self):
        c = Constraint(name="c", objective_function=lambda c: 1.0, lte=10.0)
        assert c.gte is None
        assert c.lte == 10.0

    def test_accepts_both_bounds(self):
        c = Constraint(name="c", objective_function=lambda c: 1.0, gte=0.0, lte=1.0)
        assert (c.gte, c.lte) == (0.0, 1.0)

    def test_rejects_empty_feasible_range(self):
        with pytest.raises(ValueError, match="empty feasible range"):
            Constraint(name="c", objective_function=lambda c: 1.0, gte=10.0, lte=5.0)

    def test_rejects_single_point_feasible_range(self):
        with pytest.raises(ValueError, match="single-point feasible range"):
            Constraint(name="c", objective_function=lambda c: 1.0, gte=5.0, lte=5.0)

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_rejects_non_finite_gte(self, bad_value):
        with pytest.raises(ValueError, match="must be a finite number"):
            Constraint(name="c", objective_function=lambda c: 1.0, gte=bad_value)

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_rejects_non_finite_lte(self, bad_value):
        with pytest.raises(ValueError, match="must be a finite number"):
            Constraint(name="c", objective_function=lambda c: 1.0, lte=bad_value)


class TestConstraintEvaluation:
    def test_evaluate_caches_value(self, case):
        c = Constraint(name="c", objective_function=lambda case: 7.5, gte=0.0)
        assert c.evaluate(case) == 7.5
        assert c.data_for_case(case) == 7.5

    def test_static_transform_applied(self, case):
        import math

        c = Constraint(
            name="c",
            objective_function=lambda case: math.e,
            gte=0.0,
            static_transform=math.log,
        )
        assert c.evaluate(case) == pytest.approx(1.0)

    def test_metric_values_for_case(self, case):
        c = Constraint(name="pressure", objective_function=lambda case: 50.0, gte=45.0)
        c.evaluate(case)
        assert c.metric_values_for_case(case) == {"pressure": 50.0}

    def test_failed_case_returns_none(self, tmp_path):
        d = tmp_path / "failed"
        d.mkdir()
        case = Case(d)
        case.success = False
        c = Constraint(name="c", objective_function=lambda case: 1.0, gte=0.0)
        assert c.evaluate(case) is None


class TestAxBackendConstraintIntegration:
    """End-to-end tests that the silent-failure pattern from issue #24 is
    fixed: a constraint metric registered via set_constraints gets its values
    sent to Ax and trials complete cleanly (no 'missing required metric'
    warning, no spurious failure marking)."""

    def _read_value(self, case, key):
        return float(case.read_metadata()["optimizer-suggestion"][key]["value"])

    def test_constraint_metric_registered_as_tracking_metric(self, make_dim):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [Objective(name="score", minimize=True, objective_function=lambda c: 1.0)]
        )
        backend.set_constraints(
            [Constraint(name="pressure", objective_function=lambda c: 50.0, gte=45.0)]
        )
        backend.initialize()

        # Both objective and constraint metrics live in experiment.metrics.
        assert "pressure" in backend.client.experiment.metrics
        # ...and the OutcomeConstraint references the constraint metric.
        opt_cfg = backend.client.experiment.optimization_config
        constraint_names = [oc.metric.name for oc in opt_cfg.outcome_constraints]
        assert constraint_names == ["pressure"]
        assert opt_cfg.outcome_constraints[0].relative is False

    def test_constraint_emits_outcome_constraint_in_optimization_config(self, make_dim):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [Objective(name="score", minimize=True, objective_function=lambda c: 1.0)]
        )
        backend.set_constraints(
            [
                Constraint(
                    name="pressure",
                    objective_function=lambda c: 50.0,
                    gte=45.0,
                    lte=100.0,
                )
            ]
        )
        backend.initialize()

        opt_cfg = backend.client.experiment.optimization_config
        bounds_by_op = {
            (oc.op.name, oc.bound): oc for oc in opt_cfg.outcome_constraints
        }
        assert ("GEQ", 45.0) in bounds_by_op
        assert ("LEQ", 100.0) in bounds_by_op
        assert all(oc.relative is False for oc in opt_cfg.outcome_constraints)
        assert all(oc.metric.name == "pressure" for oc in opt_cfg.outcome_constraints)

    def test_bulut_pattern_completes_trial_without_silent_failure(
        self, tmp_path, caplog, make_suggestion_case
    ):
        """Reproducer for issue #24: constraint via set_constraints should
        cause every trial to complete cleanly, not be silently marked failed."""
        backend = AxBackend()
        backend.random_seed = 0
        backend.initialization_trials = 2
        backend.set_search_space(
            [
                Dimension.range(
                    name="x",
                    link=DictionaryLink("constant/setup").entry("x"),
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )
        objective = Objective(
            name="totalHeatLoss",
            minimize=True,
            objective_function=lambda case: self._read_value(case, "x"),
        )
        constraint = Constraint(
            name="pressure",
            objective_function=lambda case: 50.0,  # constant, satisfies gte=45
            gte=45.0,
        )
        backend.set_objectives([objective])
        backend.set_constraints([constraint])
        backend.initialize()

        # Drive a few trials end-to-end.
        cases = []
        with caplog.at_level(logging.WARNING):
            for cycle in range(3):
                params = backend.ask(max_cases=1)[0]
                p = {dim.name: value for dim, value in params.items()}
                case = make_suggestion_case(
                    tmp_path, f"trial-{cycle:02d}", {"x": p["x"]}
                )
                cases.append(case)
                objective.batch_evaluate(cases)
                constraint.batch_evaluate(cases)
                backend.tell(cases)

        # The smoking-gun warning Ax emits when a metric in raw_data is missing.
        assert not any(
            "Marking the trial as failed" in record.message for record in caplog.records
        ), (
            "Trials should not be silently marked failed; constraint values are in raw_data"
        )

        # All trials completed.
        for case in cases:
            trial = backend.client.experiment.trials[
                backend._trial_index_case_mapping[case]
            ]
            assert trial.status.is_completed

    def test_no_constraints_means_no_optimization_config_override(self, make_dim):
        """Sanity: if no constraints and no scalarized objective, Ax's default
        config is left in place — no unnecessary overrides."""
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [Objective(name="score", minimize=True, objective_function=lambda c: 1.0)]
        )
        backend.initialize()

        opt_cfg = backend.client.experiment.optimization_config
        assert opt_cfg.outcome_constraints == []

    def test_metric_name_clash_between_objective_and_constraint_rejected(
        self, make_dim
    ):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [Objective(name="dup", minimize=True, objective_function=lambda c: 1.0)]
        )
        backend.set_constraints(
            [Constraint(name="dup", objective_function=lambda c: 1.0, gte=0.0)]
        )
        with pytest.raises(ValueError, match="unique"):
            backend._verify_configuration()


class TestSetConstraintsTypeCheck:
    """Guards the `set_constraints` arg so misuse fails loudly at registration
    time instead of crashing later with an AttributeError inside evaluate().
    """

    def test_rejects_objective_in_list(self):
        backend = AxBackend()
        obj = Objective(name="o", minimize=True, objective_function=lambda c: 1.0)
        with pytest.raises(TypeError, match="must be a Constraint"):
            backend.set_constraints([obj])  # type: ignore[list-item]

    def test_rejects_string_in_list(self):
        backend = AxBackend()
        with pytest.raises(TypeError, match="must be a Constraint"):
            backend.set_constraints(["not a constraint"])  # type: ignore[list-item]

    def test_rejects_non_iterable_non_constraint(self):
        backend = AxBackend()
        with pytest.raises(TypeError, match="expects a Constraint or an iterable"):
            backend.set_constraints(42)  # type: ignore[arg-type]

    def test_accepts_single_constraint_instance(self):
        backend = AxBackend()
        c = Constraint(name="c", objective_function=lambda c: 1.0, gte=0.0)
        backend.set_constraints(c)  # type: ignore[arg-type]
        assert backend.constraints == [c]

    def test_accepts_tuple_of_constraints(self):
        backend = AxBackend()
        c1 = Constraint(name="c1", objective_function=lambda c: 1.0, gte=0.0)
        c2 = Constraint(name="c2", objective_function=lambda c: 2.0, lte=5.0)
        backend.set_constraints((c1, c2))  # type: ignore[arg-type]
        assert backend.constraints == [c1, c2]


@pytest.mark.slow
class TestMOOWithConstraint:
    """End-to-end: Ax MOO (two Objectives, no ScalarizedObjective) with a
    tracking Constraint. Exercises the qEHVI + OutcomeConstraint path that
    none of the single-objective canaries hit. This is the exact pattern from
    issue #24 (two heat objectives + a pressure constraint).

    Setup: two paraboloids with minima at (0, 0) and (3, 0), both minimized.
    The feasibility constraint y >= 1.0 pushes the Pareto front onto the
    y=1 line, so the reachable corners are (0, 1) (f1=1, f2=10) and (3, 1)
    (f1=10, f2=1). If the constraint is being respected, BO explores both
    corners. If it's being ignored (OutcomeConstraint not installed or not
    honored by qEHVI), BO drifts toward (0, 0) and (3, 0) — infeasible — and
    the feasible trials are limited to whatever Sobol init happened to sample.
    """

    N_INIT = 5
    N_BO = 40
    PARETO_CORNER_TOLERANCE = 3.0  # near-corner threshold for f1/f2

    def test_bo_explores_constrained_pareto_front(self, tmp_path, make_suggestion_case):
        backend = AxBackend()
        backend.random_seed = 0
        backend.initialization_trials = self.N_INIT

        def _read(case, key):
            return float(case.read_metadata()["optimizer-suggestion"][key]["value"])

        # Thresholds bound qEHVI's reference point so hypervolume is
        # well-defined across the search space.
        f1 = Objective(
            name="f1",
            minimize=True,
            objective_function=lambda case: (
                _read(case, "x") ** 2 + _read(case, "y") ** 2
            ),
            threshold=15.0,
        )
        f2 = Objective(
            name="f2",
            minimize=True,
            objective_function=lambda case: (
                (_read(case, "x") - 3.0) ** 2 + _read(case, "y") ** 2
            ),
            threshold=15.0,
        )
        y_bound = Constraint(
            name="y_bound",
            objective_function=lambda case: _read(case, "y"),
            gte=1.0,
        )

        backend.set_search_space(
            [
                Dimension.range(
                    name="x",
                    link=DictionaryLink("constant/setup").entry("x"),
                    lower=-2.0,
                    upper=5.0,
                ),
                Dimension.range(
                    name="y",
                    link=DictionaryLink("constant/setup").entry("y"),
                    lower=-2.0,
                    upper=3.0,
                ),
            ]
        )
        backend.set_objectives([f1, f2])
        backend.set_constraints([y_bound])
        backend.initialize()

        cases: list[Case] = []
        feasible_f1_values: list[float] = []
        feasible_f2_values: list[float] = []

        for cycle in range(self.N_INIT + self.N_BO):
            suggestion = backend.ask(max_cases=1)[0]
            params = {dim.name: value for dim, value in suggestion.items()}
            case = make_suggestion_case(tmp_path, f"trial-{cycle:02d}", params)
            cases.append(case)
            f1.batch_evaluate(cases)
            f2.batch_evaluate(cases)
            y_bound.batch_evaluate(cases)
            backend.tell(cases)

            y_val = y_bound.data_for_case(case)
            if y_val is not None and y_val >= 1.0:
                feasible_f1_values.append(f1.data_for_case(case))
                feasible_f2_values.append(f2.data_for_case(case))

        assert feasible_f1_values, (
            f"BO produced no feasible trials across {self.N_INIT + self.N_BO} "
            f"cycles. Constraint values likely aren't reaching Ax, or qEHVI "
            f"isn't using the OutcomeConstraint at all."
        )

        min_f1 = min(feasible_f1_values)
        min_f2 = min(feasible_f2_values)

        assert min_f1 < self.PARETO_CORNER_TOLERANCE, (
            f"BO didn't find a feasible trial near the (0, 1) Pareto corner. "
            f"Best feasible f1={min_f1:.3f} (target < {self.PARETO_CORNER_TOLERANCE}). "
            f"qEHVI is probably not exploring that end of the feasible front."
        )
        assert min_f2 < self.PARETO_CORNER_TOLERANCE, (
            f"BO didn't find a feasible trial near the (3, 1) Pareto corner. "
            f"Best feasible f2={min_f2:.3f} (target < {self.PARETO_CORNER_TOLERANCE}). "
            f"qEHVI is probably not exploring that end of the feasible front."
        )


class TestMOOWithObjectiveBounds:
    """Top-level MOO with Objective(..., gte/lte) takes a separate Ax path from
    single-objective bounds: the config must stay MultiObjectiveOptimizationConfig
    while adding OutcomeConstraints on the derived bound tracking metric.
    """

    def test_bounded_objective_preserves_moo_config_and_completes_trial(
        self, tmp_path, make_suggestion_case
    ):
        from ax.core.optimization_config import MultiObjectiveOptimizationConfig

        backend = AxBackend()
        backend.random_seed = 0
        backend.initialization_trials = 1

        def _read(case, key):
            return float(case.read_metadata()["optimizer-suggestion"][key]["value"])

        drag = Objective(
            name="drag",
            minimize=True,
            objective_function=lambda case: _read(case, "x") ** 2,
            lte=0.5,
            threshold=2.0,
        )
        lift = Objective(
            name="lift",
            minimize=False,
            objective_function=lambda case: 1.0 + _read(case, "x"),
            gte=1.25,
            threshold=0.0,
        )

        backend.set_search_space(
            [
                Dimension.range(
                    name="x",
                    link=DictionaryLink("constant/setup").entry("x"),
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )
        backend.set_objectives([drag, lift])
        backend.initialize()

        opt_cfg = backend.client.experiment.optimization_config
        assert isinstance(opt_cfg, MultiObjectiveOptimizationConfig)
        assert {
            (oc.metric.name, oc.op.name, oc.bound, oc.relative)
            for oc in opt_cfg.outcome_constraints
        } == {
            ("drag__bound", "LEQ", 0.5, False),
            ("lift__bound", "GEQ", 1.25, False),
        }

        case = make_suggestion_case(tmp_path, "finished", {"x": 0.5})
        drag.batch_evaluate([case])
        lift.batch_evaluate([case])
        backend.tell([case])

        trial = backend.client.get_trial(backend._trial_index_case_mapping[case])
        assert trial.status.is_completed
        metric_names = set(backend.client.experiment.fetch_data().df["metric_name"])
        assert {"drag", "drag__bound", "lift", "lift__bound"} <= metric_names


@pytest.mark.slow
class TestConstraintConvergenceCanary:
    """BO on an unconstrained quadratic whose minimum lies outside a feasible
    region defined by a Constraint. If Ax's OutcomeConstraint integration
    regresses (bound not applied, or the tracking metric isn't registered),
    BO drifts toward the unconstrained optimum and never finds a feasible
    low-objective point; the assertion on best-feasible objective value
    catches that.
    """

    N_INIT = 5
    N_BO = 20
    CONSTRAINED_OPT = 4.0  # f = x1**2 + x2**2 at (x1=2, x2=0), subject to x1 >= 2
    TOLERANCE = 2.0  # ~50% slack; broken path typically lands at 8-15

    def test_bo_concentrates_in_feasible_region(self, tmp_path, make_suggestion_case):
        backend = AxBackend()
        backend.random_seed = 0
        backend.initialization_trials = self.N_INIT

        def _read(case, key):
            return float(case.read_metadata()["optimizer-suggestion"][key]["value"])

        objective = Objective(
            name="f",
            minimize=True,
            objective_function=lambda case: (
                _read(case, "x1") ** 2 + _read(case, "x2") ** 2
            ),
        )
        constraint = Constraint(
            name="x1_bound",
            objective_function=lambda case: _read(case, "x1"),
            gte=2.0,
        )

        backend.set_search_space(
            [
                Dimension.range(
                    name="x1",
                    link=DictionaryLink("constant/setup").entry("x1"),
                    lower=-5.0,
                    upper=5.0,
                ),
                Dimension.range(
                    name="x2",
                    link=DictionaryLink("constant/setup").entry("x2"),
                    lower=-5.0,
                    upper=5.0,
                ),
            ]
        )
        backend.set_objectives([objective])
        backend.set_constraints([constraint])
        backend.initialize()

        cases = []
        best_feasible = float("inf")

        for cycle in range(self.N_INIT + self.N_BO):
            suggestion = backend.ask(max_cases=1)[0]
            params = {dim.name: value for dim, value in suggestion.items()}
            case = make_suggestion_case(tmp_path, f"trial-{cycle:02d}", params)
            cases.append(case)
            objective.batch_evaluate(cases)
            constraint.batch_evaluate(cases)
            backend.tell(cases)

            f_val = objective.data_for_case(case)
            c_val = constraint.data_for_case(case)
            assert f_val is not None and c_val is not None
            if c_val >= 2.0:
                best_feasible = min(best_feasible, f_val)

        assert best_feasible - self.CONSTRAINED_OPT < self.TOLERANCE, (
            f"BO failed to find a near-optimal feasible point. Best feasible "
            f"f: {best_feasible:.4f} (target: {self.CONSTRAINED_OPT:.4f} "
            f"± {self.TOLERANCE}). Likely that Ax's OutcomeConstraint is no "
            f"longer being installed or respected at acquisition time."
        )


class TestBatchProcessWithConstraints:
    def test_batch_process_persists_constraint_outputs(self, tmp_path, make_dim):
        """`batch_process` must evaluate constraints alongside objectives and
        write their values to `metadata["constraint-outputs"]` with gte/lte
        annotations."""
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [Objective(name="score", minimize=True, objective_function=lambda c: 1.0)]
        )
        backend.set_constraints(
            [
                Constraint(
                    name="pressure",
                    objective_function=lambda c: 50.0,
                    gte=45.0,
                    lte=100.0,
                )
            ]
        )

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        case = Case(case_dir)

        backend.batch_process([case])

        metadata = case.read_metadata()
        assert metadata is not None
        assert metadata["objective-outputs"]["score"]["value"] == 1.0
        assert metadata["constraint-outputs"]["pressure"]["value"] == 50.0
        assert metadata["constraint-outputs"]["pressure"]["gte"] == 45.0
        assert metadata["constraint-outputs"]["pressure"]["lte"] == 100.0

    def test_batch_process_without_constraints_omits_section(self, tmp_path, make_dim):
        """Don't write an empty constraint-outputs section when there are no
        constraints — keeps metadata.toml uncluttered for the common case."""
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [Objective(name="score", minimize=True, objective_function=lambda c: 1.0)]
        )

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        case = Case(case_dir)

        backend.batch_process([case])

        metadata = case.read_metadata()
        assert metadata is not None
        assert "constraint-outputs" not in metadata


class TestOffloadedAcquisitionWithConstraints:
    """Coverage for the acquisition-offload round-trip when constraints are
    registered. The snapshot already includes the constraint metric values
    via `objective_function_outputs`; the receiving side must feed them into
    `raw_data` so the reconstructed experiment doesn't hit the silent-failure
    mode that motivated this PR."""

    def test_data_snapshot_includes_constraint_values(
        self, tmp_path, make_dim, make_suggestion_case
    ):
        backend = AxBackend()
        backend.offload_acquisition = True
        backend.set_search_space([make_dim(name="x")])
        objective = Objective(
            name="score", minimize=True, objective_function=lambda c: 1.0
        )
        constraint = Constraint(
            name="pressure", objective_function=lambda c: 55.0, gte=45.0
        )
        backend.set_objectives([objective])
        backend.set_constraints([constraint])
        backend.initialize()

        case = make_suggestion_case(tmp_path, "finished", {"x": 0.5})
        objective.batch_evaluate([case])
        constraint.batch_evaluate([case])

        _, data_snapshot = backend.prepare_for_acquisition_offload([case], [], tmp_path)
        snapshot = json.loads(data_snapshot.read_text())
        objectives_dict = snapshot["finished_cases"][case.name]["objectives"]
        # Both objective and constraint metric values are in the snapshot.
        assert objectives_dict == {"score": 1.0, "pressure": 55.0}

    def test_offloaded_round_trip_with_constraint(
        self, tmp_path, make_dim, make_suggestion_case
    ):
        backend = AxBackend()
        backend.offload_acquisition = True
        backend.random_seed = 0
        backend.initialization_trials = 1
        backend.set_search_space([make_dim(name="x")])
        objective = Objective(
            name="score", minimize=True, objective_function=lambda c: 1.0
        )
        constraint = Constraint(
            name="pressure", objective_function=lambda c: 55.0, gte=45.0
        )
        backend.set_objectives([objective])
        backend.set_constraints([constraint])
        backend.initialize()

        case = make_suggestion_case(tmp_path, "finished", {"x": 0.5})
        objective.batch_evaluate([case])
        constraint.batch_evaluate([case])

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
        assert len(result["parametrizations"]) == 1


class TestOffloadedAcquisitionWithObjectiveBounds:
    """Acquisition offload must serialize and replay the derived bound alias for
    bounded Objectives, not only explicit Constraint metrics.
    """

    def test_offloaded_round_trip_with_bounded_objective(
        self, tmp_path, make_dim, make_suggestion_case
    ):
        backend = AxBackend()
        backend.offload_acquisition = True
        backend.random_seed = 0
        backend.initialization_trials = 1
        backend.set_search_space([make_dim(name="x")])
        objective = Objective(
            name="score",
            minimize=True,
            objective_function=lambda c: 0.4,
            lte=0.5,
        )
        backend.set_objectives([objective])
        backend.initialize()

        case = make_suggestion_case(tmp_path, "finished", {"x": 0.5})
        objective.batch_evaluate([case])

        model_snapshot, data_snapshot = backend.prepare_for_acquisition_offload(
            [case], [], tmp_path
        )
        snapshot = json.loads(data_snapshot.read_text())
        assert snapshot["finished_cases"][case.name]["objectives"] == {
            "score": 0.4,
            "score__bound": 0.4,
        }

        output_path = tmp_path / "acquisition_result.json"
        AxBackend.offloaded_acquisition(
            model_snapshot=model_snapshot,
            data_snapshot=data_snapshot,
            num_trials=1,
            output_path=output_path,
        )

        result = json.loads(output_path.read_text())
        assert result["optimizer"] == "AxBackend"
        assert len(result["parametrizations"]) == 1


@pytest.mark.slow
class TestSessionLoopWithConstraint:
    """End-to-end: drive Session.local_optimization with a real AxBackend
    configured with a Constraint. The manager/job cycle is replaced by
    directly staging 'completed' cases in the archival dir between iterations.

    Catches orchestration bugs where constraint values might be dropped in
    the finished-case → batch_process → tell handoff, or where the next ask()
    ignores constraint values that did reach the backend. The single-object
    canary above drives ask/tell directly; this one drives the Session layer.
    """

    N_INIT = 5
    N_CYCLES = 15
    FEASIBLE_OPT_F = 1.0  # x=1, f=x**2
    TOLERANCE = 3.0  # slack for surrogate noise + Sobol warm-up

    def _make_foam_dir(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "constant").mkdir(exist_ok=True)
        (path / "system").mkdir(exist_ok=True)
        return path

    def _stage_finished_case(self, session, case_idx, x_val):
        """Put a 'completed' case in the archival dir with suggestion metadata
        set — simulates a job that ran and finished outside this test."""
        name = f"job_{case_idx:05d}_{case_idx:08x}"
        case_dir = self._make_foam_dir(session.archival_dir / name)
        case = Case(case_dir)
        case.update_metadata(
            {"x": {"value": float(x_val)}},
            entry_header="optimizer-suggestion",
        )
        case.success = True
        case.persist_to_file()
        return case

    def test_session_loop_drives_ax_with_constraint(self, tmp_path):
        import random

        from flowboost.session.session import Session

        session = Session(name="t", data_dir=tmp_path / "session")

        # Reconfigure the backend Session created by default.
        backend = AxBackend()
        backend.random_seed = 0
        backend.initialization_trials = self.N_INIT
        backend.set_search_space(
            [
                Dimension.range(
                    name="x",
                    link=DictionaryLink("constant/setup").entry("x"),
                    lower=-5.0,
                    upper=5.0,
                ),
            ]
        )

        def _read(case, key):
            return float(case.read_metadata()["optimizer-suggestion"][key]["value"])

        # Unconstrained optimum at x=0 (f=0); constraint x >= 1 shifts
        # constrained optimum to x=1 (f=1). If constraint values are dropped
        # anywhere in the Session → Backend handoff, BO drifts toward x=0.
        objective = Objective(
            name="f",
            minimize=True,
            objective_function=lambda c: _read(c, "x") ** 2,
        )
        constraint = Constraint(
            name="x_bound",
            objective_function=lambda c: _read(c, "x"),
            gte=1.0,
        )
        backend.set_objectives([objective])
        backend.set_constraints([constraint])
        session.backend = backend
        backend.initialize()

        # Seed the archival dir with Sobol-like starting data. Without this,
        # the first local_optimization call has nothing to tell() the backend
        # with and Ax just generates more Sobol points instead of exercising
        # the BO + constraint path we care about.
        rng = random.Random(42)
        for i in range(self.N_INIT):
            self._stage_finished_case(session, i, rng.uniform(-5.0, 5.0))

        # Drive the session loop: each cycle produces a suggestion, which we
        # stage as finished for the next iteration. Exercises the full
        # batch_process → tell → ask pipeline.
        x_trajectory: list[float] = []
        best_feasible = float("inf")

        for cycle in range(self.N_CYCLES):
            suggestions = session.local_optimization(num_new_cases=1)
            assert len(suggestions) == 1, (
                f"cycle {cycle}: expected 1 suggestion, got {len(suggestions)}"
            )

            dim_to_value = suggestions[0]
            x_val = float(next(iter(dim_to_value.values())))
            x_trajectory.append(x_val)

            # Stage this suggestion as a finished case for the next cycle.
            self._stage_finished_case(session, 1000 + cycle, x_val)

            f_val = x_val**2
            if x_val >= 1.0:
                best_feasible = min(best_feasible, f_val)

        # Loop completed without raising — orchestration handshake is intact.
        # Best feasible f close to the constrained optimum — constraint
        # information actually influenced acquisition across the Session.
        assert best_feasible - self.FEASIBLE_OPT_F < self.TOLERANCE, (
            f"Session-driven BO didn't find a near-optimal feasible point. "
            f"Best feasible f={best_feasible:.4f} "
            f"(target: {self.FEASIBLE_OPT_F} ± {self.TOLERANCE}). "
            f"Trajectory: {['%.3f' % x for x in x_trajectory]}. "
            f"If constraint values were lost between Case metadata and "
            f"backend.tell(), BO would drift toward x=0 and this would fail."
        )

        # Sanity: late-cycle trials should cluster near the feasible region.
        # Soft OutcomeConstraints don't enforce strict feasibility — Ax's
        # acquisition lands arbitrarily close to the bound (e.g. x=0.999 with
        # gte=1.0), so a 10% tolerance accommodates that while still catching
        # "BO ignores the constraint and drifts toward the unconstrained
        # optimum at x=0".
        late_near_feasible = sum(1 for x in x_trajectory[-5:] if x >= 0.9)
        assert late_near_feasible >= 2, (
            f"Only {late_near_feasible}/5 of the last cycles landed at or "
            f"above the constraint boundary. Late trajectory: "
            f"{['%.3f' % x for x in x_trajectory[-5:]]}. BO doesn't appear "
            f"to be steering toward the feasible region."
        )


@pytest.mark.slow
class TestSessionLoopWithObjectiveBounds:
    """End-to-end Session loop for Objective(..., gte/lte) sugar.

    This catches regressions where the derived `{objective}__bound` metric is
    registered at Ax setup time but lost in the Session → batch_process → tell
    handoff during repeated stateless replay.
    """

    N_INIT = 5
    N_CYCLES = 15
    CONSTRAINED_OPT = 1.0  # f=x**2 with f >= 1.0; optimum lies at x=±1.
    TOLERANCE = 0.5
    NEAR_BOUNDARY_VALUE = 0.9

    def _make_foam_dir(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "constant").mkdir(exist_ok=True)
        (path / "system").mkdir(exist_ok=True)
        return path

    def _stage_finished_case(self, session, case_idx, x_val):
        name = f"job_{case_idx:05d}_{case_idx:08x}"
        case_dir = self._make_foam_dir(session.archival_dir / name)
        case = Case(case_dir)
        case.update_metadata(
            {"x": {"value": float(x_val)}},
            entry_header="optimizer-suggestion",
        )
        case.success = True
        case.persist_to_file()
        return case

    def test_session_loop_drives_ax_with_bounded_objective(self, tmp_path):
        import random

        from flowboost.session.session import Session

        session = Session(name="t", data_dir=tmp_path / "session")
        backend = AxBackend()
        backend.random_seed = 0
        backend.initialization_trials = self.N_INIT
        backend.set_search_space(
            [
                Dimension.range(
                    name="x",
                    link=DictionaryLink("constant/setup").entry("x"),
                    lower=-5.0,
                    upper=5.0,
                ),
            ]
        )

        def _read(case, key):
            return float(case.read_metadata()["optimizer-suggestion"][key]["value"])

        objective = Objective(
            name="f",
            minimize=True,
            objective_function=lambda c: _read(c, "x") ** 2,
            gte=1.0,
        )
        backend.set_objectives([objective])
        session.backend = backend
        backend.initialize()

        rng = random.Random(42)
        for i in range(self.N_INIT):
            self._stage_finished_case(session, i, rng.uniform(-5.0, 5.0))

        x_trajectory: list[float] = []
        best_near_boundary = float("inf")
        for cycle in range(self.N_CYCLES):
            suggestions = session.local_optimization(num_new_cases=1)
            assert len(suggestions) == 1, (
                f"cycle {cycle}: expected 1 suggestion, got {len(suggestions)}"
            )
            x_val = float(next(iter(suggestions[0].values())))
            x_trajectory.append(x_val)
            self._stage_finished_case(session, 1000 + cycle, x_val)

            f_val = x_val**2
            if f_val >= self.NEAR_BOUNDARY_VALUE:
                best_near_boundary = min(best_near_boundary, f_val)

        failed_trials = [
            trial
            for trial in backend.client.experiment.trials.values()
            if trial.status.is_failed
        ]
        assert failed_trials == []

        metric_names = set(backend.client.experiment.fetch_data().df["metric_name"])
        assert {"f", "f__bound"} <= metric_names

        assert best_near_boundary - self.CONSTRAINED_OPT < self.TOLERANCE, (
            f"Session-driven bounded Objective BO didn't find a near-boundary "
            f"point. Best near-boundary f={best_near_boundary:.4f} "
            f"(target: {self.CONSTRAINED_OPT} ± {self.TOLERANCE}). "
            f"Trajectory: {['%.3f' % x for x in x_trajectory]}."
        )

        late_near_feasible = sum(
            1 for x in x_trajectory[-5:] if x**2 >= self.NEAR_BOUNDARY_VALUE
        )
        assert late_near_feasible >= 3, (
            f"Only {late_near_feasible}/5 late cycles landed near the bounded "
            f"objective's feasible boundary. Late trajectory: "
            f"{['%.3f' % x for x in x_trajectory[-5:]]}."
        )


class TestObjectiveBoundsIntegration:
    """`Objective(..., gte=/lte=)` sugar: bound is attached to a derived
    tracking metric at Ax setup time so that Ax never sees a constraint on
    an objective metric directly. These tests pin that wiring and assert the
    sugar behaves identically to the explicit `Objective + Constraint` dual
    registration under a fixed random seed.
    """

    def test_derived_tracking_metric_registered_on_experiment(self, make_dim):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [
                Objective(
                    name="drag",
                    minimize=True,
                    objective_function=lambda c: 0.04,
                    lte=0.05,
                )
            ]
        )
        backend.initialize()

        assert "drag" in backend.client.experiment.metrics
        assert "drag__bound" in backend.client.experiment.metrics

    def test_outcome_constraint_wired_to_derived_metric(self, make_dim):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [
                Objective(
                    name="drag",
                    minimize=True,
                    objective_function=lambda c: 0.04,
                    gte=0.01,
                    lte=0.05,
                )
            ]
        )
        backend.initialize()

        opt_cfg = backend.client.experiment.optimization_config
        wired = {
            (oc.metric.name, oc.op.name, oc.bound) for oc in opt_cfg.outcome_constraints
        }
        assert wired == {("drag__bound", "GEQ", 0.01), ("drag__bound", "LEQ", 0.05)}
        assert all(oc.relative is False for oc in opt_cfg.outcome_constraints)

    def test_derived_name_collision_rejected(self, make_dim):
        """If the user registers a Constraint with the same name as an
        Objective's derived bound metric, `_verify_configuration` must catch
        the clash before Ax sees a duplicate metric."""
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [
                Objective(
                    name="drag",
                    minimize=True,
                    objective_function=lambda c: 0.04,
                    lte=0.05,
                )
            ]
        )
        backend.set_constraints(
            [
                Constraint(
                    name="drag__bound",
                    objective_function=lambda c: 0.04,
                    lte=0.05,
                )
            ]
        )
        with pytest.raises(ValueError, match="unique"):
            backend._verify_configuration()

    def test_batch_process_persists_objective_bounds_in_metadata(
        self, tmp_path, make_dim
    ):
        """The metadata-facing view of a bounded Objective is a single
        objective-outputs entry carrying `gte`/`lte` alongside `value`, not
        a separate constraint-outputs entry for the derived name."""
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            [
                Objective(
                    name="drag",
                    minimize=True,
                    objective_function=lambda c: 0.04,
                    lte=0.05,
                )
            ]
        )

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        case = Case(case_dir)
        backend.batch_process([case])

        metadata = case.read_metadata()
        assert metadata is not None
        assert metadata["objective-outputs"]["drag"]["value"] == 0.04
        assert metadata["objective-outputs"]["drag"]["lte"] == 0.05
        assert metadata["objective-outputs"]["drag"]["minimize"] is True
        # The derived name is an Ax-layer artifact; it should not leak into
        # the user-facing constraint-outputs section.
        assert "constraint-outputs" not in metadata


@pytest.mark.slow
class TestObjectiveBoundsEquivalence:
    """Sugar `Objective(..., gte=X)` must produce the same suggestion
    trajectory as the explicit `Objective + Constraint("__bound", ...)`
    pattern under a fixed random seed. Anything else would mean the sugar
    has a subtly different Ax configuration under the hood.
    """

    N_INIT = 3
    N_BO = 10

    def _quad(self, case):
        return float(case.read_metadata()["optimizer-suggestion"]["x"]["value"]) ** 2

    def _run(self, tmp_path, subdir, setup_fn, make_suggestion_case):
        case_parent = tmp_path / subdir
        case_parent.mkdir()

        backend, objective, constraint = setup_fn()
        backend.random_seed = 0
        backend.initialization_trials = self.N_INIT
        backend.set_search_space(
            [
                Dimension.range(
                    name="x",
                    link=DictionaryLink("constant/setup").entry("x"),
                    lower=-5.0,
                    upper=5.0,
                ),
            ]
        )
        backend.initialize()

        trajectory = []
        cases = []
        for cycle in range(self.N_INIT + self.N_BO):
            suggestion = backend.ask(max_cases=1)[0]
            params = {d.name: v for d, v in suggestion.items()}
            case = make_suggestion_case(case_parent, f"trial-{cycle:02d}", params)
            cases.append(case)
            objective.batch_evaluate(cases)
            if constraint is not None:
                constraint.batch_evaluate(cases)
            backend.tell(cases)
            trajectory.append(float(params["x"]))
        return trajectory

    def test_sugar_matches_explicit_dual_registration(
        self, tmp_path, make_suggestion_case
    ):
        def sugar():
            backend = AxBackend()
            obj = Objective(
                name="f",
                minimize=True,
                objective_function=self._quad,
                gte=2.0,
            )
            backend.set_objectives([obj])
            return backend, obj, None

        def explicit():
            backend = AxBackend()
            obj = Objective(name="f", minimize=True, objective_function=self._quad)
            con = Constraint(
                name="f__bound",
                objective_function=self._quad,
                gte=2.0,
            )
            backend.set_objectives([obj])
            backend.set_constraints([con])
            return backend, obj, con

        sugar_traj = self._run(tmp_path, "sugar", sugar, make_suggestion_case)
        explicit_traj = self._run(tmp_path, "explicit", explicit, make_suggestion_case)

        assert sugar_traj == pytest.approx(explicit_traj, abs=1e-9), (
            f"Sugar and explicit-dual trajectories diverge. "
            f"Sugar: {[f'{x:.4f}' for x in sugar_traj]} "
            f"Explicit: {[f'{x:.4f}' for x in explicit_traj]}. "
            f"The sugar must produce an identical Ax configuration."
        )


class TestScalarizedInnerBoundsIntegration:
    """Inner Objectives inside a ScalarizedObjective may carry bounds. Each
    bounded inner produces a derived tracking metric at the Ax layer, with
    OutcomeConstraints on it. The scalarized sum remains the optimization
    target; the bound steers acquisition away from violating the inner term.
    """

    def test_derived_tracking_metric_registered_for_inner(self, make_dim):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        inner_bounded = Objective(
            name="drag",
            minimize=True,
            objective_function=lambda c: 0.04,
            lte=0.05,
        )
        inner_plain = Objective(
            name="lift",
            minimize=False,
            objective_function=lambda c: 1.0,
        )
        backend.set_objectives(
            ScalarizedObjective(
                "ratio",
                minimize=False,
                objectives=[inner_plain, inner_bounded],
                weights=[0.7, -0.3],
            )
        )
        backend.initialize()

        metrics = backend.client.experiment.metrics
        assert "lift" in metrics
        assert "drag" in metrics
        assert "drag__bound" in metrics
        assert "lift__bound" not in metrics

    def test_outcome_constraint_wired_to_inner_derived_metric(self, make_dim):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            ScalarizedObjective(
                "ratio",
                minimize=False,
                objectives=[
                    Objective(
                        name="lift",
                        minimize=False,
                        objective_function=lambda c: 1.0,
                    ),
                    Objective(
                        name="drag",
                        minimize=True,
                        objective_function=lambda c: 0.04,
                        lte=0.05,
                    ),
                ],
                weights=[0.7, -0.3],
            )
        )
        backend.initialize()

        opt_cfg = backend.client.experiment.optimization_config
        wired = {
            (oc.metric.name, oc.op.name, oc.bound) for oc in opt_cfg.outcome_constraints
        }
        assert wired == {("drag__bound", "LEQ", 0.05)}
        assert all(oc.relative is False for oc in opt_cfg.outcome_constraints)

    def test_inner_bound_collision_rejected(self, make_dim):
        """A Constraint named `{inner.name}__bound` collides with the derived
        tracking metric the inner's bounds would install."""
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            ScalarizedObjective(
                "ratio",
                minimize=True,
                objectives=[
                    Objective(
                        name="drag",
                        minimize=True,
                        objective_function=lambda c: 0.04,
                        lte=0.05,
                    ),
                    Objective(
                        name="mass", minimize=True, objective_function=lambda c: 1.0
                    ),
                ],
                weights=[1.0, 1.0],
            )
        )
        backend.set_constraints(
            [
                Constraint(
                    name="drag__bound",
                    objective_function=lambda c: 0.04,
                    lte=0.05,
                )
            ]
        )
        with pytest.raises(ValueError, match="unique"):
            backend._verify_configuration()

    def test_batch_process_persists_inner_bounds_in_metadata(self, tmp_path, make_dim):
        """`component_bounds` is an additive sibling of `components`: only
        present when at least one inner carries bounds, preserving the
        existing flat `components` shape for the common case."""
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            ScalarizedObjective(
                "ratio",
                minimize=True,
                objectives=[
                    Objective(
                        name="drag",
                        minimize=True,
                        objective_function=lambda c: 0.04,
                        lte=0.05,
                    ),
                    Objective(
                        name="mass", minimize=True, objective_function=lambda c: 1.0
                    ),
                ],
                weights=[1.0, 1.0],
            )
        )

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        case = Case(case_dir)
        backend.batch_process([case])

        metadata = case.read_metadata()
        assert metadata is not None
        entry = metadata["objective-outputs"]["ratio"]
        assert entry["components"] == {"drag": 0.04, "mass": 1.0}
        assert entry["component_bounds"] == {"drag": {"lte": 0.05}}

    def test_no_component_bounds_when_no_inner_is_bounded(self, tmp_path, make_dim):
        backend = AxBackend()
        backend.set_search_space([make_dim()])
        backend.set_objectives(
            ScalarizedObjective(
                "agg",
                minimize=True,
                objectives=[
                    Objective(
                        name="a", minimize=True, objective_function=lambda c: 2.0
                    ),
                    Objective(
                        name="b", minimize=True, objective_function=lambda c: 3.0
                    ),
                ],
            )
        )
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        case = Case(case_dir)
        backend.batch_process([case])

        metadata = case.read_metadata()
        assert metadata is not None
        entry = metadata["objective-outputs"]["agg"]
        assert entry["components"] == {"a": 2.0, "b": 3.0}
        assert "component_bounds" not in entry


@pytest.mark.slow
class TestScalarizedInnerBoundsEquivalence:
    """Inner-bound sugar (`ScalarizedObjective` with a bounded inner Objective)
    must produce the same BO trajectory under a fixed seed as the explicit
    pattern: unbounded inner Objectives in the ScalarizedObjective plus an
    explicit Constraint on the same underlying callable, named after the
    inner's derived bound metric.
    """

    N_INIT = 3
    N_BO = 10

    def _lift(self, case):
        # Positive, bounded away from zero so maximize is well-behaved.
        x = float(case.read_metadata()["optimizer-suggestion"]["x"]["value"])
        return 1.0 + x * 0.1

    def _drag(self, case):
        # x^2 puts the unconstrained minimum at x=0 (drag=0); the lte=0.5
        # bound becomes binding once x**2 approaches 0.5 (x~0.7).
        x = float(case.read_metadata()["optimizer-suggestion"]["x"]["value"])
        return x**2

    def _run(self, tmp_path, subdir, setup_fn, make_suggestion_case):
        case_parent = tmp_path / subdir
        case_parent.mkdir()

        backend, scalarized, constraint = setup_fn()
        backend.random_seed = 0
        backend.initialization_trials = self.N_INIT
        backend.set_search_space(
            [
                Dimension.range(
                    name="x",
                    link=DictionaryLink("constant/setup").entry("x"),
                    lower=-3.0,
                    upper=3.0,
                ),
            ]
        )
        backend.initialize()

        trajectory = []
        cases = []
        for cycle in range(self.N_INIT + self.N_BO):
            suggestion = backend.ask(max_cases=1)[0]
            params = {d.name: v for d, v in suggestion.items()}
            case = make_suggestion_case(case_parent, f"trial-{cycle:02d}", params)
            cases.append(case)
            scalarized.batch_evaluate(cases)
            if constraint is not None:
                constraint.batch_evaluate(cases)
            backend.tell(cases)
            trajectory.append(float(params["x"]))
        return trajectory

    def test_inner_bound_sugar_matches_explicit_constraint(
        self, tmp_path, make_suggestion_case
    ):
        def sugar():
            backend = AxBackend()
            lift = Objective("lift", minimize=False, objective_function=self._lift)
            drag = Objective(
                "drag", minimize=True, objective_function=self._drag, lte=0.5
            )
            scalarized = ScalarizedObjective(
                "ratio", minimize=False, objectives=[lift, drag], weights=[0.7, -0.3]
            )
            backend.set_objectives(scalarized)
            return backend, scalarized, None

        def explicit():
            backend = AxBackend()
            lift = Objective("lift", minimize=False, objective_function=self._lift)
            drag = Objective("drag", minimize=True, objective_function=self._drag)
            scalarized = ScalarizedObjective(
                "ratio", minimize=False, objectives=[lift, drag], weights=[0.7, -0.3]
            )
            con = Constraint(name="drag__bound", objective_function=self._drag, lte=0.5)
            backend.set_objectives(scalarized)
            backend.set_constraints([con])
            return backend, scalarized, con

        sugar_traj = self._run(tmp_path, "sugar", sugar, make_suggestion_case)
        explicit_traj = self._run(tmp_path, "explicit", explicit, make_suggestion_case)

        assert sugar_traj == pytest.approx(explicit_traj, abs=1e-9), (
            f"Inner-bound sugar and explicit-Constraint trajectories diverge. "
            f"Sugar: {[f'{x:.4f}' for x in sugar_traj]} "
            f"Explicit: {[f'{x:.4f}' for x in explicit_traj]}. The two "
            f"configurations should produce identical Ax experiments."
        )
