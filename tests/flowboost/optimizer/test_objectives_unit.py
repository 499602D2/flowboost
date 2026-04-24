"""Unit tests for Objective and ScalarizedObjective — no OpenFOAM needed."""

import math

import pytest

from flowboost.openfoam.case import Case
from flowboost.optimizer.objectives import Objective, ScalarizedObjective


@pytest.fixture
def failed_case(tmp_path):
    d = tmp_path / "failed_case"
    d.mkdir()
    c = Case(d)
    c.success = False
    return c


class TestObjectiveEvaluate:
    def test_basic_evaluate(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 42.0)
        result = obj.evaluate(case)
        assert result == 42.0

    def test_skips_failed_case(self, failed_case):
        calls = []
        obj = Objective(
            "test", minimize=True, objective_function=lambda c: calls.append(1) or 1.0
        )
        result = obj.evaluate(failed_case)
        assert result is None
        assert calls == [], "objective_function should not be called for failed cases"

    def test_none_return_marks_case_failed(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: None)
        result = obj.evaluate(case)
        assert result is None
        assert case.success is False

    def test_save_value_false_does_not_store(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 5.0)
        obj.evaluate(case, save_value=False)
        assert obj.data_for_case(case) is None

    def test_save_value_true_stores(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 5.0)
        obj.evaluate(case, save_value=True)
        assert obj.data_for_case(case) == 5.0

    def test_kwargs_passed_to_function(self, case):
        def fn(c, *, multiplier):
            return multiplier * 2

        obj = Objective(
            "test",
            minimize=True,
            objective_function=fn,
            objective_function_kwargs={"multiplier": 10},
        )
        assert obj.evaluate(case) == 20

    def test_static_transform_applied(self, case):
        obj = Objective(
            "test",
            minimize=True,
            objective_function=lambda c: math.e,
            static_transform=math.log,
        )
        assert obj.evaluate(case) == pytest.approx(1.0)
        assert obj.data_for_case(case) == pytest.approx(1.0)


class TestObjectiveBatch:
    def test_batch_evaluate(self, tmp_path):
        cases = []
        for i in range(3):
            d = tmp_path / f"case_{i}"
            d.mkdir()
            cases.append(Case(d))

        obj = Objective("test", minimize=True, objective_function=lambda c: 1.0)
        results = obj.batch_evaluate(cases)
        assert results == [1.0, 1.0, 1.0]


class TestObjectiveDataForCase:
    def test_returns_evaluated_value(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 7.0)
        obj.evaluate(case, save_value=True)
        assert obj.data_for_case(case) == 7.0

    def test_missing_case_returns_none(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 7.0)
        assert obj.data_for_case(case) is None


class TestObjectiveMetricValues:
    def test_evaluated(self, case):
        obj = Objective("score", minimize=True, objective_function=lambda c: 3.0)
        obj.evaluate(case)
        assert obj.metric_values_for_case(case) == {"score": 3.0}

    def test_unevaluated(self, case):
        obj = Objective("score", minimize=True, objective_function=lambda c: 3.0)
        assert obj.metric_values_for_case(case) == {}


class TestObjectiveExceptionPropagation:
    def test_exception_propagates(self, case):
        def exploding_fn(c):
            raise ZeroDivisionError("boom")

        obj = Objective("test", minimize=True, objective_function=exploding_fn)
        with pytest.raises(ZeroDivisionError, match="boom"):
            obj.evaluate(case)

    def test_exception_type_preserved(self, case):
        def key_error_fn(c):
            raise KeyError("missing_col")

        obj = Objective("test", minimize=True, objective_function=key_error_fn)
        with pytest.raises(KeyError):
            obj.evaluate(case)


class TestObjectiveBounds:
    def test_defaults_are_none(self):
        obj = Objective("f", minimize=True, objective_function=lambda c: 1.0)
        assert obj.gte is None
        assert obj.lte is None
        assert obj.has_bounds is False

    def test_gte_only(self):
        obj = Objective("f", minimize=True, objective_function=lambda c: 1.0, gte=0.5)
        assert (obj.gte, obj.lte) == (0.5, None)
        assert obj.has_bounds is True

    def test_lte_only(self):
        obj = Objective("f", minimize=True, objective_function=lambda c: 1.0, lte=9.0)
        assert (obj.gte, obj.lte) == (None, 9.0)
        assert obj.has_bounds is True

    def test_both_bounds(self):
        obj = Objective(
            "f", minimize=True, objective_function=lambda c: 1.0, gte=0.0, lte=9.0
        )
        assert (obj.gte, obj.lte) == (0.0, 9.0)
        assert obj.has_bounds is True

    def test_bound_metric_name(self):
        obj = Objective(
            "drag", minimize=True, objective_function=lambda c: 1.0, lte=0.05
        )
        assert obj.bound_metric_name == "drag__bound"

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="empty feasible range"):
            Objective(
                "f", minimize=True, objective_function=lambda c: 1.0, gte=10.0, lte=5.0
            )

    def test_rejects_single_point_bounds(self):
        with pytest.raises(ValueError, match="single-point feasible range"):
            Objective(
                "f", minimize=True, objective_function=lambda c: 1.0, gte=5.0, lte=5.0
            )

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_rejects_non_finite_gte(self, bad_value):
        with pytest.raises(ValueError, match="must be a finite number"):
            Objective(
                "f", minimize=True, objective_function=lambda c: 1.0, gte=bad_value
            )

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_rejects_non_finite_lte(self, bad_value):
        with pytest.raises(ValueError, match="must be a finite number"):
            Objective(
                "f", minimize=True, objective_function=lambda c: 1.0, lte=bad_value
            )

    def test_metric_values_for_case_emits_bound_alias(self, case):
        obj = Objective(
            "drag", minimize=True, objective_function=lambda c: 0.04, lte=0.05
        )
        obj.evaluate(case)
        # The bounded Objective contributes both its own metric and the
        # derived bound tracking metric under the same value, so Ax's raw_data
        # covers the tracking metric the OutcomeConstraint references.
        assert obj.metric_values_for_case(case) == {"drag": 0.04, "drag__bound": 0.04}

    def test_metric_values_for_case_without_bounds_is_unchanged(self, case):
        obj = Objective("drag", minimize=True, objective_function=lambda c: 0.04)
        obj.evaluate(case)
        assert obj.metric_values_for_case(case) == {"drag": 0.04}


class TestScalarizedObjective:
    def _make_objectives(self):
        obj1 = Objective("a", minimize=True, objective_function=lambda c: 2.0)
        obj2 = Objective("b", minimize=True, objective_function=lambda c: 3.0)
        return [obj1, obj2]

    def test_empty_objectives_raises(self):
        with pytest.raises(ValueError, match="at least one objective"):
            ScalarizedObjective("agg", minimize=True, objectives=[])

    def test_mismatched_weights_raises(self):
        objs = self._make_objectives()
        with pytest.raises(ValueError, match="weights must match"):
            ScalarizedObjective("agg", minimize=True, objectives=objs, weights=[1.0])

    def test_default_weights_are_ones(self):
        objs = self._make_objectives()
        agg = ScalarizedObjective("agg", minimize=True, objectives=objs)
        assert agg.weights == [1.0, 1.0]

    def test_evaluate_signed_weighted_sum(self, tmp_path):
        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        agg = ScalarizedObjective(
            "agg",
            minimize=True,
            objectives=self._make_objectives(),
            weights=[0.5, 0.5],
        )
        # 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        assert agg.evaluate(case) == pytest.approx(2.5)
        assert agg.data_for_case(case) == pytest.approx(2.5)

    def test_negative_weight_flips_inner_direction(self, tmp_path):
        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        # Lift-to-drag-style: maximize lift (a), minimize drag (b),
        # maximize the scalarized sum. Negative weight on drag flips its
        # contribution so increasing the scalar means decreasing drag.
        # 0.7 * 2.0 - 0.3 * 3.0 = 1.4 - 0.9 = 0.5
        objs = [
            Objective("a", minimize=False, objective_function=lambda c: 2.0),
            Objective("b", minimize=True, objective_function=lambda c: 3.0),
        ]
        agg = ScalarizedObjective(
            "agg", minimize=False, objectives=objs, weights=[0.7, -0.3]
        )
        assert agg.evaluate(case) == pytest.approx(0.5)

    def test_inconsistent_direction_rejected(self):
        # Maximizing scalar but inner is minimize=True with positive weight:
        # weight sign implies "maximize this term", contradicts inner direction.
        objs = [Objective("a", minimize=True, objective_function=lambda c: 1.0)]
        with pytest.raises(ValueError, match="Inconsistent direction"):
            ScalarizedObjective("agg", minimize=False, objectives=objs, weights=[1.0])

    def test_inner_threshold_rejected(self):
        objs = [
            Objective(
                "a", minimize=True, objective_function=lambda c: 1.0, threshold=0.5
            ),
        ]
        with pytest.raises(ValueError, match="threshold is not honored"):
            ScalarizedObjective("agg", minimize=True, objectives=objs)

    def test_inner_failure_short_circuits(self, tmp_path):
        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        objs = [
            Objective("a", minimize=True, objective_function=lambda c: 2.0),
            Objective("b", minimize=True, objective_function=lambda c: None),
        ]
        agg = ScalarizedObjective("agg", minimize=True, objectives=objs)
        assert agg.evaluate(case) is None
        assert case.success is False

    def test_metric_values_for_case_returns_inner_metrics(self, tmp_path):
        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        agg = ScalarizedObjective(
            "agg", minimize=True, objectives=self._make_objectives()
        )
        agg.evaluate(case)
        # Ax-native scalarization gets per-inner-metric values, not the scalar.
        assert agg.metric_values_for_case(case) == {"a": 2.0, "b": 3.0}

    def test_metric_values_empty_when_unevaluated(self, tmp_path):
        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        agg = ScalarizedObjective(
            "agg", minimize=True, objectives=self._make_objectives()
        )
        assert agg.metric_values_for_case(case) == {}

    def test_data_for_case(self, tmp_path):
        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        agg = ScalarizedObjective(
            "agg", minimize=True, objectives=self._make_objectives()
        )
        agg.evaluate(case)
        # Default weights are [1.0, 1.0] → sum = 5.0
        assert agg.data_for_case(case) == pytest.approx(5.0)


class TestScalarizedObjectiveWeightValidation:
    @pytest.mark.parametrize(
        "bad_weight",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "pos_inf", "neg_inf"],
    )
    def test_non_finite_weight_rejected(self, bad_weight):
        objs = [
            Objective("a", minimize=True, objective_function=lambda c: 1.0),
            Objective("b", minimize=True, objective_function=lambda c: 2.0),
        ]
        with pytest.raises(ValueError, match="not finite"):
            ScalarizedObjective(
                "agg", minimize=True, objectives=objs, weights=[bad_weight, 1.0]
            )

    def test_all_zero_weights_rejected(self):
        objs = [
            Objective("a", minimize=True, objective_function=lambda c: 1.0),
            Objective("b", minimize=True, objective_function=lambda c: 2.0),
        ]
        with pytest.raises(ValueError, match="no non-zero weights"):
            ScalarizedObjective(
                "agg", minimize=True, objectives=objs, weights=[0.0, 0.0]
            )

    def test_zero_weight_warns_and_accepts(self, caplog):
        objs = [
            Objective("a", minimize=True, objective_function=lambda c: 1.0),
            Objective("b", minimize=True, objective_function=lambda c: 2.0),
        ]
        with caplog.at_level("WARNING"):
            agg = ScalarizedObjective(
                "agg", minimize=True, objectives=objs, weights=[0.0, 1.0]
            )
        assert agg.weights == [0.0, 1.0]
        assert any("weight 0" in rec.message for rec in caplog.records)

    def test_zero_weight_skips_direction_check(self, caplog):
        # Mixed-direction inner: if the zero-weight term weren't skipped, the
        # direction check would reject this because w=0 is neither positive
        # nor negative, and the strict `w > 0` comparison misclassifies it.
        objs = [
            Objective("a", minimize=False, objective_function=lambda c: 1.0),
            Objective("b", minimize=True, objective_function=lambda c: 2.0),
        ]
        with caplog.at_level("WARNING"):
            agg = ScalarizedObjective(
                "agg", minimize=True, objectives=objs, weights=[0.0, 1.0]
            )
        assert agg.weights == [0.0, 1.0]

    def test_negative_zero_weight_warns_and_accepts(self, caplog):
        objs = [
            Objective("a", minimize=True, objective_function=lambda c: 1.0),
            Objective("b", minimize=True, objective_function=lambda c: 2.0),
        ]
        with caplog.at_level("WARNING"):
            agg = ScalarizedObjective(
                "agg", minimize=True, objectives=objs, weights=[-0.0, 1.0]
            )
        assert agg.weights == [-0.0, 1.0]
        assert any("weight 0" in rec.message for rec in caplog.records)


class TestScalarizedObjectiveInnerValidation:
    def test_nested_scalarized_rejected(self):
        inner = ScalarizedObjective(
            "inner",
            minimize=True,
            objectives=[
                Objective("a", minimize=True, objective_function=lambda c: 1.0)
            ],
            weights=[1.0],
        )
        with pytest.raises(TypeError, match="Nested scalarization"):
            ScalarizedObjective(
                "outer", minimize=True, objectives=[inner], weights=[1.0]
            )

    def test_non_objective_rejected(self):
        with pytest.raises(TypeError, match="must be an Objective"):
            ScalarizedObjective(
                "agg",
                minimize=True,
                objectives=["not an objective"],  # type: ignore[list-item]
                weights=[1.0],
            )

    def test_duplicate_inner_names_rejected(self):
        objs = [
            Objective("same", minimize=True, objective_function=lambda c: 1.0),
            Objective("same", minimize=True, objective_function=lambda c: 2.0),
        ]
        with pytest.raises(ValueError, match="duplicate inner objective names"):
            ScalarizedObjective("agg", minimize=True, objectives=objs)

    def test_same_instance_twice_rejected_via_name_check(self):
        # Same instance shares the same name by construction, so the
        # duplicate-name check catches it. Pinned here so a future reorder
        # doesn't silently let this through.
        obj = Objective("solo", minimize=True, objective_function=lambda c: 1.0)
        with pytest.raises(ValueError, match="duplicate inner objective names"):
            ScalarizedObjective("agg", minimize=True, objectives=[obj, obj])

    def test_bounded_inner_objective_accepted(self):
        """Inner Objectives may carry bounds. The bound attaches to a derived
        tracking metric on the same inner name, so Ax never sees a constraint
        on an objective metric directly. Semantics: 'optimize the scalarized
        sum, subject to inner_a being within its bound'."""
        objs = [
            Objective("a", minimize=True, objective_function=lambda c: 1.0, lte=0.5),
            Objective("b", minimize=True, objective_function=lambda c: 2.0),
        ]
        agg = ScalarizedObjective("agg", minimize=True, objectives=objs)
        assert agg.objectives[0].has_bounds is True
        assert agg.objectives[0].bound_metric_name == "a__bound"
