"""Unit tests for Objective and AggregateObjective — no OpenFOAM needed."""

import pytest

from flowboost.openfoam.case import Case
from flowboost.optimizer.objectives import (
    AggregateObjective,
    Objective,
    ScikitNormalizationStep,
)


@pytest.fixture
def case(tmp_path):
    d = tmp_path / "test_case"
    d.mkdir()
    return Case(d)


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
        assert obj.data_for_case(case, post_processed=False) is None

    def test_save_value_true_stores(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 5.0)
        obj.evaluate(case, save_value=True)
        assert obj.data_for_case(case, post_processed=False) == 5.0

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

    def test_batch_post_process(self, tmp_path):
        cases = []
        for i in range(3):
            d = tmp_path / f"case_{i}"
            d.mkdir()
            cases.append(Case(d))

        obj = Objective("test", minimize=True, objective_function=lambda c: 1.0)
        outputs = [1.0, 2.0, 3.0]
        result = obj.batch_post_process(cases, outputs)
        assert result == [1.0, 2.0, 3.0]


class TestObjectiveDataForCase:
    def test_pre_processed(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 7.0)
        obj.evaluate(case, save_value=True)
        assert obj.data_for_case(case, post_processed=False) == 7.0

    def test_missing_case_returns_none(self, case):
        obj = Objective("test", minimize=True, objective_function=lambda c: 7.0)
        assert obj.data_for_case(case, post_processed=True) is None
        assert obj.data_for_case(case, post_processed=False) is None


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


class TestAggregateObjective:
    def _make_objectives(self):
        obj1 = Objective("a", minimize=True, objective_function=lambda c: 2.0)
        obj2 = Objective("b", minimize=True, objective_function=lambda c: 3.0)
        return [obj1, obj2]

    def test_empty_objectives_raises(self):
        with pytest.raises(ValueError, match="at least one objective"):
            AggregateObjective(
                "agg",
                minimize=True,
                objectives=[],
                threshold=0.0,
            )

    def test_mismatched_weights_raises(self):
        objs = self._make_objectives()
        with pytest.raises(ValueError, match="weights must match"):
            AggregateObjective(
                "agg", minimize=True, objectives=objs, threshold=0.0, weights=[1.0]
            )

    def test_default_weights(self):
        objs = self._make_objectives()
        agg = AggregateObjective("agg", minimize=True, objectives=objs, threshold=0.0)
        assert agg.weights == [1.0, 1.0]

    def test_aggregate_outputs(self):
        objs = self._make_objectives()
        agg = AggregateObjective(
            "agg", minimize=True, objectives=objs, threshold=0.0, weights=[0.5, 0.5]
        )
        result = agg.aggregate_outputs([(2.0, 3.0)])
        assert result == [2.5]

    def test_aggregate_multiple_cases(self):
        objs = self._make_objectives()
        agg = AggregateObjective(
            "agg", minimize=True, objectives=objs, threshold=0.0, weights=[1.0, 2.0]
        )
        result = agg.aggregate_outputs([(1.0, 1.0), (2.0, 3.0)])
        assert result == [3.0, 8.0]

    def test_batch_process(self, tmp_path):
        obj1 = Objective("a", minimize=True, objective_function=lambda c: 2.0)
        obj2 = Objective("b", minimize=True, objective_function=lambda c: 3.0)

        cases = []
        for i in range(2):
            d = tmp_path / f"case_{i}"
            d.mkdir()
            cases.append(Case(d))

        agg = AggregateObjective(
            "agg",
            minimize=True,
            objectives=[obj1, obj2],
            threshold=0.0,
            weights=[1.0, 1.0],
        )
        result = agg.batch_process(cases)
        assert result == [5.0, 5.0]

    def test_data_for_case(self, tmp_path):
        obj1 = Objective("a", minimize=True, objective_function=lambda c: 2.0)
        obj2 = Objective("b", minimize=True, objective_function=lambda c: 3.0)

        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        agg = AggregateObjective(
            "agg",
            minimize=True,
            objectives=[obj1, obj2],
            threshold=0.0,
        )
        agg.batch_process([case])
        assert agg.data_for_case(case, post_processed=True) == 5.0
        assert agg.data_for_case(case, post_processed=False) == [2.0, 3.0]


class TestScikitNormalizationStep:
    def test_rejects_object_without_fit_transform(self):
        with pytest.raises(ValueError, match="fit_transform"):
            ScikitNormalizationStep({"not": "a normalizer"})

    def test_rejects_lambda(self):
        with pytest.raises(ValueError, match="fit_transform"):
            ScikitNormalizationStep(lambda x: x)

    def test_accepts_valid_normalizer(self):
        from sklearn.preprocessing import MinMaxScaler

        step = ScikitNormalizationStep(MinMaxScaler())
        result = step.evaluate([1.0, 2.0, 3.0])
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)


class TestAddNormalizationStep:
    def test_unsupported_method_raises(self):
        obj = Objective("test", minimize=True, objective_function=lambda c: 1.0)
        with pytest.raises(ValueError, match="Unsupported"):
            obj.add_normalization_step("z-score")

    def test_supported_methods(self):
        for method in ("min-max", "yeo-johnson"):
            obj = Objective("test", minimize=True, objective_function=lambda c: 1.0)
            obj.add_normalization_step(method)
            assert len(obj._post_processing_steps) == 1


class TestExecutePostProcessingSteps:
    def test_mismatched_counts_raises(self, tmp_path):
        d = tmp_path / "case"
        d.mkdir()
        case = Case(d)

        obj = Objective("test", minimize=True, objective_function=lambda c: 1.0)
        with pytest.raises(ValueError, match="Case count"):
            obj.execute_post_processing_steps(cases=[case], outputs=[1.0, 2.0])

    def test_empty_returns_empty(self):
        obj = Objective("test", minimize=True, objective_function=lambda c: 1.0)
        result = obj.execute_post_processing_steps(cases=[], outputs=[])
        assert result == {}

    def test_step_that_changes_cardinality_raises(self, tmp_path):
        cases = []
        for i in range(2):
            d = tmp_path / f"case_{i}"
            d.mkdir()
            cases.append(Case(d))

        obj = Objective("test", minimize=True, objective_function=lambda c: 1.0)
        obj.attach_post_processing_step(lambda outputs: list(outputs)[:1])

        with pytest.raises(ValueError, match="changed output cardinality"):
            obj.execute_post_processing_steps(cases=cases, outputs=[1.0, 2.0])


class TestAggregateBatchPostProcessing:
    def test_step_that_changes_column_cardinality_raises(self, tmp_path):
        cases = []
        for i in range(2):
            d = tmp_path / f"case_{i}"
            d.mkdir()
            cases.append(Case(d))

        obj1 = Objective("a", minimize=True, objective_function=lambda c: 2.0)
        obj2 = Objective("b", minimize=True, objective_function=lambda c: 3.0)
        agg = AggregateObjective(
            "agg",
            minimize=True,
            objectives=[obj1, obj2],
            threshold=0.0,
        )
        agg.attach_post_processing_step(lambda outputs: list(outputs)[:1])

        with pytest.raises(ValueError, match="changed output cardinality"):
            agg.batch_post_process(cases=cases, outputs=[(2.0, 3.0), (4.0, 5.0)])
