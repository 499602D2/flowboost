import math

import polars as pl

from flowboost.openfoam.case import Case
from flowboost.optimizer.objectives import Objective


def max_temp_objective(case: Case) -> int:
    # Returns a value of 1955 for the default data
    df = case.data.simple_function_object_reader("averagePT")
    assert df is not None, "Dataframe could not be loaded"
    return int(df.select(pl.max("volAverage(T)")).item())


def test_objective_initialization():
    """Construction-only sanity check; no objective_function execution."""
    objective = Objective(
        name="test_objective",
        minimize=True,
        objective_function=max_temp_objective,
    )
    assert objective.name == "test_objective"
    assert objective.minimize is True
    assert callable(objective.objective_function)
    assert objective.static_transform is None


def test_evaluate_method(test_case):
    objective = Objective(
        name="test_evaluate", minimize=True, objective_function=max_temp_objective
    )
    result = objective.evaluate(test_case)
    assert result == 1955, f"Result {result} != 1955"


def test_static_transform_applied(test_case):
    """A static_transform is applied per evaluation, before caching."""
    objective = Objective(
        name="logged",
        minimize=True,
        objective_function=max_temp_objective,
        static_transform=math.log,
    )
    result = objective.evaluate(test_case)
    assert result == math.log(1955)
    assert objective.data_for_case(test_case) == math.log(1955)


def test_data_retrieval(test_case):
    """data_for_case returns the cached evaluated value."""
    objective = Objective(
        name="data_retrieval_test",
        minimize=True,
        objective_function=max_temp_objective,
    )
    objective.evaluate(test_case)
    assert objective.data_for_case(test_case) == 1955


def test_evaluation_returns_python_floats(test_case):
    objective = Objective(
        name="float_objective",
        minimize=True,
        objective_function=lambda _: 1.0,
    )

    outputs = objective.batch_evaluate([test_case])
    stored = objective.data_for_case(test_case)
    case_outputs = test_case.objective_function_outputs([objective])

    assert outputs == [1.0]
    assert type(outputs[0]) is float
    assert type(stored) is float
    assert case_outputs == {"float_objective": 1.0}
