
import polars as pl
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from flowboost.openfoam.case import Case
from flowboost.optimizer.objectives import Objective


def max_temp_objective(case: Case) -> int:
    # Returns a value of 1955 for the default data
    df: pl.DataFrame = case.data.simple_function_object_reader("averagePT")
    return int(df.select(pl.max("volAverage(T)")).item())


def test_objective_initialization():
    """Test initialization without changes, as it doesn't involve function execution."""
    objective = Objective(
        name="Test Objective",
        minimize=True,
        objective_function=max_temp_objective,
        normalization_step="min-max"
    )
    assert objective.name == "Test Objective"
    assert objective.minimize is True
    assert callable(objective.objective_function)
    assert isinstance(objective._post_processing_steps[0][0], MinMaxScaler)


def test_evaluate_method(test_case):
    objective = Objective(
        name="Test Evaluate",
        minimize=True,
        objective_function=max_temp_objective
    )
    result = objective.evaluate(test_case)
    assert result == 1955, f"Result {result} != 1955"


def test_normalization_step_addition():
    objective = Objective(
        name="Test Normalization",
        minimize=True,
        objective_function=max_temp_objective,
        normalization_step="box-cox"
    )
    assert isinstance(objective._post_processing_steps[0][0], PowerTransformer)
    assert objective._post_processing_steps[0][0].method == "box-cox"


def test_data_retrieval_post_processing(test_case):
    """ Test if data retrieval and post-processing work as expected """
    objective = Objective(
        name="Data Retrieval Test",
        minimize=True,
        objective_function=max_temp_objective
    )
    objective.evaluate(test_case)  # No need to patch; controlled by mock_case
    out = objective.data_for_case(test_case, post_processed=False)
    assert out == 1955, f"Out == {out} != 1955"

    objective.attach_post_processing_step(
        step=lambda x, **kwargs: [val + 1 for val in x])
    objective.execute_post_processing_steps()
    post_out = objective.data_for_case(test_case, post_processed=True)

    assert post_out == 1955+1, f"Post-proc out = {post_out} != 1955+1"
