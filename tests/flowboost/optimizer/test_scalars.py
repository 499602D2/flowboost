import numpy as np
import pytest

from flowboost.optimizer.scalars import coerce_objective_scalar


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (np.float64(2.5), 2.5),
        (np.array([3.5]), 3.5),
        ([np.array([4.5])], 4.5),
        (([5.5],), 5.5),
    ],
)
def test_coerce_objective_scalar_accepts_scalar_like_values(value, expected):
    result = coerce_objective_scalar(value)
    assert result == pytest.approx(expected)
    assert type(result) is float


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [1.0, 2.0],
        np.array([1.0, 2.0]),
        "not-a-number",
    ],
)
def test_coerce_objective_scalar_rejects_non_scalar_values(value):
    with pytest.raises(ValueError, match="scalar-like|numeric"):
        coerce_objective_scalar(value)
