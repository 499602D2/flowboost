"""Extended tests for coerce_objective_scalar — NaN, Inf, bool, nested containers."""

import math

import numpy as np
import pytest

from flowboost.optimizer.scalars import coerce_objective_scalar


class TestNaNAndInf:
    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            coerce_objective_scalar(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            coerce_objective_scalar(float("inf"))

    def test_negative_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            coerce_objective_scalar(float("-inf"))

    def test_numpy_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            coerce_objective_scalar(np.float64("nan"))

    def test_numpy_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            coerce_objective_scalar(np.array([np.inf]))


class TestBoolCoercion:
    def test_true_becomes_float(self):
        result = coerce_objective_scalar(True)
        assert result == 1.0
        assert type(result) is float

    def test_false_becomes_float(self):
        result = coerce_objective_scalar(False)
        assert result == 0.0
        assert type(result) is float


class TestDeeplyNested:
    def test_nested_list(self):
        result = coerce_objective_scalar([[[[[42.0]]]]])
        assert result == 42.0
        assert type(result) is float

    def test_nested_ndarray(self):
        result = coerce_objective_scalar(np.array([[[[3.14]]]]))
        assert result == pytest.approx(3.14)
        assert type(result) is float

    def test_mixed_nesting(self):
        result = coerce_objective_scalar([np.array([7.0])])
        assert result == 7.0
        assert type(result) is float
