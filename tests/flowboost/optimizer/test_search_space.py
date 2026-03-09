"""Tests for Dimension type coercion logic."""

import numpy as np
import pytest

from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.search_space import Dimension


# Minimal DictionaryLink for constructing Dimension instances in tests
def _dummy_link() -> DictionaryLink:
    return DictionaryLink("system/controlDict").entry("dummy")


class TestDimensionCoerce:
    """Tests for Dimension._coerce() — the static coercion method."""

    # --- int coercion ---

    def test_int_from_string(self):
        assert Dimension._coerce("3", "int") == 3
        assert type(Dimension._coerce("3", "int")) is int

    def test_int_from_float(self):
        assert Dimension._coerce(3.0, "int") == 3
        assert type(Dimension._coerce(3.0, "int")) is int

    def test_int_from_string_float(self):
        """String '3.0' should coerce to int 3."""
        assert Dimension._coerce("3.0", "int") == 3
        assert type(Dimension._coerce("3.0", "int")) is int

    def test_int_from_bool(self):
        """bool is a subclass of int; _coerce(True, 'int') must return int, not bool."""
        result = Dimension._coerce(True, "int")
        assert result == 1
        assert type(result) is int  # not bool!

    def test_int_already_int(self):
        assert Dimension._coerce(42, "int") == 42
        assert type(Dimension._coerce(42, "int")) is int

    def test_int_lossy_truncation_warns(self, caplog):
        """Truncating 3.7 → 3 should log a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            result = Dimension._coerce(3.7, "int")
        assert result == 3
        assert "Lossy" in caplog.text

    def test_int_from_numpy_int(self):
        result = Dimension._coerce(np.int64(5), "int")
        assert result == 5
        assert type(result) is int

    def test_int_from_numpy_float(self):
        result = Dimension._coerce(np.float64(5.0), "int")
        assert result == 5
        assert type(result) is int

    # --- float coercion ---

    def test_float_from_string(self):
        assert Dimension._coerce("3.14", "float") == 3.14
        assert type(Dimension._coerce("3.14", "float")) is float

    def test_float_from_int(self):
        assert Dimension._coerce(3, "float") == 3.0
        assert type(Dimension._coerce(3, "float")) is float

    def test_float_already_float(self):
        assert Dimension._coerce(3.14, "float") is not None
        assert type(Dimension._coerce(3.14, "float")) is float

    def test_float_from_numpy_float(self):
        result = Dimension._coerce(np.float64(2.718), "float")
        assert result == pytest.approx(2.718)
        assert type(result) is float

    # --- bool coercion ---

    def test_bool_from_string_true(self):
        for s in ("true", "True", "TRUE", "1", "t", "y", "yes"):
            assert Dimension._coerce(s, "bool") is True

    def test_bool_from_string_false(self):
        for s in ("false", "False", "0", "no", "n"):
            assert Dimension._coerce(s, "bool") is False

    def test_bool_already_bool(self):
        assert Dimension._coerce(True, "bool") is True
        assert Dimension._coerce(False, "bool") is False

    # --- str coercion ---

    def test_str_from_int(self):
        assert Dimension._coerce(42, "str") == "42"
        assert type(Dimension._coerce(42, "str")) is str

    def test_str_from_float(self):
        result = Dimension._coerce(3.14, "str")
        assert type(result) is str

    def test_str_already_str(self):
        assert Dimension._coerce("hello", "str") == "hello"

    # --- error cases ---

    def test_unknown_value_type_raises(self):
        with pytest.raises(ValueError, match="Unknown value_type"):
            Dimension._coerce(42, "complex")

    def test_unconvertible_value_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Dimension._coerce("not_a_number", "float")


class TestDimensionCoerceValue:
    """Tests for the instance method dim.coerce_value()."""

    def test_coerce_value_uses_dimension_value_type(self):
        dim = Dimension.range("x", _dummy_link(), 0, 10, dtype=int)
        assert dim.value_type == "int"
        result = dim.coerce_value("5")
        assert result == 5
        assert type(result) is int

    def test_coerce_value_float(self):
        dim = Dimension.range("x", _dummy_link(), 0.0, 10.0, dtype=float)
        result = dim.coerce_value("3.14")
        assert result == pytest.approx(3.14)
        assert type(result) is float

    def test_coerce_value_passthrough_when_no_value_type(self):
        dim = Dimension("x", "range")
        assert dim.value_type is None
        raw = "untouched"
        assert dim.coerce_value(raw) is raw

    def test_coerce_value_choice_str(self):
        dim = Dimension.choice("s", _dummy_link(), ["a", "b", "c"])
        assert dim.value_type == "str"
        assert dim.coerce_value(42) == "42"
