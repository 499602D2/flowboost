"""Tests for Dimension type coercion logic."""

import numpy as np
import pytest

from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.search_space import Dimension, _default_digits_for_bounds


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

    def test_bool_from_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            Dimension._coerce("maybe", "bool")

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


class TestDefaultDigitsForBounds:
    """Tests for the magnitude-aware default `digits` helper."""

    @pytest.mark.parametrize(
        "lower,upper,expected",
        [
            (500.0, 2000.0, 8),   # engineering range
            (1.0, 100.0, 9),      # unit-scale
            (0.1, 1.0, 11),       # sub-unit
            (1e-9, 1e-7, 18),     # small-magnitude — must not round to zero
            (-100.0, 100.0, 9),   # symmetric around zero
            (0.0, 0.0, 12),       # degenerate — falls back to default
        ],
    )
    def test_default_digits_scales_with_magnitude(self, lower, upper, expected):
        assert _default_digits_for_bounds(lower, upper) == expected

    def test_small_magnitude_bounds_are_not_zeroed(self):
        """A value well inside [1e-9, 1e-7] must survive rounding intact."""
        digits = _default_digits_for_bounds(1e-9, 1e-7)
        value = 1.23456789e-8
        assert round(value, digits) == pytest.approx(value, rel=1e-6)


class TestDimensionRangeDefaultDigits:
    """Range dimensions pick a sensible `digits` default for float dtypes."""

    def test_float_range_gets_magnitude_aware_default(self):
        dim = Dimension.range("x", _dummy_link(), 500.0, 2000.0)
        assert dim.digits == 8

    def test_explicit_digits_overrides_default(self):
        dim = Dimension.range("x", _dummy_link(), 500.0, 2000.0, digits=4)
        assert dim.digits == 4

    def test_negative_digits_disables_rounding(self):
        dim = Dimension.range("x", _dummy_link(), 500.0, 2000.0, digits=-1)
        assert dim.digits is None

    def test_int_range_leaves_digits_unset(self):
        dim = Dimension.range("x", _dummy_link(), 0, 10, dtype=int)
        assert dim.digits is None

    def test_default_digits_collapse_float_noise(self):
        """The practical purpose of the default: near-duplicate floats that BO
        produces when converging on a box boundary must round to the same
        value, so Ax's hash-based arm dedup can catch them."""
        dim = Dimension.range("heatSource", _dummy_link(), 500.0, 2000.0)
        noisy = [500.0, 500.0000000000001, 500.00000000000034]
        rounded = {round(v, dim.digits) for v in noisy}
        assert rounded == {500.0}
