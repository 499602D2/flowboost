"""Extended tests for Dimension constructors and validation."""

import pytest

from flowboost.openfoam.dictionary import DictionaryLink
from flowboost.optimizer.search_space import Dimension


def _dummy_link() -> DictionaryLink:
    return DictionaryLink("system/controlDict").entry("dummy")


class TestDimensionNameValidation:
    def test_space_in_name_raises(self):
        with pytest.raises(ValueError, match="spaces"):
            Dimension("bad name", "range")

    def test_valid_name_ok(self):
        dim = Dimension("validName", "range")
        assert dim.name == "validName"


class TestDimensionChoice:
    def test_empty_choices_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Dimension.choice("x", _dummy_link(), [])

    def test_inferred_type_str(self):
        dim = Dimension.choice("x", _dummy_link(), ["a", "b", "c"])
        assert dim.value_type == "str"

    def test_inferred_type_int(self):
        dim = Dimension.choice("x", _dummy_link(), [1, 2, 3])
        assert dim.value_type == "int"

    def test_explicit_dtype(self):
        dim = Dimension.choice("x", _dummy_link(), [1, 2, 3], dtype=float)
        assert dim.value_type == "float"
        assert dim.values is not None
        assert all(isinstance(v, float) for v in dim.values)

    def test_explicit_bool_dtype_rejects_invalid_tokens(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            Dimension.choice("x", _dummy_link(), ["true", "maybe"], dtype=bool)


class TestDimensionFixed:
    def test_fixed_int(self):
        dim = Dimension.fixed("x", _dummy_link(), 42)
        assert dim.type == "fixed"
        assert dim.values == [42]
        assert dim.value_type == "int"

    def test_fixed_float(self):
        dim = Dimension.fixed("x", _dummy_link(), 3.14)
        assert dim.value_type == "float"

    def test_fixed_str(self):
        dim = Dimension.fixed("x", _dummy_link(), "hello")
        assert dim.value_type == "str"

    def test_fixed_bool(self):
        dim = Dimension.fixed("x", _dummy_link(), True)
        assert dim.value_type == "bool"


class TestDimensionRange:
    def test_range_basic(self):
        dim = Dimension.range("x", _dummy_link(), 0.0, 1.0)
        assert dim.type == "range"
        assert dim.bounds == [0.0, 1.0]

    def test_linked_entry(self):
        link = _dummy_link()
        dim = Dimension.range("x", link, 0.0, 1.0)
        assert dim.linked_entry is link

    def test_log_scale(self):
        dim = Dimension.range("x", _dummy_link(), 1e-5, 1e-1, log_scale=True)
        assert dim.log_scale is True


class TestDimensionUnsupportedType:
    def test_complex_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            Dimension._get_value_type_str(complex)
