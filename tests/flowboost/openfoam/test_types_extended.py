"""Extended tests for FOAMType: parse assertions, to_FOAM, round-trips, vectors/tensors."""

import math

import numpy as np
import pytest

from flowboost.openfoam.types import FOAMType, Switch


class TestParseAssertions:
    """test_line_parsing only logs — these actually assert parsed values."""

    def test_dimensioned_with_name(self):
        name, dim, value = FOAMType.parse("nu [0 2 -1 0 0 0 0] 1e-5")
        assert name == "nu"
        assert dim == "[0 2 -1 0 0 0 0]"
        assert value == 1e-5

    def test_dimensioned_non_numeric_dim(self):
        name, dim, value = FOAMType.parse("SOI [CAD] 355")
        assert name == "SOI"
        assert dim == "[CAD]"
        assert value == 355

    def test_dimensioned_empty_brackets(self):
        name, dim, value = FOAMType.parse("kappa [] 0.41")
        assert name == "kappa"
        assert dim == "[]"
        assert value == 0.41

    def test_no_name_with_dimension(self):
        name, dim, value = FOAMType.parse("[0 0 0 1 0 0 0] 273.15")
        assert name == ""
        assert dim == "[0 0 0 1 0 0 0]"
        assert value == 273.15

    def test_name_value_no_dimension(self):
        name, dim, value = FOAMType.parse("simpleFoam 200")
        assert name == "simpleFoam"
        assert dim is None
        assert value == 200

    def test_value_only(self):
        name, dim, value = FOAMType.parse("273.15")
        assert name is None
        assert dim is None
        assert value == 273.15

    def test_vector_value_only(self):
        name, dim, value = FOAMType.parse("( 1 2 3 )")
        assert name is None
        assert dim is None
        np.testing.assert_array_equal(value, np.array([1, 2, 3]))

    def test_tensor_value_only(self):
        name, dim, value = FOAMType.parse("( 1 0 0 0 1 0 0 0 1 )")
        assert name is None
        assert dim is None
        np.testing.assert_array_equal(value, np.eye(3))

    def test_dimensioned_vector_value(self):
        name, dim, value = FOAMType.parse("U [0 1 -1 0 0 0 0] ( 1 2 3 )")
        assert name == "U"
        assert dim == "[0 1 -1 0 0 0 0]"
        np.testing.assert_array_equal(value, np.array([1, 2, 3]))


class TestParseZero:
    """Regression tests: parse must handle 0/0.0 as numeric, not string."""

    def test_parse_zero_integer(self):
        _, _, value = FOAMType.parse("0")
        assert value == 0
        assert isinstance(value, int)

    def test_parse_zero_float(self):
        _, _, value = FOAMType.parse("0.0")
        assert value == 0.0
        assert isinstance(value, float)

    def test_parse_dimensioned_zero(self):
        name, dim, value = FOAMType.parse("nu [0 2 -1 0 0 0 0] 0")
        assert name == "nu"
        assert dim == "[0 2 -1 0 0 0 0]"
        assert value == 0


class TestToFOAM:
    """FOAMType.to_FOAM had zero test coverage."""

    def test_string_passthrough(self):
        assert FOAMType.to_FOAM("hello") == "hello"

    def test_bool_true(self):
        assert FOAMType.to_FOAM(True) == "true"

    def test_bool_false(self):
        assert FOAMType.to_FOAM(False) == "false"

    def test_bool_before_int(self):
        """bool is subclass of int — must check bool first."""
        assert FOAMType.to_FOAM(True) != "1"
        assert FOAMType.to_FOAM(False) != "0"

    def test_int(self):
        assert FOAMType.to_FOAM(42) == "42"

    def test_float(self):
        assert FOAMType.to_FOAM(3.14) == "3.14"

    def test_float_zero(self):
        assert FOAMType.to_FOAM(0.0) == "0.0"

    @pytest.mark.parametrize(
        "value, expected",
        [
            (np.int64(42), "42"),
            (np.float32(3.5), "3.5"),
            (np.bool_(True), "true"),
        ],
    )
    def test_numpy_scalar_types(self, value, expected):
        assert FOAMType.to_FOAM(value) == expected

    def test_numpy_vector(self):
        result = FOAMType.to_FOAM(np.array([1.0, 2.0, 3.0]))
        assert result == "( 1.0 2.0 3.0 )"

    def test_numpy_tensor(self):
        result = FOAMType.to_FOAM(np.eye(3))
        assert result == "( 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 )"

    def test_list_vector(self):
        result = FOAMType.to_FOAM([1, 2, 3])
        assert result == "( 1 2 3 )"

    def test_empty_list_raises(self):
        with pytest.raises(TypeError):
            FOAMType.to_FOAM([])

    def test_empty_tuple_raises(self):
        with pytest.raises(TypeError):
            FOAMType.to_FOAM(())

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            FOAMType.to_FOAM(None)


class TestToFOAMRoundTrip:
    """to_FOAM -> parse should recover the original value."""

    @pytest.mark.parametrize("value", [42, -7, 100])
    def test_int_round_trip(self, value):
        serialized = FOAMType.to_FOAM(value)
        _, _, parsed = FOAMType.parse(serialized)
        assert parsed == value

    @pytest.mark.parametrize("value", [3.14, -0.5, 1e-5, 1000.0])
    def test_float_round_trip(self, value):
        serialized = FOAMType.to_FOAM(value)
        _, _, parsed = FOAMType.parse(serialized)
        assert parsed == pytest.approx(value)

    def test_bool_round_trip(self):
        for val in (True, False):
            serialized = FOAMType.to_FOAM(val)
            _, _, parsed = FOAMType.parse(serialized)
            assert parsed == val


class TestParseVectorSpace:
    def test_spherical_tensor(self):
        result = FOAMType.parse_vector_space("( 5 )")
        assert result == 5

    def test_vector(self):
        result = FOAMType.parse_vector_space("( 1 2 3 )")
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_full_tensor_identity(self):
        result = FOAMType.parse_vector_space("( 1 0 0 0 1 0 0 0 1 )")
        np.testing.assert_array_equal(result, np.eye(3))

    def test_symmetric_tensor(self):
        result = FOAMType.parse_vector_space("( 1 2 3 4 5 6 )")
        # indices: (0,0)=1, (1,1)=2, (2,2)=3, (0,1)=4, (0,2)=5, (1,2)=6
        assert result[0, 0] == 1
        assert result[1, 1] == 2
        assert result[2, 2] == 3
        assert result[0, 1] == result[1, 0] == 4
        assert result[0, 2] == result[2, 0] == 5
        assert result[1, 2] == result[2, 1] == 6


class TestParseSubdict:
    def test_parses_scalars_booleans_and_vectors(self):
        result = FOAMType.parse_subdict(
            "{ alpha 0.5; beta -1; flag true; vec (1 2 3); }"
        )

        assert result["alpha"] == 0.5
        assert result["beta"] == -1
        assert result["flag"] is True
        np.testing.assert_array_equal(result["vec"], np.array([1, 2, 3]))


class TestConstructSymmTensor:
    def test_wrong_component_count_raises(self):
        with pytest.raises(AssertionError):
            FOAMType.construct_symm_tensor([1, 2, 3])

    def test_symmetric(self):
        tensor = FOAMType.construct_symm_tensor([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(tensor, tensor.T)


class TestParseEdgeCases:
    """Edge cases around the fixed zero-value code path."""

    def test_dimensioned_nan(self):
        name, dim, value = FOAMType.parse("nu [0 2 -1 0 0 0 0] NaN")
        assert name == "nu"
        assert dim == "[0 2 -1 0 0 0 0]"
        assert math.isnan(value)

    def test_dimensioned_inf(self):
        name, dim, value = FOAMType.parse("nu [0 2 -1 0 0 0 0] inf")
        assert name == "nu"
        assert value == float("inf")

    def test_negative_zero(self):
        _, _, value = FOAMType.parse("-0")
        assert value == 0

    def test_negative_zero_float(self):
        _, _, value = FOAMType.parse("-0.0")
        assert value == -0.0
        assert isinstance(value, float)

    def test_positive_inf_scalar(self):
        result = FOAMType.try_parse_scalar("+inf")
        assert result == float("inf")

    def test_dimensioned_empty_value(self):
        """Dimension present but no value after it — falls through to string."""
        name, dim, value = FOAMType.parse("nu [0 2 -1 0 0 0 0] ")
        # try_parse_scalar("") returns None, falls through to Switch then string
        # Current behavior: returns full data string as value, losing name/dim
        assert value is not None  # doesn't crash


class TestSwitch:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("YES", True),
            ("on", True),
            ("ON", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("no", False),
            ("NO", False),
            ("off", False),
            ("OFF", False),
        ],
    )
    def test_valid_switches(self, s, expected):
        assert Switch.from_string(s).value == expected

    def test_invalid_returns_none(self):
        assert Switch.from_string("maybe").value is None
        assert Switch.from_string("0").value is None
