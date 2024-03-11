import logging
import math
from timeit import timeit

import pytest

from flowboost.openfoam.types import FOAMType

test_cases = [
    ("123", 123),
    ("-123", -123),
    ("123.456", 123.456),
    ("-123.456", -123.456),
    ("1.23e10", 1.23e10),
    ("-1.23E-10", -1.23e-10),
    ("+123", 123),
    (" 123 ", 123),
    (".5", 0.5),
    ("+.5", 0.5),
    ("-0.5", -0.5),
    ("abc", None),
    ("", None),
    ("   ", None),
    ("12..3", None),
    ("1e1.5", None),
    ("1.2.3", None),
    (" 123 ", 123),
    ("1.23e+10", 1.23e+10),
    ("9" * 309, int("9"*309)),
    ("0.0000001", 0.0000001),
    ("0.01", 0.01),
    ("1000.0", 1000.0),
    ("++123", None),
    ("--123", None),
    (".", None),
    ("+.", None),
    ("\u0033\u0034\u0035", 345),
    ("2e308", float("2e308")),
    ("-2e308", float("-2e308")),
    ("1 2 3", None),
    ("0x1A", None),
    ("0o123", None),
    ("0b101", None),

    ("9223372036854775807", 9223372036854775807),  # Max int
    ("-9223372036854775808", -9223372036854775808),  # Min int
    ("1.7976931348623157e+308", 1.7976931348623157e+308),  # Max float
    ("-1.7976931348623157e+308", -1.7976931348623157e+308),  # Min float

    # Scientific notation variations
    ("2E2", 200.0),  # Uppercase E
    ("2e-2", 0.02),  # Negative exponent
    ("2.e10", 2.e10),  # Decimal point without fractional part
    ("-.5e-1", -0.05),  # Negative number, negative exponent

    # Leading and trailing zeros
    ("000123", 123),  # Leading zeros
    ("123.4500", 123.45),  # Trailing zeros after decimal

    # Invalid numeric formats
    ("123.4.5", None),  # Multiple decimal points
    ("1e2e3", None),  # Multiple 'e's
    ("e9", None),  # Starting with 'e'
    ("12 34", None),  # Space within number

    # Special values
    ("NaN", float("NaN")),  # Not a Number
    ("inf", float("inf")),  # Infinity
    ("-inf", float("-inf")),  # Negative Infinity

    # Locale-specific formats (assuming not supported)
    ("1,234.56", None),  # Comma as thousand separator
    ("1.234,56", None),  # Comma as decimal separator in some locales

    # More malformed inputs
    ("123abc", None),  # Alphanumeric
    ("--2", None),  # Double minus
    ("++2", None),  # Double plus
    ("", None),  # Empty string
    ("   ", None),  # Whitespace only
    (".e1", None),  # Dot followed by 'e'

    # Edge cases
    ("0", 0),  # Zero
    ("-0", 0),  # Negative zero
    (".0", 0.0),  # Decimal zero
    ("-0.0", -0.0),  # Negative decimal zero
    ("0e0", 0.0),  # Zero in scientific notation
    ("0x0", None),  # Hexadecimal (assuming not supported)
    ("0o0", None),  # Octal (assuming not supported)
    ("0b0", None),  # Binary (assuming not supported)
]


@pytest.mark.parametrize("input_str,expected_output", test_cases)
def test_try_parse_scalar(input_str, expected_output):
    result = FOAMType.try_parse_scalar(input_str)

    if result and input_str.lower() == "nan":
        assert math.isnan(result), \
            f"Failed on '{input_str}': expected {
                expected_output}, got {result}"
    else:
        assert result == expected_output, \
            f"Failed on '{input_str}': expected {
                expected_output}, got {result}"


def test_timing():
    test_strings = ["123", "-123", "123.456", "-123.456", "1.23e10", "-1.23E-10",
                    "abc", "123.456.789"] * 1000
    n = 100
    total_operations = len(test_strings) * n

    # Time the efficient_try_parse_scalar function
    times = timeit(lambda: [FOAMType.try_parse_scalar(s)
                   for s in test_strings], number=n)
    ns_per_op = (times / total_operations) * 1e9

    # Print out the timing results
    logging.info(f"FOAMType.try_parse_scalar: {ns_per_op:.1f} ns/op.")
