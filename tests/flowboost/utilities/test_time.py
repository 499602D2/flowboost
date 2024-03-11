from datetime import timedelta

from flowboost.utilities.time import td_format


def test_seconds():
    assert td_format(timedelta(seconds=1)) == "1 second"
    assert td_format(timedelta(seconds=30)) == "30 seconds"


def test_minutes_seconds():
    assert td_format(timedelta(minutes=1, seconds=30)
                     ) == "1 minute, 30 seconds"


def test_hours_minutes_seconds():
    assert td_format(timedelta(hours=1, minutes=1, seconds=1)
                     ) == "1 hour, 1 minute, 1 second"


def test_days():
    assert td_format(timedelta(days=1)) == "1 day"
    assert td_format(timedelta(days=2)) == "2 days"


def test_months_days():
    # Assuming 30 days per month as per your implementation
    assert td_format(timedelta(days=30)) == "1 month"
    assert td_format(timedelta(days=61)) == "2 months, 1 day"


def test_years_months_days():
    # Assuming 365 days per year, 30 days per month
    assert td_format(timedelta(days=365)) == "1 year"
    assert td_format(timedelta(days=395)) == "1 year, 1 month"


def test_negative_duration():
    assert td_format(timedelta(seconds=-1)) == "In the past"


def test_zero_seconds():
    assert td_format(timedelta(seconds=0)) == "In the past"


def test_precision_single_unit():
    assert td_format(timedelta(hours=1, minutes=30,
                     seconds=45), precision=1) == "1 hour"


def test_precision_two_units():
    assert td_format(timedelta(hours=1, minutes=30, seconds=45),
                     precision=2) == "1 hour, 30 minutes"


def test_precision_three_units():
    assert td_format(timedelta(days=1, hours=2, minutes=30),
                     precision=3) == "1 day, 2 hours, 30 minutes"


def test_precision_with_days():
    assert td_format(timedelta(days=365, hours=23), precision=1) == "1 year"
    assert td_format(timedelta(days=30, hours=23, minutes=59),
                     precision=2) == "1 month, 23 hours"


def test_precision_exceeding_units():
    # Test case where precision is higher than the available non-zero units
    assert td_format(timedelta(hours=1, minutes=30),
                     precision=5) == "1 hour, 30 minutes"


def test_negative_duration_with_precision():
    # Negative duration should ignore precision and return "In the past"
    assert td_format(timedelta(seconds=-5), precision=2) == "In the past"


def test_zero_seconds_with_precision():
    # Zero seconds should ignore precision and return "In the past"
    assert td_format(timedelta(seconds=0), precision=3) == "In the past"
