from datetime import timedelta
from typing import Optional

PERIODS = [
    ('year',   3600*24*365),
    ('month',  3600*24*30),
    ('day',    3600*24),
    ('hour',   3600),
    ('minute', 60),
    ('second', 1)
]


def td_format(td: timedelta, precision: Optional[int] = None) -> str:
    """
    Generate a human-readable, comma-separated string from a timedelta.

    Args:
        td (timedelta): Timedelta
        precision (Optional[int], optional): Number of units to include. \
            Defaults to None.

    Returns:
        str: A human-readable string
    """
    seconds = int(td.total_seconds())
    if seconds == 0:
        return "0 seconds"
    elif seconds < 0:
        return "In the past"

    strings = []
    for period_name, period_seconds in PERIODS:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            strings.append("%s %s%s" % (period_value, period_name, has_s))

            if precision and len(strings) >= precision:
                break

    return ", ".join(strings)
