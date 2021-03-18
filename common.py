import re
from datetime import datetime


def datetime_from_string(string: str) -> datetime:
    """
    Get a string (e.g. a S3-key) containing a \d{8}T\d{6} pattern,
    return a datetime object. Raise a ValueError if any other number
    than one pattern is found.
    """

    matching_datetimes = re.findall(r"\d{8}T\d{6}", string)
    nfindings = len(matching_datetimes)
    if nfindings != 1:
        raise ValueError(
            f"1 expected, but found {nfindings} datetime strings within this string:\n{string}"
        )

    return datetime.strptime(matching_datetimes[0], "%Y%m%dT%H%M%S")
