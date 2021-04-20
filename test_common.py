from datetime import datetime

import pytest

from common import datetime_from_string


def test_datetime_from_string():

    TEST_KEY = r"raw/red/REN/station_id_41/version=1/year=2020/month=04/day=03/station_id_41_20200403T101328.jpg"
    assert datetime_from_string(TEST_KEY) == datetime(2020, 4, 3, 10, 13, 28)

    with pytest.raises(ValueError):
        datetime_from_string(TEST_KEY[:-5])
