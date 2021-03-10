from datetime import datetime

import pytest

from download_images import date_from_key, valid_by_datetime


def test_date_from_key():

    TEST_KEY = r"raw/red/REN/station_id_41/version=1/year=2020/month=04/day=03/station_id_41_20200403T101328.jpg"
    assert date_from_key(TEST_KEY) == datetime(2020, 4, 3, 10, 13, 28)

    with pytest.raises(ValueError):
        date_from_key(TEST_KEY[:-5])


def test_valid_by_datetime():

    TEST_KEY = r"raw/red/REN/station_id_41/version=1/year=2020/month=04/day=03/station_id_41_20200403T101328.jpg"

    assert valid_by_datetime(TEST_KEY, datetime(2020, 1, 1), datetime(2020, 12, 31))
    assert valid_by_datetime(TEST_KEY, None, datetime(2020, 12, 31))
    assert valid_by_datetime(TEST_KEY, datetime(2020, 1, 1), None)
    assert not valid_by_datetime(TEST_KEY, datetime(2021, 1, 1), datetime(2021, 12, 31))
