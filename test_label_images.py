import pytest

from label_images import assign_image_to_set, get_text_y, get_weekday


def test_assign_image_to_set():

    with pytest.raises(ValueError):
        assign_image_to_set(5, 0.3, 0.2, 0.1)

    reply = assign_image_to_set(20)
    assert reply[0] == "Train"
    assert reply[1] == "Valid"
    assert reply[13] == "Test"


def test_get_text_y():

    gen = get_text_y()
    assert next(gen) == 40
    assert next(gen) == 60


def test_get_weekday():

    assert get_weekday("station_id_41_20210305T091000.jpg") == "Friday"
