import pytest

from mmeval import OneMinusNormEditDistance


def test_init():
    with pytest.raises(AssertionError):
        OneMinusNormEditDistance(letter_case='fake')


def test_one_minus_norm_edit_distance_metric():
    metric = OneMinusNormEditDistance(letter_case='lower')
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['1-N.E.D'] - 0.7) < 1e-7
    metric = OneMinusNormEditDistance(letter_case='upper')
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['1-N.E.D'] - 0.7) < 1e-7
    metric = OneMinusNormEditDistance()
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['1-N.E.D'] - 0.6) < 1e-7


test_one_minus_norm_edit_distance_metric()
