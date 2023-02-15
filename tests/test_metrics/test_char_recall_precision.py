import pytest

from mmeval import CharRecallPrecision


def test_init():
    with pytest.raises(AssertionError):
        CharRecallPrecision(letter_case='fake')


def test_char_recall_precision_metric():
    metric = CharRecallPrecision(letter_case='lower')
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['recall'] - 0.7) < 1e-7
    assert abs(res['precision'] - 1) < 1e-7

    metric = CharRecallPrecision(letter_case='upper')
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['recall'] - 0.7) < 1e-7
    assert abs(res['precision'] - 1) < 1e-7

    metric = CharRecallPrecision(letter_case='unchanged')
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['recall'] - 0.6) < 1e-7
    assert abs(res['precision'] - 6.0 / 7) < 1e-7
