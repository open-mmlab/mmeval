import pytest

from mmeval import OneMinusNormEditDistance


def test_init():
    with pytest.raises(AssertionError):
        OneMinusNormEditDistance(letter_case='fake')


@pytest.mark.parametrize(
    argnames=['letter_case', 'expected'],
    argvalues=[('unchanged', 0.6), ('upper', 0.7), ('lower', 0.7)])
def test_one_minus_norm_edit_distance_metric(letter_case, expected):
    metric = OneMinusNormEditDistance(letter_case=letter_case)
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['1-N.E.D'] - expected) < 1e-7
    metric.reset()
    for pred, label in zip(['helL', 'HEL'], ['hello', 'HELLO']):
        metric.add([pred], [label])
    res = metric.compute()
    assert abs(res['1-N.E.D'] - expected) < 1e-7
