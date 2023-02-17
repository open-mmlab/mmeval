import pytest

from mmeval import CharRecallPrecision


def test_init():
    with pytest.raises(AssertionError):
        CharRecallPrecision(letter_case='fake')


@pytest.mark.parametrize(
    argnames=['letter_case', 'recall', 'precision'],
    argvalues=[
        ('lower', 0.7, 1),
        ('upper', 0.7, 1),
        ('unchanged', 0.6, 6.0 / 7),
    ])
def test_char_recall_precision_metric(letter_case, recall, precision):
    metric = CharRecallPrecision(letter_case=letter_case)
    res = metric(['helL', 'HEL'], ['hello', 'HELLO'])
    assert abs(res['recall'] - recall) < 1e-7
    assert abs(res['precision'] - precision) < 1e-7
    metric.reset()
    for pred, label in zip(['helL', 'HEL'], ['hello', 'HELLO']):
        metric.add([pred], [label])
    res = metric.compute()
    assert abs(res['recall'] - recall) < 1e-7
    assert abs(res['precision'] - precision) < 1e-7
