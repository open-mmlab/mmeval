import pytest

from mmeval import WordAccuracy


def test_init():
    with pytest.raises(AssertionError):
        WordAccuracy(mode=1)
    with pytest.raises(AssertionError):
        WordAccuracy(mode=[1, 2])
    with pytest.raises(AssertionError):
        WordAccuracy(mode='micro')
    metric = WordAccuracy(mode=['ignore_case', 'ignore_case', 'exact'])
    assert metric.mode == {'ignore_case', 'ignore_case', 'exact'}


def test_word_accuracy():
    metric = WordAccuracy(mode=['exact', 'ignore_case', 'ignore_case_symbol'])
    res = metric(['hello', 'hello', 'hello'], ['hello', 'HELLO', '$HELLO$'])
    assert abs(res['accuracy'] - 1. / 3) < 1e-7
    assert abs(res['ignore_case_accuracy'] - 2. / 3) < 1e-7
    assert abs(res['ignore_case_symbol_accuracy'] - 1.0) < 1e-7
    metric.reset()
    for pred, label in zip(['hello', 'hello', 'hello'],
                           ['hello', 'HELLO', '$HELLO$']):
        metric.add([pred], [label])
    res = metric.compute()
    assert abs(res['accuracy'] - 1. / 3) < 1e-7
    assert abs(res['ignore_case_accuracy'] - 2. / 3) < 1e-7
    assert abs(res['ignore_case_symbol_accuracy'] - 1.0) < 1e-7
