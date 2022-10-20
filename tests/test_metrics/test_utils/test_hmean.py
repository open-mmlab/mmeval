import pytest

from mmeval.metrics.utils import compute_hmean


def test_compute_hmean():
    with pytest.raises(AssertionError):
        compute_hmean(0, 0, 0.0, 0)
    with pytest.raises(AssertionError):
        compute_hmean(0, 0, 0, 0.0)
    with pytest.raises(AssertionError):
        compute_hmean([1], 0, 0, 0)
    with pytest.raises(AssertionError):
        compute_hmean(0, [1], 0, 0)

    _, _, hmean = compute_hmean(2, 2, 2, 2)
    assert hmean == 1

    _, _, hmean = compute_hmean(0, 0, 2, 2)
    assert hmean == 0
