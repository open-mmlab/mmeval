# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.metrics import SlicedWassersteinDistance as SWD


@pytest.mark.parametrize(
    argnames=['init_kwargs', 'preds', 'gts', 'results'],
    argvalues=[
        ({'resolution': 32},
         [np.random.rand(3, 32, 32) for _ in range(100)],
         [np.random.rand(3, 32, 32) for _ in range(100)],
         [16.495922580361366, 24.15413036942482, 20.325026474893093])]
)
def test_swd(init_kwargs, preds, gts, results):
    swd = SWD(**init_kwargs)
    swd_results = swd(preds, gts)
    for out, res in zip(swd_results.values(), results):
        np.testing.assert_almost_equal(out / 100, res / 100, decimal=1)
    swd.reset()
    assert swd._results == []
