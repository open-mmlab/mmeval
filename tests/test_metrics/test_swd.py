# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.metrics import SlicedWassersteinDistance as SWD


@pytest.mark.parametrize(
    argnames=['init_kwargs', 'preds', 'gts', 'results'],
    argvalues=[
        ({'resolution': 32},
         [np.ones((3, 32, 32)) * i for i in range(100)],
         [np.ones((3, 32, 32)) * 2 * i for i in range(100)],
         [198.67430960025712, 33.72058904027052, 116.19744932026381])]
)
def test_swd(init_kwargs, preds, gts, results):
    swd = SWD(**init_kwargs)
    swd_results = swd(preds, gts)
    for out, res in zip(swd_results.values(), results):
        np.testing.assert_almost_equal(out / 100, res / 100, decimal=1)
    swd.reset()
    assert swd._results == []

    swd.add(preds[:50], gts[:50])
    swd.add(preds[50:], gts[50:])
    swd_results = swd.compute()
    for out, res in zip(swd_results.values(), results):
        np.testing.assert_almost_equal(out / 100, res / 100, decimal=1)
