# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import MSE


def test_mse():
    preds = [np.ones((32, 32, 3))]
    gts = [np.ones((32, 32, 3)) * 2]
    mask = np.ones((32, 32, 3)) * 2
    mask[:16] *= 0

    mse = MSE()
    mse_results = mse(preds, gts)
    assert isinstance(mse_results, dict)
    np.testing.assert_almost_equal(mse_results['mse'], 0.000015378700496)

    mse = MSE()
    mse_results = mse(preds, gts)
    assert isinstance(mse_results, dict)
    np.testing.assert_almost_equal(mse_results['mse'], 0.000015378700496)
