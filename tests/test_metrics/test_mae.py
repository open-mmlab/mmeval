# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import MAE


def test_mae():
    preds = [np.ones((32, 32, 3))]
    gts = [np.ones((32, 32, 3)) * 2]
    mask = np.ones((32, 32, 3)) * 2
    mask[:16] *= 0

    mae = MAE()
    mae_results = mae(preds, gts)
    assert isinstance(mae_results, dict)
    np.testing.assert_almost_equal(mae_results['mae'], 0.003921568627)

    mae = MAE()
    mae_results = mae(preds, gts, [mask])
    assert isinstance(mae_results, dict)
    np.testing.assert_almost_equal(mae_results['mae'], 0.003921568627)
