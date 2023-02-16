# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import MeanAbsoluteError


def test_mae():
    # test image input
    preds = [np.ones((32, 32, 3))]
    gts = [np.ones((32, 32, 3)) * 2]
    mask = np.ones((32, 32, 3)) * 2
    mask[:16] *= 0

    mae = MeanAbsoluteError()
    mae_results = mae(preds, gts)
    assert isinstance(mae_results, dict)
    np.testing.assert_almost_equal(mae_results['mae'], 0.003921568627)

    mae = MeanAbsoluteError()
    mae_results = mae(preds, gts, [mask])
    assert isinstance(mae_results, dict)
    np.testing.assert_almost_equal(mae_results['mae'], 0.003921568627)

    # test video input
    preds = [np.ones((5, 32, 32, 3))]
    gts = [np.ones((5, 32, 32, 3)) * 2]
    mask = np.ones((5, 32, 32, 3)) * 2
    mask[:, :16] *= 0

    mae = MeanAbsoluteError()
    mae_results = mae(preds, gts)
    assert isinstance(mae_results, dict)
    np.testing.assert_almost_equal(mae_results['mae'], 0.003921568627)

    mae = MeanAbsoluteError()
    mae_results = mae(preds, gts, [mask])
    assert isinstance(mae_results, dict)
    np.testing.assert_almost_equal(mae_results['mae'], 0.003921568627)
