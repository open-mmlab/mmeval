# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import ConnectivityError


def test_connectivity_error():
    pred_alpha = np.zeros((32, 32), dtype=np.uint8)
    gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
    trimap = np.zeros((32, 32), dtype=np.uint8)
    trimap[:16, :16] = 128
    trimap[16:, 16:] = 255

    connectivity_error = ConnectivityError()
    metric_results = connectivity_error(pred_alpha, gt_alpha, trimap)
    assert isinstance(metric_results, dict)
    np.testing.assert_almost_equal(metric_results['connectivity_error'], 0.008)
