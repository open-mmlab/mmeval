# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import MattingMSE


def test_matting_mse():
    pred_alpha = np.zeros((32, 32), dtype=np.uint8)
    gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
    trimap = np.zeros((32, 32), dtype=np.uint8)
    trimap[:16, :16] = 128
    trimap[16:, 16:] = 255

    mattingmse = MattingMSE()
    mattingmse_results = mattingmse(pred_alpha, gt_alpha, trimap)
    assert isinstance(mattingmse_results, dict)
    np.testing.assert_almost_equal(mattingmse_results['MattingMSE'], 1.0)
