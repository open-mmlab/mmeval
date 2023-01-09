# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import MattingSAD


def test_matting_mse():
    pred_alpha = np.zeros((32, 32), dtype=np.uint8)
    gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255

    mattingsad = MattingSAD()
    mattingsad_results = mattingsad(pred_alpha, gt_alpha)
    assert isinstance(mattingsad_results, dict)
    np.testing.assert_almost_equal(mattingsad_results['MattingSAD'], 0.032)
