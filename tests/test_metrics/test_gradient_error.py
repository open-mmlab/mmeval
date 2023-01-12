# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import GradientError


def test_gradient_error():
    np.random.seed(0)
    pred_alpha = np.random.randn(32, 32).astype('uint8')
    gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
    trimap = np.zeros((32, 32), dtype=np.uint8)
    trimap[:16, :16] = 128
    trimap[16:, 16:] = 255

    gradienterror = GradientError()
    gradienterror_results = gradienterror(pred_alpha, gt_alpha, trimap)
    assert isinstance(gradienterror_results, dict)
    np.testing.assert_almost_equal(gradienterror_results['GradientError'],
                                   0.0935)
