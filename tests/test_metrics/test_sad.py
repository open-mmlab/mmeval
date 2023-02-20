# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval import SumAbsoluteDifferences as SAD


def test_sad():
    prediction = np.zeros((32, 32), dtype=np.uint8)
    groundtruth = np.ones((32, 32), dtype=np.uint8) * 255

    sad = SAD()
    metric_results = sad(prediction, groundtruth)
    assert isinstance(metric_results, dict)
    np.testing.assert_almost_equal(metric_results['sad'], 0.032)
