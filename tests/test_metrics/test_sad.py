# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import SAD


def test_sad():
    prediction = np.zeros((32, 32), dtype=np.uint8)
    groundtruth = np.ones((32, 32), dtype=np.uint8) * 255

    sad = SAD()
    sad_results = sad(prediction, groundtruth)
    assert isinstance(sad_results, dict)
    np.testing.assert_almost_equal(sad_results['SAD'], 0.032)
