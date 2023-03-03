# Copyright (c) OpenMMLab. All rights reserved.
from numpy.testing import assert_array_almost_equal

from mmeval.metrics import ActivityNetAR

prediction1 = [
    [[53, 68, 0.21], [38, 110, 0.54], [65, 128, 0.95], [69, 93, 0.98],
     [99, 147, 0.84], [28, 96, 0.84], [18, 92, 0.22], [40, 66, 0.36],
     [14, 29, 0.75], [67, 105, 0.25], [2, 7, 0.94], [25, 112, 0.49],
     [7, 83, 0.9], [75, 159, 0.42], [99, 176, 0.62], [89, 186, 0.56],
     [50, 200, 0.5]],
]

annotation1 = [[
    [30.02, 180],
]]

prediction2 = [[[53, 68, 0.21], [38, 110, 0.54], [65, 128,
                                                  0.95], [69, 93, 0.98],
                [99, 147, 0.84], [28, 96, 0.84], [4, 26, 0.42], [40, 66, 0.36],
                [14, 29, 0.75], [67, 105, 0.25], [2, 7, 0.94], [25, 112, 0.49],
                [7, 83, 0.9], [75, 159, 0.42], [99, 176, 0.62],
                [89, 186, 0.56], [50, 200, 0.5]],
               [[45, 88, 0.31], [34, 83, 0.54], [76, 113,
                                                 0.95], [34, 85, 0.88],
                [99, 147, 0.84], [28, 96, 0.84], [4, 26, 0.42], [40, 66, 0.36],
                [14, 29, 0.75], [67, 105, 0.25], [2, 7, 0.94], [25, 112, 0.49],
                [7, 83, 0.9], [75, 159, 0.42], [99, 176, 0.62],
                [89, 186, 0.56], [50, 200, 0.5]]]

annotation2 = [[
    [38.2, 110.35],
], [[38.2, 110.35], [40, 66]]]


def test_activitynet_ar():
    # stateless
    anet_metric = ActivityNetAR()
    output = anet_metric(prediction1, annotation1)
    assert 'auc' in output
    assert_array_almost_equal(output['auc'], 54.2)
    for k in 1, 5, 10, 100:
        assert f'AR@{k}' in output
        assert 0 <= output[f'AR@{k}'] <= 1

    # statefull
    anet_metric = ActivityNetAR()
    anet_metric.add(prediction1, annotation1)
    anet_metric.add(prediction2, annotation2)
    output = anet_metric.compute()
    assert 'auc' in output
    assert_array_almost_equal(output['auc'], 74.9625)
    for k in 1, 5, 10, 100:
        assert f'AR@{k}' in output
        assert 0 <= output[f'AR@{k}'] <= 1
