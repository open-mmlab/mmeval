# Copyright (c) OpenMMLab. All rights reserved.
from numpy.testing import assert_array_almost_equal

from mmeval.metrics import ActivityNetAR

prediction1 = [{
    'video_name':
    'v_--1DO2V4K74',
    'annotations': [{
        'segment': [30.02, 180],
        'label': 1
    }],
    'proposal_list': [{
        'segment': [53, 68],
        'score': 0.21
    }, {
        'segment': [38, 110],
        'score': 0.54
    }, {
        'segment': [65, 128],
        'score': 0.95
    }, {
        'segment': [69, 93],
        'score': 0.98
    }, {
        'segment': [99, 147],
        'score': 0.84
    }, {
        'segment': [28, 96],
        'score': 0.84
    }, {
        'segment': [18, 92],
        'score': 0.22
    }, {
        'segment': [40, 66],
        'score': 0.36
    }, {
        'segment': [14, 29],
        'score': 0.75
    }, {
        'segment': [67, 105],
        'score': 0.25
    }, {
        'segment': [2, 7],
        'score': 0.94
    }, {
        'segment': [25, 112],
        'score': 0.49
    }, {
        'segment': [7, 83],
        'score': 0.9
    }, {
        'segment': [75, 159],
        'score': 0.42
    }, {
        'segment': [99, 176],
        'score': 0.62
    }, {
        'segment': [89, 186],
        'score': 0.56
    }, {
        'segment': [50, 200],
        'score': 0.5
    }]
}]

prediction2 = [{
    'video_name':
    'v_--6bJUbfpnQ',
    'annotations': [{
        'segment': [2.57, 24.91],
        'label': 2
    }],
    'proposal_list': [{
        'segment': [53, 68],
        'score': 0.21
    }, {
        'segment': [38, 110],
        'score': 0.54
    }, {
        'segment': [65, 128],
        'score': 0.95
    }, {
        'segment': [69, 93],
        'score': 0.98
    }, {
        'segment': [99, 147],
        'score': 0.84
    }, {
        'segment': [28, 96],
        'score': 0.84
    }, {
        'segment': [4, 26],
        'score': 0.42
    }, {
        'segment': [40, 66],
        'score': 0.36
    }, {
        'segment': [14, 29],
        'score': 0.75
    }, {
        'segment': [67, 105],
        'score': 0.25
    }, {
        'segment': [2, 7],
        'score': 0.94
    }, {
        'segment': [25, 112],
        'score': 0.49
    }, {
        'segment': [7, 83],
        'score': 0.9
    }, {
        'segment': [75, 159],
        'score': 0.42
    }, {
        'segment': [99, 176],
        'score': 0.62
    }, {
        'segment': [89, 186],
        'score': 0.56
    }, {
        'segment': [50, 200],
        'score': 0.5
    }]
}, {
    'video_name':
    'v_--cdisBubfr',
    'annotations': [{
        'segment': [38.2, 110.35],
        'label': 3
    }],
    'proposal_list': [{
        'segment': [53, 68],
        'score': 0.21
    }, {
        'segment': [38, 110],
        'score': 0.54
    }, {
        'segment': [65, 128],
        'score': 0.95
    }, {
        'segment': [69, 93],
        'score': 0.98
    }, {
        'segment': [99, 147],
        'score': 0.84
    }, {
        'segment': [28, 96],
        'score': 0.84
    }, {
        'segment': [4, 26],
        'score': 0.42
    }, {
        'segment': [40, 66],
        'score': 0.36
    }, {
        'segment': [14, 29],
        'score': 0.75
    }, {
        'segment': [67, 105],
        'score': 0.25
    }, {
        'segment': [2, 7],
        'score': 0.94
    }, {
        'segment': [25, 112],
        'score': 0.49
    }, {
        'segment': [7, 83],
        'score': 0.9
    }, {
        'segment': [75, 159],
        'score': 0.42
    }, {
        'segment': [99, 176],
        'score': 0.62
    }, {
        'segment': [89, 186],
        'score': 0.56
    }, {
        'segment': [50, 200],
        'score': 0.5
    }]
}]


def test_activitynet_ar():
    # stateless
    anet_metric = ActivityNetAR()
    output = anet_metric(prediction1)
    assert 'auc' in output
    assert_array_almost_equal(output['auc'], 54.2)
    for k in 1, 5, 10, 100:
        assert f'AR@{k}' in output
        assert 0 <= output[f'AR@{k}'] <= 1

    # statefull
    anet_metric = ActivityNetAR()
    anet_metric.add(prediction1)
    anet_metric.add(prediction2)
    output = anet_metric.compute()
    assert 'auc' in output
    assert_array_almost_equal(output['auc'], 72.5)
    for k in 1, 5, 10, 100:
        assert f'AR@{k}' in output
        assert 0 <= output[f'AR@{k}'] <= 1
