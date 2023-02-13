# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.metrics import SignalNoiseRatio


def test_snr_init():
    with pytest.raises(AssertionError):
        SignalNoiseRatio(crop_border=0, input_order='HH')

    with pytest.raises(AssertionError):
        SignalNoiseRatio(crop_border=0, convert_to='ABC')

    with pytest.raises(AssertionError):
        SignalNoiseRatio(crop_border=0, convert_to='y', channel_order='qwe')


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'img1', 'img2', 'results'],
    argvalues=[
        ({'crop_border': 0}, [np.ones((32, 32))],
         [np.ones((32, 32)) * 2], 6.020599913279624),
        ({'crop_border': 0, 'input_order': 'HWC'}, [np.ones((32, 32, 3))],
         [np.ones((32, 32, 3)) * 2], 6.020599913279624),
        ({'crop_border': 0, 'input_order': 'CHW'}, [np.ones((3, 32, 32))],
         [np.ones((3, 32, 32)) * 2], 6.020599913279624),
        ({'crop_border': 2}, [np.ones((32, 32))],
         [np.ones((32, 32)) * 2], 6.020599913279624),
        ({'crop_border': 3, 'input_order': 'HWC'}, [np.ones((32, 32, 3))],
         [np.ones((32, 32, 3)) * 2], 6.020599913279624),
        ({'crop_border': 4}, [np.ones((3, 32, 32))],
         [np.ones((3, 32, 32)) * 2], 6.020599913279624),
        ({}, [np.ones((32, 32, 3))], [np.ones((32, 32, 3)) * 2],
         6.020599913279624),
        ({'input_order': 'HWC', 'convert_to': 'Y'}, [np.ones((32, 32, 3))],
         [np.ones((32, 32, 3)) * 2], 26.290039980499536),
        ({}, [np.ones((32, 32))], [np.ones((32, 32))], float('inf')),
        ({}, [np.zeros((32, 32))], [np.ones((32, 32))], 0),
        ({}, [np.ones((5, 3, 32, 32))], [np.ones((5, 3, 32, 32)) * 2],
         6.020599913279624)
    ]
)
def test_snr(metric_kwargs, img1, img2, results):
    snr = SignalNoiseRatio(**metric_kwargs)
    snr_results = snr(img1, img2)
    assert isinstance(snr_results, dict)
    np.testing.assert_almost_equal(snr_results['snr'], results)
