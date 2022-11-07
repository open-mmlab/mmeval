# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest
from unittest.mock import Mock, patch

from mmeval.metrics import SNR


def test_snr_init():
    with pytest.raises(AssertionError):
        SNR(crop_border=0, input_order='HH')

    with pytest.raises(AssertionError):
        SNR(crop_border=0, convert_to='ABC')

    with pytest.raises(AssertionError):
        SNR(crop_border=0, convert_to='y', channel_order='qwe')


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
        ({}, [np.zeros((32, 32))], [np.ones((32, 32))], 0)
    ]
)
def test_snr(metric_kwargs, img1, img2, results):
    snr = SNR(**metric_kwargs)
    snr_results = snr(img1, img2)
    assert isinstance(snr_results, dict)
    np.testing.assert_almost_equal(snr_results['snr'], results)


def test_psnr_channel_order_checking(caplog):
    snr = SNR(crop_border=0, channel_order='RGB')
    img1, img2 = [np.ones((32, 32))], [np.ones((32, 32)) * 2]
    target_warn_msg = ('Input \'channel_order\'(BGR) is different '
                       'from \'self.channel_order\'(RGB).')
    with patch('mmeval.metrics.snr.reorder_and_crop',
               Mock(return_value=np.ones((32, 32)))) as process_fn:
        snr.add(img1, img2, channel_order='BGR')
    assert target_warn_msg in caplog.text
    assert process_fn.call_args.kwargs['channel_order'] == 'BGR'
