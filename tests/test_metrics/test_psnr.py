# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from unittest.mock import Mock, patch

from mmeval.metrics import PSNR


def test_psnr_init():
    with pytest.raises(AssertionError):
        PSNR(crop_border=0, input_order='HH')

    with pytest.raises(AssertionError):
        PSNR(crop_border=0, convert_to='ABC')

    with pytest.raises(AssertionError):
        PSNR(crop_border=0, convert_to='y', channel_order='qwe')


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'img1', 'img2', 'results'],
    argvalues=[({
        'crop_border': 0
    }, [np.ones((32, 32))], [np.ones((32, 32)) * 2], 48.1308036086791),
               ({
                   'crop_border': 0,
                   'input_order': 'HWC'
               }, [np.ones((32, 32, 3))], [np.ones(
                   (32, 32, 3)) * 2], 48.1308036086791),
               ({
                   'crop_border': 0,
                   'input_order': 'CHW'
               }, [np.ones((3, 32, 32))], [np.ones(
                   (3, 32, 32)) * 2], 48.1308036086791),
               ({
                   'crop_border': 2
               }, [np.ones((32, 32))], [np.ones(
                   (32, 32)) * 2], 48.1308036086791),
               ({
                   'crop_border': 3,
                   'input_order': 'HWC'
               }, [np.ones((32, 32, 3))], [np.ones(
                   (32, 32, 3)) * 2], 48.1308036086791),
               ({
                   'crop_border': 4
               }, [np.ones((3, 32, 32))], [np.ones(
                   (3, 32, 32)) * 2], 48.1308036086791),
               ({}, [np.ones((32, 32, 3))], [np.ones(
                   (32, 32, 3)) * 2], 48.1308036086791),
               ({
                   'input_order': 'HWC',
                   'convert_to': 'Y'
               }, [np.ones((32, 32, 3))], [np.ones(
                   (32, 32, 3)) * 2], 49.45272242415597),
               ({}, [np.ones((32, 32))], [np.ones((32, 32))], float('inf')),
               ({}, [np.zeros((32, 32))], [np.ones((32, 32)) * 255], 0)])
def test_psnr(metric_kwargs, img1, img2, results):
    psnr = PSNR(**metric_kwargs)
    psnr_results = psnr(img1, img2)
    assert isinstance(psnr_results, dict)
    np.testing.assert_almost_equal(psnr_results['psnr'], results)


def test_psnr_channel_order_checking(caplog):
    psnr = PSNR(crop_border=0, channel_order='RGB')
    img1, img2 = [np.ones((32, 32))], [np.ones((32, 32)) * 2]
    target_warn_msg = ('Input \'channel_order\'(BGR) is different '
                       'from \'self.channel_order\'(RGB).')
    with patch('mmeval.metrics.psnr.reorder_and_crop',
               Mock(return_value=np.ones((32, 32)))) as process_fn:
        psnr.add(img1, img2, channel_order='BGR')
    assert target_warn_msg in caplog.text
    assert process_fn.call_args.kwargs['channel_order'] == 'BGR'
