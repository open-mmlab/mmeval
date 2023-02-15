# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmeval.metrics import PeakSignalNoiseRatio


def test_psnr_init():
    with pytest.raises(AssertionError):
        PeakSignalNoiseRatio(crop_border=0, input_order='HH')

    with pytest.raises(AssertionError):
        PeakSignalNoiseRatio(crop_border=0, convert_to='ABC')

    with pytest.raises(AssertionError):
        PeakSignalNoiseRatio(
            crop_border=0, convert_to='y', channel_order='qwe')


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
               ({}, [np.zeros((32, 32))], [np.ones((32, 32)) * 255], 0),
               ({}, [np.ones((5, 3, 32, 32))], [np.ones(
                   (5, 3, 32, 32)) * 2], 48.1308036086791)])
def test_psnr(metric_kwargs, img1, img2, results):
    psnr = PeakSignalNoiseRatio(**metric_kwargs)
    psnr_results = psnr(img1, img2)
    assert isinstance(psnr_results, dict)
    np.testing.assert_almost_equal(psnr_results['psnr'], results)
