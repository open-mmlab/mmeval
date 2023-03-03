# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import os
import pytest

from mmeval.metrics import NaturalImageQualityEvaluator


def test_niqe():
    img_path = os.path.join(os.path.dirname(__file__), '../data/baboon.png')
    img = cv2.imread(img_path)

    predictions = [img]

    niqe = NaturalImageQualityEvaluator(
        crop_border=0,
        input_order='HWC',
        convert_to='gray',
        channel_order='bgr')
    result = niqe(predictions)
    np.testing.assert_almost_equal(result['niqe'], 5.73155, decimal=5)

    niqe = NaturalImageQualityEvaluator(
        crop_border=0, input_order='CHW', convert_to='y', channel_order='bgr')
    result = niqe([img.transpose(2, 0, 1)])
    np.testing.assert_almost_equal(result['niqe'], 5.72996, decimal=5)

    niqe = NaturalImageQualityEvaluator(
        crop_border=0,
        input_order='CHW',
        convert_to='gray',
        channel_order='bgr')
    result = niqe([img.transpose(2, 0, 1)])
    np.testing.assert_almost_equal(result['niqe'], 5.73155, decimal=5)

    niqe = NaturalImageQualityEvaluator(
        crop_border=6, input_order='HWC', convert_to='y', channel_order='bgr')
    result = niqe(predictions)
    np.testing.assert_almost_equal(result['niqe'], 6.10088, decimal=5)

    with pytest.raises(ValueError):
        niqe = NaturalImageQualityEvaluator(convert_to='a')

    niqe = NaturalImageQualityEvaluator(
        crop_border=0,
        input_order='HWC',
        convert_to='gray',
        channel_order='bgr')
    niqe.add(predictions)
    niqe.add(predictions)
    result = niqe.compute()
    np.testing.assert_almost_equal(result['niqe'], 5.73155, decimal=5)
