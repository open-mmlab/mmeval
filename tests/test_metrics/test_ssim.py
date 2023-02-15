# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval import StructuralSimilarity as SSIM


def test_ssim_init():
    with pytest.raises(AssertionError):
        SSIM(crop_border=0, input_order='HH')

    with pytest.raises(AssertionError):
        SSIM(crop_border=0, input_order='HWC', convert_to='Z')

    with pytest.raises(AssertionError):
        SSIM(crop_border=0, input_order='HWC', convert_to='y',
             channel_order='abc')


@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'img1', 'img2', 'results'],
    argvalues=[
        ({'crop_border': 0, 'input_order': 'HWC'}, [np.ones((32, 32))],
         [np.ones((32, 32)) * 2], 0.9130623),
        ({'input_order': 'HWC'}, [np.ones((32, 32, 3))],
         [np.ones((32, 32, 3)) * 2], 0.9130623),
        ({'input_order': 'CHW'}, [np.ones((3, 32, 32))],
         [np.ones((3, 32, 32)) * 2], 0.9130623),
        ({'crop_border': 2, 'input_order': 'HWC'}, [np.ones((32, 32, 3))],
         [np.ones((32, 32, 3)) * 2], 0.9130623),
        ({'crop_border': 3, 'input_order': 'HWC'}, [np.ones((32, 32, 3))],
         [np.ones((32, 32, 3)) * 2], 0.9130623),
        ({'crop_border': 4}, [np.ones((3, 32, 32))],
         [np.ones((3, 32, 32)) * 2], 0.9130623),
        ({'convert_to': 'Y', 'input_order': 'HWC'}, [np.ones((32, 32, 3))],
         [np.ones((32, 32, 3)) * 2], 0.9987801),
        ({}, [np.ones((5, 3, 32, 32))], [np.ones((5, 3, 32, 32)) * 2],
         0.9130623)
    ]
)
def test_ssim(metric_kwargs, img1, img2, results):
    ssim = SSIM(**metric_kwargs)
    ssim_results = ssim(img1, img2)
    assert isinstance(ssim_results, dict)
    np.testing.assert_almost_equal(ssim_results['ssim'], results)
