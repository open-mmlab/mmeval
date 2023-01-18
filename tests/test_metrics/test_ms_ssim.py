# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.metrics import MultiScaleStructureSimilarity as MS_SSIM


def test_ms_ssim_init():
    ms_ssim = MS_SSIM()
    assert (ms_ssim.weights == np.array(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])).all()

    ms_ssim = MS_SSIM(weights=[0.22, 0.72])
    assert (ms_ssim.weights == np.array([0.22, 0.72])).all()


@pytest.mark.parametrize(
    argnames=['init_kwargs', 'inputs', 'results'],
    argvalues=[
        ({'input_order': 'CHW'}, [np.ones((3, 64, 64)) * 255.] * 4, 1),
        ({'input_order': 'HWC'}, [np.zeros((64, 64, 3)) * 255.] * 4, 1),
        ({'input_order': 'HWC', 'filter_size': 0},
         [np.zeros((64, 64, 3)) * 255.] * 4, 1),
        ({}, [np.ones((3, 64, 64)) * 255, np.zeros((3, 64, 64))],
         0.2929473249689634),
        ({'input_order': 'HWC'},
         [np.ones((64, 64, 3)) * 255, np.zeros((64, 64, 3))],
         0.2929473249689634),
        ({'input_order': 'HWC', 'filter_size': 0},
         [np.ones((64, 64, 3)) * 255, np.zeros((64, 64, 3))],
         0.29295045137405396)]
    )
def test_ms_ssim(init_kwargs, inputs, results):
    ms_ssim = MS_SSIM(**init_kwargs)
    ms_ssim_results = ms_ssim(inputs)
    np.testing.assert_allclose(
        ms_ssim_results['ms-ssim'], results)


def test_raise_error():
    ms_ssim = MS_SSIM()
    inputs = [np.random.randint(0, 255, (3, 64, 64))] * 3
    with pytest.raises(AssertionError):
        ms_ssim(inputs)

    # shape checking
    with pytest.raises(RuntimeError):
        ms_ssim.compute_ms_ssim(
            np.random.randint(0, 255, (64, 64, 3)),
            np.random.randint(0, 255, (3, 64, 64))
        )

    with pytest.raises(RuntimeError):
        ms_ssim.compute_ms_ssim(
            np.random.randint(0, 255, (64, 64, 3)),
            np.random.randint(0, 255, (64, 64, 3))
        )
