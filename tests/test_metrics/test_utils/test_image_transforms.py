# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmeval.metrics.utils.image_transforms import (_convert_input_type_range,
                                                   _convert_output_type_range,
                                                   bgr2ycbcr, reorder_and_crop,
                                                   reorder_image, rgb2ycbcr)


def assert_image_almost_equal(x, y, atol=1):
    assert x.dtype == np.uint8
    assert y.dtype == np.uint8
    assert np.all(np.abs(x.astype(np.int32) - y.astype(np.int32)) <= atol)


def test_convert_input_type_range():
    with pytest.raises(TypeError):
        # The img type should be np.float32 or np.uint8
        in_img = np.random.rand(10, 10, 3).astype(np.uint64)
        _convert_input_type_range(in_img)
    # np.float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = _convert_input_type_range(in_img)
    assert out_img.dtype == np.float32
    assert np.absolute(out_img).mean() < 1
    # np.uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = _convert_input_type_range(in_img)
    assert out_img.dtype == np.float32
    assert np.absolute(out_img).mean() < 1


def test_convert_output_type_range():
    with pytest.raises(TypeError):
        # The dst_type should be np.float32 or np.uint8
        in_img = np.random.rand(10, 10, 3).astype(np.float32)
        _convert_output_type_range(in_img, np.uint64)
    # np.float32
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.float32)
    out_img = _convert_output_type_range(in_img, np.float32)
    assert out_img.dtype == np.float32
    assert np.absolute(out_img).mean() < 1
    # np.uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.float32)
    out_img = _convert_output_type_range(in_img, np.uint8)
    assert out_img.dtype == np.uint8
    assert np.absolute(out_img).mean() > 1


def test_rgb2ycbcr():
    with pytest.raises(TypeError):
        # The img type should be np.float32 or np.uint8
        in_img = np.random.rand(10, 10, 3).astype(np.uint64)
        rgb2ycbcr(in_img)

    # float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = rgb2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            computed_ycbcr[i, j, :] = [y, cb, cr]
    computed_ycbcr /= 255.
    assert_array_almost_equal(out_img, computed_ycbcr, decimal=2)
    # y_only=True
    out_img = rgb2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            computed_y[i, j] = y
    computed_y /= 255.
    assert_array_almost_equal(out_img, computed_y, decimal=2)

    # uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = rgb2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            y, cb, cr = y.round(), cb.round(), cr.round()
            computed_ycbcr[i, j, :] = [y, cb, cr]
    assert_image_almost_equal(out_img, computed_ycbcr)
    # y_only=True
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = rgb2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            r, g, b = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            y = y.round()
            computed_y[i, j] = y
    assert_image_almost_equal(out_img, computed_y)


def test_bgr2ycbcr():
    # float32
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = bgr2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            computed_ycbcr[i, j, :] = [y, cb, cr]
    computed_ycbcr /= 255.
    assert_array_almost_equal(out_img, computed_ycbcr, decimal=2)
    # y_only=True
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    out_img = bgr2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            computed_y[i, j] = y
    computed_y /= 255.
    assert_array_almost_equal(out_img, computed_y, decimal=2)

    # uint8
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = bgr2ycbcr(in_img)
    computed_ycbcr = np.empty_like(in_img)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            cb = 128 - r * 37.797 - g * 74.203 + b * 112.0
            cr = 128 + r * 112.0 - g * 93.786 - b * 18.214
            y, cb, cr = y.round(), cb.round(), cr.round()
            computed_ycbcr[i, j, :] = [y, cb, cr]
    assert_image_almost_equal(out_img, computed_ycbcr)
    # y_only = True
    in_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    out_img = bgr2ycbcr(in_img, y_only=True)
    computed_y = np.empty_like(out_img, dtype=out_img.dtype)
    in_img = in_img / 255.
    for i in range(in_img.shape[0]):
        for j in range(in_img.shape[1]):
            b, g, r = in_img[i, j]
            y = 16 + r * 65.481 + g * 128.553 + b * 24.966
            y = y.round()
            computed_y[i, j] = y
    assert_image_almost_equal(out_img, computed_y)


def test_reorder_image():
    img_hw = np.ones((32, 32))
    img_hwc = np.ones((32, 32, 3))
    img_chw = np.ones((3, 32, 32))

    with pytest.raises(ValueError):
        reorder_image(img_hw, 'HH')

    output = reorder_image(img_hw)
    assert output.shape == (32, 32, 1)

    output = reorder_image(img_hwc)
    assert output.shape == (32, 32, 3)

    output = reorder_image(img_chw, input_order='CHW')
    assert output.shape == (32, 32, 3)


def test_reorder_and_crop():
    img = np.random.randint(0, 255, size=(4, 4, 3))
    new_img = reorder_and_crop(img, 0, 'HWC', None, 'rgb')
    assert new_img.shape == (4, 4, 3)
    assert new_img.dtype == np.float64

    img = np.random.randint(0, 255, size=(4, 4, 3))
    with pytest.raises(ValueError):
        reorder_and_crop(img, 0, 'HWC', convert_to='z')

    with pytest.raises(ValueError):
        reorder_and_crop(img, 0, 'HWC', convert_to='Y', channel_order='abc')

    new_img = reorder_and_crop(img, 0, 'HWC', convert_to='Y',
                               channel_order='rgb')
    assert new_img.shape == (4, 4, 1)
    assert new_img.dtype == np.float64

    new_img = reorder_and_crop(img, 0, 'HWC', convert_to='Y',
                               channel_order='bgr')
    assert new_img.shape == (4, 4, 1)
    assert new_img.dtype == np.float64

    new_img = reorder_and_crop(img, 0, 'HWC', convert_to='Y',
                               channel_order='rgb')
    assert new_img.shape == (4, 4, 1)
    assert new_img.dtype == np.float64

    new_img = reorder_and_crop(np.random.randint(0, 255, size=(32, 32, 3)), 4,
                               'HWC', None, 'rgb')
    assert new_img.shape == (24, 24, 3)

    new_img = reorder_and_crop(np.random.randint(0, 255, size=(32, 32, 3)), 4,
                               'HWC', 'Y', 'rgb')
    assert new_img.shape == (24, 24, 1)
