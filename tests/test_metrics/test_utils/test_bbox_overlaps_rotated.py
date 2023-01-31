# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmeval.metrics.utils.bbox_overlaps_rotated import (
    calculate_bboxes_area_rotated, calculate_overlaps_rotated, le90_to_oc,
    qbox_to_rbox)


def test_qbox_to_rbox():
    # test invalid input
    qboxes = np.array([[13, 11, 13, 31, 23, 31, 23],
                       [90, 100, 110, 100, 110, 140, 90]])
    with pytest.raises(AssertionError):
        qbox_to_rbox(qboxes)
    qboxes = np.array([13, 11, 13, 31, 23, 31, 23, 11])
    with pytest.raises(AssertionError):
        qbox_to_rbox(qboxes)
    # test output
    qboxes = np.array([[13, 11, 13, 31, 23, 31, 23, 11],
                       [90, 100, 110, 100, 110, 140, 90, 140],
                       [140, 140, 150, 140, 150, 160, 140, 160],
                       [240, 250, 250, 250, 250, 260, 240, 260]])
    rboxes = np.array([[18, 21, 20, 10, np.pi / 2],
                       [100, 120, 40, 20, np.pi / 2],
                       [145, 150, 20, 10, np.pi / 2],
                       [245, 255, 10, 10, np.pi / 2]])
    result = qbox_to_rbox(qboxes)
    assert result.shape == (4, 5)
    assert_array_almost_equal(result, rboxes)


def test_le90_to_oc():
    # test invalid input
    rboxes = np.array([[13, 11, 13, 31, 23, 31, 23],
                       [90, 100, 110, 100, 110, 140, 90]])
    with pytest.raises(AssertionError):
        le90_to_oc(rboxes)
    rboxes = np.array([18, 21, 20, 10, np.pi / 2])
    with pytest.raises(AssertionError):
        le90_to_oc(rboxes)
    # test output
    rboxes_oc = np.array([[18, 21, 20, 10, np.pi / 2],
                          [100, 120, 40, 20, np.pi / 2],
                          [145, 150, 20, 10, np.pi / 2],
                          [245, 255, 10, 10, np.pi / 2]])
    rboxes_le90 = np.array([[18, 21, 20, 10, -np.pi / 2],
                            [100, 120, 40, 20, -np.pi / 2],
                            [145, 150, 20, 10, -np.pi / 2],
                            [245, 255, 10, 10, -np.pi / 2]])
    result = le90_to_oc(rboxes_le90)
    assert result.shape == (4, 5)
    assert_array_almost_equal(result, rboxes_oc)


def test_calculate_bboxes_area_rotated():
    # test input shape (n,5)
    rboxes = np.array([[18, 21, 20, 10, np.pi / 2],
                       [100, 120, 40, 20, np.pi / 2],
                       [145, 150, 20, 10, np.pi / 2],
                       [245, 255, 10, 10, np.pi / 2]])
    areas = np.array([200, 800, 200, 100])
    results = calculate_bboxes_area_rotated(rboxes)
    assert results.shape == (4, )
    assert_array_almost_equal(results, areas)
    # test input shape (5,)
    rbox = np.array([18, 21, 20, 10, np.pi / 2])
    area = np.array([
        200,
    ])
    result = calculate_bboxes_area_rotated(rbox)
    assert_array_almost_equal(result, area)


def test_calculate_overlaps_rotated():
    # test invalid input
    rboxes1 = np.array([[18, 21, 20, 10], [100, 120, 40, 20],
                        [145, 150, 20, 10], [245, 255, 10, 10]])
    rboxes2 = np.array([[18, 21, 20, 10], [100, 120, 40, 20],
                        [145, 150, 20, 10], [245, 255, 10, 10]])
    with pytest.raises(AssertionError):
        result = calculate_overlaps_rotated(rboxes1, rboxes2)
    rboxes1 = np.array([18, 21, 20, 10])
    rboxes2 = np.array([100, 120, 40, 20])
    with pytest.raises(AssertionError):
        result = calculate_overlaps_rotated(rboxes1, rboxes2)
    # test output
    rboxes1 = np.array([[18, 21, 20, 10, -np.pi / 2],
                        [100, 120, 40, 20, -np.pi / 2],
                        [145, 150, 20, 10, -np.pi / 2],
                        [245, 255, 10, 10, -np.pi / 2]])
    rboxes2 = np.array([[18, 21, 20, 10, -np.pi / 2],
                        [100, 120, 40, 20, -np.pi / 2],
                        [145, 150, 20, 10, -np.pi / 2],
                        [245, 255, 10, 10, -np.pi / 2]])
    overlaps = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
    result = calculate_overlaps_rotated(rboxes1, rboxes2)
    assert result.shape == (4, 4)
    assert_array_almost_equal(result, overlaps)
