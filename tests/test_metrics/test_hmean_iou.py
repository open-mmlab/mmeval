# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pytest

from mmeval.metrics import HmeanIoU
from mmeval.metrics.hmean_iou import compute_hmean


def test_compute_hmean():
    with pytest.raises(AssertionError):
        compute_hmean(0, 0, 0.0, 0)
    with pytest.raises(AssertionError):
        compute_hmean(0, 0, 0, 0.0)
    with pytest.raises(AssertionError):
        compute_hmean([1], 0, 0, 0)
    with pytest.raises(AssertionError):
        compute_hmean(0, [1], 0, 0)

    _, _, hmean = compute_hmean(2, 2, 2, 2)
    assert hmean == 1

    _, _, hmean = compute_hmean(0, 0, 2, 2)
    assert hmean == 0


def test_hmean_iou_metric():
    # We denote the polygons as the following.
    # gt_polys: gt_a, gt_b, gt_c, gt_d_ignored
    # pred_polys: pred_a, pred_b, pred_c, pred_d

    # There are two pairs of matches: (gt_a, pred_a) and (gt_b, pred_b),
    # because the IoU > threshold.

    # gt_c and pred_c do not match any of the polygons.

    # pred_d is ignored in the recall computation since it overlaps
    # gt_d_ignored and the precision > ignore_precision_thr.
    gt_polygons = [[
        [0, 0, 1, 0, 1, 1, 0, 1],
        [2, 0, 3, 0, 3, 1, 2, 1],
        [10, 0, 11, 0, 11, 1, 10, 1],
        [1, 0, 2, 0, 2, 1, 1, 1],
    ], [
        np.array([0, 0, 1, 0, 1, 1, 0, 1]),
    ]]
    pred_polygons = [[
        [0, 0, 1, 0, 1, 1, 0, 1],
        [2, 0.1, 3, 0.1, 3, 1.1, 2, 1.1],
        [1, 1, 2, 1, 2, 2, 1, 2],
        [1, -0.5, 2, -0.5, 2, 0.5, 1, 0.5],
    ], [
        np.array([0, 0, 1, 0, 1, 1, 0, 1]),
        np.array([0, 0, 1, 0, 1, 1, 0, 1])
    ]]
    pred_scores = [[1, 1, 1, 0.001], np.array([1, 0.95])]
    gt_ignore_flags = [[False, False, False, True], [False]]

    hmeaniou = HmeanIoU()
    results = hmeaniou(pred_polygons, pred_scores, gt_polygons,
                       gt_ignore_flags)
    p = 3 / 5
    r = 3 / 4
    hmean = 2 * p * r / (p + r)
    assert results['best']['precision'] == pytest.approx(p)
    assert results['best']['recall'] == pytest.approx(r)
    assert results['best']['hmean'] == pytest.approx(hmean)


def test_hmean_iou_strategy():
    hmeaniou = HmeanIoU()
    iou_metric = np.array([[1, 1], [1, 0]])
    assert hmeaniou._max_matching(iou_metric) == 2
    assert hmeaniou._vanilla_matching(iou_metric) == 1
