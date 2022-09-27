# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pickle
import pytest

from mmeval.core.base_metric import BaseMetric
from mmeval.detection.voc_map import VOCMeanAP


def _gen_bboxes(num_bboxes, img_w, img_h):
    x = np.random.rand(num_bboxes, ) * img_w
    y = np.random.rand(num_bboxes, ) * img_h
    w = np.random.rand(num_bboxes, ) * (img_w - x)
    h = np.random.rand(num_bboxes, ) * (img_h - y)
    return np.stack([x, y, x + w, y + h], axis=1)


def _gen_prediction(num_pred=10, num_classes=10, img_w=256, img_h=256):
    prediction = {
        'bboxes': _gen_bboxes(num_pred, img_w, img_h),
        'scores': np.random.rand(num_pred, ),
        'labels': np.random.randint(0, num_classes, size=(num_pred, ))
    }
    return prediction


def _gen_groundtruth(num_gt=10,
                     num_ignored_gt=2,
                     num_classes=10,
                     img_w=256,
                     img_h=256):
    groundtruth = {
        'bboxes': _gen_bboxes(num_gt, img_w, img_h),
        'labels': np.random.randint(0, num_classes, size=(num_gt, )),
        'bboxes_ignore': _gen_bboxes(num_ignored_gt, img_w, img_h),
        'labels_ignore':
        np.random.randint(0, num_classes, size=(num_ignored_gt, ))
    }
    return groundtruth


@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {
            'iou_thrs': [0.5, 0.75]
        },
        {
            'scale_ranges': [(20, 40)]
        },
        {
            'iou_thrs': [0.5, 0.75],
            'scale_ranges': [(20, 40)]
        },
        {
            'eval_mode': '11points'
        },
        {
            'eval_mode': 'area'
        },
        {
            'nproc': -1
        },
    ])
def test_metric_interface(metric_kwargs):
    num_classes = 10
    voc_map = VOCMeanAP(num_classes=num_classes, **metric_kwargs)
    assert isinstance(voc_map, BaseMetric)

    metric_results = voc_map(
        predictions=[
            _gen_prediction(num_classes=num_classes) for _ in range(10)
        ],
        groundtruths=[
            _gen_groundtruth(num_classes=num_classes) for _ in range(10)
        ],
    )
    assert isinstance(metric_results, dict)
    assert 'mAP' in metric_results


def test_metric_accurate():
    results_pkl_fpath = 'tests/test_detection/data/retinanet_r50_fpn_1x_voc0712.500.pkl'  # noqa: E501
    with open(results_pkl_fpath, 'rb') as f:
        results = pickle.load(f)

    voc_map = VOCMeanAP(
        num_classes=20, eval_mode='area', use_legacy_coordinate=True)

    for pred, ann in results:
        voc_map.add([
            pred,
        ], [
            ann,
        ])
    metric_results = voc_map.compute()

    np.testing.assert_almost_equal(metric_results['mAP'], 0.8224697)


if __name__ == '__main__':
    pytest.main([__file__, '-vv', '--capture=no'])
