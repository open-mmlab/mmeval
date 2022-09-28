# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pickle
import pytest
import random

from mmeval.core.base_metric import BaseMetric
from mmeval.detection.oid_map import OIDMeanAP, get_relation_matrix


def _gen_bboxes(num_bboxes, img_w, img_h):
    x = np.random.rand(num_bboxes, ) * img_w
    y = np.random.rand(num_bboxes, ) * img_h
    w = np.random.rand(num_bboxes, ) * (img_w - x)
    h = np.random.rand(num_bboxes, ) * (img_h - y)
    return np.stack([x, y, x + w, y + h], axis=1)


def _gen_prediction(num_pred=10, num_classes=10, img_w=256, img_h=256):
    predictions = {
        'bboxes': _gen_bboxes(num_pred, img_w, img_h),
        'scores': np.random.rand(num_pred, ),
        'labels': np.random.randint(0, num_classes, size=(num_pred, ))
    }
    return predictions


def _gen_groundtruth(num_gt=10, num_classes=10, img_w=256, img_h=256):

    instances = []
    for bbox in _gen_bboxes(num_gt, img_w, img_h):
        instances.append({
            'bbox': bbox.tolist(),
            'bbox_label': random.randint(0, num_classes - 1),
            'is_group_of': random.randint(0, 1) == 0,
        })

    groundtruth = {
        'instances': instances,
        'image_level_labels':
        np.random.randint(0, num_classes, size=(num_gt, )),
    }
    return groundtruth


RELATION_MATRIX = get_relation_matrix(
    hierarchy_file='tests/test_detection/data/bbox_labels_600_hierarchy.json',
    label_file='tests/test_detection/data/class-descriptions-boxable.csv')


@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        # multi scale and multi ranges for OIDMeanAP are not supported now.
        # {'scale_ranges': [(20, 40)]},
        {
            'nproc': -1
        },
    ])
def test_metric_interface(metric_kwargs):

    num_classes = 601
    oid_map = OIDMeanAP(
        num_classes=num_classes,
        class_relation_matrix=RELATION_MATRIX,
        **metric_kwargs)
    assert isinstance(oid_map, BaseMetric)

    metric_results = oid_map(
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
    results_pkl_fpath = 'tests/test_detection/data/openimages_detection.500.pkl'  # noqa: E501
    with open(results_pkl_fpath, 'rb') as f:
        results = pickle.load(f)

    oid_map = OIDMeanAP(num_classes=601, class_relation_matrix=RELATION_MATRIX)
    for pred, ann in results:
        oid_map.add([pred], [ann])
    metric_results = oid_map.compute()

    np.testing.assert_almost_equal(metric_results['mAP'], 0.731455445)


if __name__ == '__main__':
    pytest.main([__file__, '-vv', '--capture=no'])
