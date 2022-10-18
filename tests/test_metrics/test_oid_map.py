# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest
import random

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics.oid_map import OIDMeanAP, get_relation_matrix


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


mini_file = 'tests/test_metrics/data/bbox_labels_600_hierarchy_mini.json'
RELATION_MATRIX = get_relation_matrix(
    hierarchy_file=mini_file,
    label_file='tests/test_metrics/data/class-descriptions-boxable.csv')


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


def test_metric_invalid_usage():
    with pytest.raises(AssertionError):
        OIDMeanAP(eval_mode='xxx')

    oid_map = OIDMeanAP()

    with pytest.raises(RuntimeError):
        oid_map.num_classes

    with pytest.raises(RuntimeError):
        oid_map.class_relation_matrix

    num_classes = 10
    oid_map = OIDMeanAP(
        num_classes=num_classes, class_relation_matrix=RELATION_MATRIX)

    with pytest.raises(KeyError):
        prediction = _gen_prediction(num_classes=num_classes)
        groundtruth = _gen_groundtruth(num_classes=num_classes)
        del prediction['bboxes']
        oid_map([prediction], [groundtruth])

    with pytest.raises(AssertionError):
        prediction = _gen_prediction(num_classes=num_classes)
        groundtruth = _gen_groundtruth(num_classes=num_classes)
        oid_map(prediction, groundtruth)


@pytest.mark.parametrize(
    argnames=('predictions', 'groundtruths', 'num_classes', 'target_mAP'),
    argvalues=[
        (
            [{
                'bboxes': np.array([
                    [ 27,  38, 173,  67],  # noqa: E201
                    [ 46, 185, 227, 248],  # noqa: E201
                    [153, 147, 250, 164],  # noqa: E201
                    [  0,   0,   0,   0],  # noqa: E201
                ]),
                'scores': np.array([0.17, 0.20, 0.48, 0.50]),
                'labels': np.array([5, 6, 7, 8])
            }],
            [{
                'instances': [
                    {
                        'bbox': [ 27,  38, 173,  67],  # noqa: E201
                        'bbox_label': 5,
                        'is_group_of': False,
                    },
                    {
                        'bbox': [120, 185, 227, 248],
                        'bbox_label': 6,
                        'is_group_of': False,
                    },
                    {
                        'bbox': [153, 100, 250, 164],
                        'bbox_label': 8,
                        'is_group_of': True,
                    },
                    {
                        'bbox': [  1,   1,   2,   2],  # noqa: E201
                        'bbox_label': 8,
                        'is_group_of': True,
                    },
                ],
                'image_level_labels': np.empty((0,)),
            }],
            10,
            0.6666666
        )
    ])
def test_metric_accurate(predictions, groundtruths, num_classes, target_mAP):
    relation_matrix = np.eye(num_classes, num_classes)
    oid_map = OIDMeanAP(
        num_classes=num_classes, class_relation_matrix=relation_matrix)
    metric_results = oid_map(predictions, groundtruths)
    np.testing.assert_almost_equal(metric_results['mAP'], target_mAP)


if __name__ == '__main__':
    pytest.main([__file__, '-vv', '--capture=no'])
