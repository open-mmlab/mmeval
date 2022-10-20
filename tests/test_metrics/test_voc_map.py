# Copyright (c) OpenMMLab. All rights reserved.

# yapf: disable

import numpy as np
import pytest

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import VOCMeanAP


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


# yapf: disable
@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'iou_thrs': [0.5, 0.75]},
        {'scale_ranges': [(20, 40)]},
        {'iou_thrs': [0.5, 0.75], 'scale_ranges': [(20, 40)]},
        {'eval_mode': '11points'},
        {'eval_mode': 'area'},
        {'nproc': -1},
    ])
# yapf: enable
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


def test_metric_invalid_usage():
    with pytest.raises(AssertionError):
        VOCMeanAP(eval_mode='xxx')

    with pytest.raises(AssertionError):
        VOCMeanAP(iou_thrs=1)

    with pytest.raises(RuntimeError):
        voc_map = VOCMeanAP()
        voc_map.num_classes

    num_classes = 10
    voc_map = VOCMeanAP(num_classes=num_classes)

    with pytest.raises(KeyError):
        prediction = _gen_prediction(num_classes=num_classes)
        groundtruth = _gen_groundtruth(num_classes=num_classes)
        del prediction['bboxes']
        voc_map([prediction], [groundtruth])

    with pytest.raises(AssertionError):
        prediction = _gen_prediction(num_classes=num_classes)
        groundtruth = _gen_groundtruth(num_classes=num_classes)
        voc_map(prediction, groundtruth)


@pytest.mark.parametrize(
    argnames=('predictions', 'groundtruths', 'num_classes', 'target_mAP'),
    argvalues=[(
        [{
            'bboxes':
            np.array([
                [27, 38, 173, 67],  # noqa: E201
                [46, 185, 227, 248],  # noqa: E201
                [153, 147, 250, 164],
                [0, 0, 0, 0],  # noqa: E201
            ]),
            'scores':
            np.array([0.17, 0.20, 0.48, 0.50]),
            'labels':
            np.array([5, 6, 7, 8])
        }],
        [{
            'bboxes':
            np.array([
                [27, 38, 173, 67],  # noqa: E201
                [120, 185, 227, 248],  # noqa: E201
                [153, 100, 250, 164],  # noqa: E201
                [1, 1, 2, 2],  # noqa: E201
            ]),
            'labels':
            np.array([5, 6, 8, 8]),
            'bboxes_ignore':
            np.empty((0, 4)),
            'labels_ignore':
            np.empty((0, )),
        }],
        10,
        0.6666666)])
def test_metric_accurate(predictions, groundtruths, num_classes, target_mAP):
    voc_map = VOCMeanAP(num_classes=num_classes)
    metric_results = voc_map(predictions, groundtruths)
    np.testing.assert_almost_equal(metric_results['mAP'], target_mAP)


if __name__ == '__main__':
    pytest.main([__file__, '-vv', '--capture=no'])
