# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmeval.metrics import InstanceSeg

seg_valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39)
class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                'garbagebin')
dataset_meta = dict(
    seg_valid_class_ids=seg_valid_class_ids, classes=class_labels)


def _demo_mm_model_output():
    """Create a superset of inputs needed to run test or train batches."""
    n_points_list = [3300, 3000]
    gt_labels_list = [[0, 0, 0, 0, 0, 0, 14, 14, 2, 2, 2],
                      [13, 13, 2, 1, 3, 3, 0, 0, 0, 0]]
    predictions = []
    groundtruths = []

    for idx, points_num in enumerate(n_points_list):
        points = np.ones(points_num) * -1
        gt = np.ones(points_num)
        info = {}
        for ii, i in enumerate(gt_labels_list[idx]):
            i = seg_valid_class_ids[i]
            points[ii * 300:(ii + 1) * 300] = ii
            gt[ii * 300:(ii + 1) * 300] = i * 1000 + ii
            info[f'{idx}_{ii}'] = {
                'mask': (points == ii),
                'label_id': i,
                'conf': 0.99
            }
        predictions.append(info)
        groundtruths.append(gt)

    return predictions, groundtruths


def _demo_mm_model_wrong_output():
    """Create a superset of inputs needed to run test or train batches."""
    n_points_list = [3300, 3000]
    gt_labels_list = [[0, 0, 0, 0, 0, 0, 14, 14, 2, 2, 2],
                      [13, 13, 2, 1, 3, 3, 0, 0, 0, 0]]
    predictions = []
    groundtruths = []

    for idx, points_num in enumerate(n_points_list):
        points = np.ones(points_num) * -1
        gt = np.ones(points_num)
        info = {}
        for ii, i in enumerate(gt_labels_list[idx]):
            i = seg_valid_class_ids[i]
            points[ii * 300:(ii + 1) * 300] = i
            gt[ii * 300:(ii + 1) * 300] = i * 1000 + ii
            info[f'{idx}_{ii}'] = {
                'mask': (points == i),
                'label_id': i,
                'conf': 0.99
            }
        predictions.append(info)
        groundtruths.append(gt)

    return predictions, groundtruths


def test_evaluate():
    predictions, groundtruths = _demo_mm_model_output()
    instance_seg_metric = InstanceSeg(dataset_meta=dataset_meta)
    res = instance_seg_metric(predictions, groundtruths)
    assert isinstance(res, dict)
    for label in [
            'cabinet', 'bed', 'chair', 'sofa', 'showercurtrain', 'toilet'
    ]:
        metrics = res['classes'][label]
        assert metrics['ap'] == 1.0
        assert metrics['ap50%'] == 1.0
        assert metrics['ap25%'] == 1.0
    predictions, groundtruths = _demo_mm_model_wrong_output()
    res = instance_seg_metric(predictions, groundtruths)
    assert abs(res['classes']['cabinet']['ap50%'] - 0.12) < 0.01
    assert abs(res['classes']['cabinet']['ap25%'] - 0.4125) < 0.01
    assert abs(res['classes']['bed']['ap50%'] - 1) < 0.01
    assert abs(res['classes']['bed']['ap25%'] - 1) < 0.01
    assert abs(res['classes']['chair']['ap50%'] - 0.375) < 0.01
    assert abs(res['classes']['chair']['ap25%'] - 0.785714) < 0.01
