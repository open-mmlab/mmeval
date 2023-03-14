# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import ProposalRecall


def _create_dummy_results():
    # create fake results
    bboxes = np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                       [150, 160, 190, 200], [250, 260, 350, 360]])
    scores = np.array([1.0, 0.98, 0.96, 0.95])

    return dict(bboxes=bboxes, scores=scores)


def _create_dummy_gts():
    # create fake gts
    bboxes = np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                       [150, 160, 190, 200], [250, 260, 350, 360]])

    return dict(bboxes=bboxes)


# TODO: move necessary function to somewhere
def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
    # random generate bounding boxes in 'xyxy' formart.
    x = np.random.rand(num_bboxes, ) * img_w
    y = np.random.rand(num_bboxes, ) * img_h
    w = np.random.rand(num_bboxes, ) * (img_w - x)
    h = np.random.rand(num_bboxes, ) * (img_h - y)
    return np.stack([x, y, x + w, y + h], axis=1)


def _gen_prediction(num_pred=10, img_w=256, img_h=256):
    prediction = {
        'bboxes': _gen_bboxes(num_pred, img_w, img_h),
        'scores': np.random.rand(num_pred, ),
    }
    return prediction


def _gen_groundtruth(num_gt=10, img_w=256, img_h=256):
    groundtruth = {
        'bboxes': _gen_bboxes(num_gt, img_w, img_h),
    }
    return groundtruth


# yapf: disable
@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'iou_thrs': [0.5, 0.75]},
        {'nproc': -1},
        {'proposal_nums': [10, 30, 100]},
        {'use_legacy_coordinate': True}
    ])
# yapf: enable
def test_metric_interface(metric_kwargs):
    voc_map = ProposalRecall(**metric_kwargs)
    assert isinstance(voc_map, BaseMetric)

    metric_results = voc_map(
        predictions=[_gen_prediction() for _ in range(10)],
        groundtruths=[_gen_groundtruth() for _ in range(10)],
    )
    assert isinstance(metric_results, dict)
    assert 'AR@100' in metric_results


def test_metric_invalid_usage():
    with pytest.raises(TypeError):
        ProposalRecall(iou_thrs=1)

    with pytest.raises(TypeError):
        ProposalRecall(proposal_nums=0.5)


def test_metric_accurate():
    # create dummy data
    predictions = [_create_dummy_results()]
    groundtruths = [_create_dummy_gts()]

    # test default proposal nums
    proposal_recall = ProposalRecall()
    metric_results = proposal_recall(predictions, groundtruths)
    target = {'AR@1': 0.25, 'AR@10': 1.0, 'AR@100': 1.0, 'AR@1000': 1.0}
    assert metric_results == target

    # test manually set proposal nums
    proposal_recall = ProposalRecall(proposal_nums=(2, 4))
    metric_results = proposal_recall(predictions, groundtruths)
    target = {'AR@2': 0.5, 'AR@4': 1.0}
    assert metric_results == target


if __name__ == '__main__':
    pytest.main([__file__, '-vv', '--capture=no'])
