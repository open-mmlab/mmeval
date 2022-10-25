# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
import pytest
from numpy.testing import assert_array_almost_equal

from mmeval.metrics import AVAMeanAP


@pytest.mark.parametrize(
    argnames=['result', 'result_custom'],
    argvalues=[
        ([{
            'video_id':
            '3reY9zJKhqN',
            'timestamp':
            1774,
            'outputs': [
                np.array([[0.362, 0.156, 0.969, 0.666, 0.106],
                          [0.442, 0.083, 0.721, 0.947, 0.162]]),
                np.array([[0.288, 0.365, 0.766, 0.551, 0.706],
                          [0.178, 0.296, 0.707, 0.995, 0.223]]),
                np.array([[0.417, 0.167, 0.843, 0.939, 0.015],
                          [0.35, 0.421, 0.57, 0.689, 0.427]])
            ]
        }, {
            'video_id':
            'HmR8SmNIoxu',
            'timestamp':
            1384,
            'outputs': [
                np.array([[0.256, 0.338, 0.726, 0.799, 0.563],
                          [0.071, 0.256, 0.64, 0.75, 0.297]]),
                np.array([[0.326, 0.036, 0.513, 0.991, 0.405],
                          [0.351, 0.035, 0.729, 0.936, 0.945]]),
                np.array([[0.051, 0.005, 0.975, 0.942, 0.424],
                          [0.347, 0.05, 0.97, 0.944, 0.396]])
            ]
        }, {
            'video_id':
            '5HNXoce1raG',
            'timestamp':
            1097,
            'outputs': [
                np.array([[0.39, 0.087, 0.833, 0.616, 0.447],
                          [0.461, 0.212, 0.627, 0.527, 0.036]]),
                np.array([[0.022, 0.394, 0.93, 0.527, 0.109],
                          [0.208, 0.462, 0.874, 0.948, 0.954]]),
                np.array([[0.206, 0.456, 0.564, 0.725, 0.685],
                          [0.106, 0.445, 0.782, 0.673, 0.367]])
            ]
        }], [{
            'video_id':
            '3reY9zJKhqN',
            'timestamp':
            1774,
            'outputs': [
                np.array([[0.362, 0.156, 0.969, 0.666, 0.106],
                          [0.442, 0.083, 0.721, 0.947, 0.162]]),
                np.array([[0.417, 0.167, 0.843, 0.939, 0.015],
                          [0.35, 0.421, 0.57, 0.689, 0.427]])
            ]
        }, {
            'video_id':
            'HmR8SmNIoxu',
            'timestamp':
            1384,
            'outputs': [
                np.array([[0.256, 0.338, 0.726, 0.799, 0.563],
                          [0.071, 0.256, 0.64, 0.75, 0.297]]),
                np.array([[0.051, 0.005, 0.975, 0.942, 0.424],
                          [0.347, 0.05, 0.97, 0.944, 0.396]])
            ]
        }, {
            'video_id':
            '5HNXoce1raG',
            'timestamp':
            1097,
            'outputs': [
                np.array([[0.39, 0.087, 0.833, 0.616, 0.447],
                          [0.461, 0.212, 0.627, 0.527, 0.036]]),
                np.array([[0.206, 0.456, 0.564, 0.725, 0.685],
                          [0.106, 0.445, 0.782, 0.673, 0.367]])
            ]
        }]),
    ])
def test_ava_map(result, result_custom):
    data_prefix = osp.normpath(osp.join(osp.dirname(__file__), 'data'))
    ann_file = osp.join(data_prefix, 'ava_detection_gt.csv')
    label_file = osp.join(data_prefix, 'ava_action_list.txt')

    ava_metric = AVAMeanAP(ann_file, label_file=label_file, num_classes=4)
    res = ava_metric(result)
    assert_array_almost_equal(res['mAP@0.5IOU'], 0.027777778)

    # custom classes
    ava_metric = AVAMeanAP(
        ann_file, label_file=label_file, num_classes=3, custom_classes=[1, 3])
    res = ava_metric(result_custom)
    assert_array_almost_equal(res['mAP@0.5IOU'], 0.04166667)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
