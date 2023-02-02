# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmeval.metrics import JhmdbPCKAccuracy, MpiiPCKAccuracy, PCKAccuracy


# yapf: disable
@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'thr': 1.0, 'norm_item': 'bbox'},
        {'thr': 0.2, 'norm_item': 'head'},
        {'thr': 0.2, 'norm_item': 'torso'},
        {'thr': 0.2, 'norm_item': ['bbox', 'torso']},
    ]
)
# yapf: enable
def test_pck_accuracy_interface(metric_kwargs):
    num_keypoints = 4
    keypoints = np.random.random((1, num_keypoints, 2)) * 10
    prediction = {'coords': keypoints}
    keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
    groundtruth = {
        'coords': keypoints,
        'mask': keypoints_visible,
        'bbox_size': np.random.random((1, 2)) * 10,  # for `bbox` norm_item
        'head_size': np.random.random((1, 2)) * 10,  # for `head` norm_item
        'torso_size': np.random.random((1, 2)) * 10,  # for `torso` norm_item
    }
    pck_accuracy = PCKAccuracy(**metric_kwargs)
    results = pck_accuracy([prediction], [groundtruth])
    assert isinstance(results, dict)


# yapf: disable
@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'prediction', 'groundtruth', 'results'],
    argvalues=[
        (
            {'thr': 0.2, 'norm_item': 'bbox'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8]]])
            },
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 7], [7, 8]]]),
                'mask': np.array([[True, True, True, True]]),
                'bbox_size': np.array([[10, 10]])
            },
            {'PCK@0.2': 0.75}
        ),
        (
            {'thr': 0.3, 'norm_item': 'head'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8]]])
            },
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 7], [7, 8]]]),
                'mask': np.array([[True, True, True, True]]),
                'head_size': np.array([[10, 10]])
            },
            {'PCKh@0.3': 1.0}
        ),
        (
            {'thr': 0.1, 'norm_item': 'torso'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8]]])
            },
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 7], [7, 8]]]),
                'mask': np.array([[True, True, True, True]]),
                'torso_size': np.array([[10, 10]])
            },
            {'tPCK@0.1': 0.75}
        )
    ]
)
# yapf: enable
def test_pck_accuracy_accurate(metric_kwargs, prediction, groundtruth,
                               results):
    pck_accuracy = PCKAccuracy(**metric_kwargs)
    assert pck_accuracy([prediction], [groundtruth]) == results


# yapf: disable
@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'thr': 1.0, 'norm_item': 'bbox'},
        {'thr': 0.2, 'norm_item': 'head'},
        {'thr': 0.2, 'norm_item': 'torso'},
        {'thr': 0.2, 'norm_item': ['bbox', 'torso']},
    ]
)
# yapf: enable
def test_jhmdb_pck_accuracy_interface(metric_kwargs):
    num_keypoints = 15
    keypoints = np.random.random((1, num_keypoints, 2)) * 10
    prediction = {'coords': keypoints}
    keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
    groundtruth = {
        'coords': keypoints,
        'mask': keypoints_visible,
        'bbox_size': np.random.random((1, 2)) * 10,  # for `bbox` norm_item
        'head_size': np.random.random((1, 2)) * 10,  # for `head` norm_item
        'torso_size': np.random.random((1, 2)) * 10,  # for `torso` norm_item
    }
    jhmdb_pck_accuracy = JhmdbPCKAccuracy(**metric_kwargs)
    results = jhmdb_pck_accuracy([prediction], [groundtruth])
    assert isinstance(results, dict)


# yapf: disable
@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'prediction', 'groundtruth', 'results'],
    argvalues=[
        (
            {'thr': 0.2, 'norm_item': 'bbox'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [7, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [4, 8], [6, 9], [6, 2], [1, 2], [2, 3]]])
            },
            {
                'coords': np.array([[
                    [1, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [1, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [1, 7], [6, 9], [6, 2], [1, 2], [2, 3]]]),
                'mask': np.array([[
                    True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True, True]]),
                'bbox_size': np.array([[10, 10]])
            },
            {
                'PCK@0.2': 0.866666666,
                'Head PCK': 1.0,
                'Sho PCK': 1.0,
                'Elb PCK': 1.0,
                'Wri PCK': 1.0,
                'Hip PCK': 0.5,
                'Knee PCK': 0.5,
                'Ank PCK': 1.0,
                'Mean PCK': 0.86666666
            }
        ),
        (
            {'thr': 0.2, 'norm_item': 'head'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [7, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [4, 8], [6, 9], [6, 2], [1, 2], [2, 3]]])
            },
            {
                'coords': np.array([[
                    [1, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [1, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [1, 7], [6, 9], [6, 2], [1, 2], [2, 3]]]),
                'mask': np.array([[
                    True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True, True]]),
                'head_size': np.array([[10, 10]])
            },
            {'PCKh@0.2': 0.866666666}
        ),
        (
            {'thr': 0.2, 'norm_item': 'torso'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [7, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [4, 8], [6, 9], [6, 2], [1, 2], [2, 3]]])
            },
            {
                'coords': np.array([[
                    [1, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [1, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [1, 7], [6, 9], [6, 2], [1, 2], [2, 3]]]),
                'mask': np.array([[
                    True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True, True]]),
                'torso_size': np.array([[10, 10]])
            },
            {
                'Head tPCK': 1.0,
                'Sho tPCK': 1.0,
                'Elb tPCK': 1.0,
                'Wri tPCK': 1.0,
                'Hip tPCK': 0.5,
                'Knee tPCK': 0.5,
                'Ank tPCK': 1.0,
                'Mean tPCK': 0.86666666
            }
        )
    ]
)
# yapf: enable
def test_jhmdb_pck_accuracy_accurate(metric_kwargs, prediction, groundtruth,
                                     results):
    jhmdb_pck_accuracy = JhmdbPCKAccuracy(**metric_kwargs)
    metric_results = jhmdb_pck_accuracy([prediction], [groundtruth])
    assert len(metric_results) == len(results)
    for key in metric_results:
        np.testing.assert_allclose(metric_results[key], results[key])


# yapf: disable
@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {'thr': 1.0, 'norm_item': 'bbox'},
        {'thr': 0.2, 'norm_item': 'head'},
        {'thr': 0.2, 'norm_item': 'torso'},
        {'thr': 0.2, 'norm_item': ['bbox', 'torso']},
    ]
)
# yapf: enable
def test_mpii_pck_accuracy_interface(metric_kwargs):
    num_keypoints = 16
    keypoints = np.random.random((1, num_keypoints, 2)) * 10
    prediction = {'coords': keypoints}
    keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
    groundtruth = {
        'coords': keypoints,
        'mask': keypoints_visible,
        'bbox_size': np.random.random((1, 2)) * 10,  # for `bbox` norm_item
        'head_size': np.random.random((1, 2)) * 10,  # for `head` norm_item
        'torso_size': np.random.random((1, 2)) * 10,  # for `torso` norm_item
    }
    mpii_pck_accuracy = MpiiPCKAccuracy(**metric_kwargs)
    results = mpii_pck_accuracy([prediction], [groundtruth])
    assert isinstance(results, dict)


# yapf: disable
@pytest.mark.parametrize(
    argnames=['metric_kwargs', 'prediction', 'groundtruth', 'results'],
    argvalues=[
        (
            {'thr': 0.2, 'norm_item': 'bbox'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [7, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [4, 8], [6, 9], [6, 2], [1, 2], [2, 3],
                    [9, 9]]])
            },
            {
                'coords': np.array([[
                    [1, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [1, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [1, 7], [6, 9], [6, 2], [1, 2], [2, 3],
                    [9, 9]]]),
                'mask': np.array([[
                    True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True]]),
                'bbox_size': np.array([[10, 10]])
            },
            {'PCK@0.2': 0.875}
        ),
        (
            {'thr': 0.2, 'norm_item': 'head'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [7, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [4, 8], [6, 9], [6, 2], [1, 2], [2, 3],
                    [9, 9]]])
            },
            {
                'coords': np.array([[
                    [1, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [1, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [1, 7], [6, 9], [6, 2], [1, 2], [2, 3],
                    [9, 9]]]),
                'mask': np.array([[
                    True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True]]),
                'head_size': np.array([[10, 10]])
            },
            {
                'Head': 100,
                'Shoulder': 100,
                'Elbow': 100,
                'Wrist': 50,
                'Hip': 100,
                'Knee': 100,
                'Ankle': 0.0,
                'PCKh': 78.57142857142856,
                'PCKh@0.1': 0.0
            }
        ),
        (
            {'thr': 0.2, 'norm_item': 'torso'},
            {
                'coords': np.array([[
                    [2, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [7, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [4, 8], [6, 9], [6, 2], [1, 2], [2, 3],
                    [9, 9]]])
            },
            {
                'coords': np.array([[
                    [1, 4], [6, 8], [6, 9], [7, 8], [3, 4],
                    [1, 8], [8, 9], [8, 8], [1, 9], [2, 8],
                    [1, 7], [6, 9], [6, 2], [1, 2], [2, 3],
                    [9, 9]]]),
                'mask': np.array([[
                    True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True]]),
                'torso_size': np.array([[10, 10]])
            },
            {'tPCK@0.2': 0.875}
        )
    ]
)
# yapf: enable
def test_mpii_pck_accuracy_accurate(metric_kwargs, prediction, groundtruth,
                                    results):
    mpii_pck_accuracy = MpiiPCKAccuracy(**metric_kwargs)
    metric_results = mpii_pck_accuracy([prediction], [groundtruth])
    assert len(metric_results) == len(results)
    for key in metric_results:
        np.testing.assert_allclose(metric_results[key], results[key])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
