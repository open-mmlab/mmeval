# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from unittest import TestCase

from mmeval.metrics import KeypointNME


class TestKeypointNME(TestCase):
    coco_sigmas = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ]).astype(np.float32)
    coco_keypoint_id2name = {
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle',
    }
    coco_dataset_meta = {
        'dataset_name': 'coco',
        'num_keypoints': 17,
        'sigmas': coco_sigmas,
        'keypoint_id2name': coco_keypoint_id2name
    }

    def _generate_data(self,
                       batch_size: int = 1,
                       num_keypoints: int = 5,
                       norm_item: str = 'box_size'):
        """Generate data_batch and data_samples according to different
        settings."""
        self.predictions = []
        self.groundtruths = []

        for i in range(batch_size):
            keypoints = np.zeros((1, num_keypoints, 2))
            keypoints[0, i] = [0.5 * i, 0.5 * i]
            keypoints_ = keypoints.copy()
            keypoints_[0, i] += np.array([0.2, 0.2]) * i
            keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
            keypoints_visible[0, (2 * i) % 8] = False

            groundtruth = {
                'coords': keypoints,
                'mask': keypoints_visible,
                norm_item: np.array([[10]]) * i
            }
            prediction = {'coords': keypoints_}

            self.predictions.append(prediction)
            self.groundtruths.append(groundtruth)

    def test_nme_evaluate(self):
        """test NME evaluation metric."""
        # test when norm_mode = 'use_norm_item'
        # test norm_item = 'box_size' like in `AFLWDataset`
        norm_item = 'box_size'
        self.aflw_dataset_meta = {
            'dataset_name': 'aflw',
            'num_keypoints': 19,
            'sigmas': np.array([]),
        }
        self._generate_data(
            batch_size=4, num_keypoints=19, norm_item=norm_item)

        nme_metric = KeypointNME(
            norm_mode='use_norm_item',
            norm_item=norm_item,
            dataset_meta=self.aflw_dataset_meta,
        )
        nme_results = nme_metric(self.predictions, self.groundtruths)
        self.assertAlmostEqual(nme_results['NME'], 0.00157135)

        # test with multiple ``add`` followed by ``compute``
        nme_metric._results = []
        nme_metric.add(self.predictions[:2], self.groundtruths[:2])
        nme_metric.add(self.predictions[2:], self.groundtruths[2:])
        nme_results = nme_metric.compute_metric(nme_metric._results)
        self.assertAlmostEqual(nme_results['NME'], 0.00157135)

        # test when norm_mode = 'keypoint_distance'
        # when `keypoint_indices = None`,
        # use default `keypoint_indices` like in `Horse10Dataset`
        self.horse10_dataset_meta = {
            'dataset_name': 'horse10',
            'num_keypoints': 22,
            'sigmas': np.array([]),
        }

        nme_metric = KeypointNME(
            norm_mode='keypoint_distance',
            dataset_meta=self.horse10_dataset_meta)
        self._generate_data(batch_size=4, num_keypoints=22)

        nme_results = nme_metric(self.predictions, self.groundtruths)
        self.assertAlmostEqual(nme_results['NME'], 0.01904762)

        # test when norm_mode = 'keypoint_distance'
        # specify custom `keypoint_indices`
        keypoint_indices = [2, 4]
        nme_metric = KeypointNME(
            norm_mode='keypoint_distance',
            keypoint_indices=keypoint_indices,
            dataset_meta=self.coco_dataset_meta)
        self._generate_data(batch_size=2, num_keypoints=17)

        nme_results = nme_metric(self.predictions, self.groundtruths)
        self.assertAlmostEqual(nme_results['NME'], 0.0)

    def test_exceptions_and_warnings(self):
        """test exceptions and warnings."""
        # test invalid norm_mode
        with self.assertRaisesRegex(
                KeyError,
                "`norm_mode` should be 'use_norm_item' or 'keypoint_distance'"
        ):
            nme_metric = KeypointNME(norm_mode='invalid')

        # test when norm_mode = 'use_norm_item' but do not specify norm_item
        with self.assertRaisesRegex(
                KeyError, '`norm_mode` is set to `"use_norm_item"`, '
                'please specify the `norm_item`'):
            nme_metric = KeypointNME(norm_mode='use_norm_item', norm_item=None)

        # test when norm_mode = 'use_norm_item'
        # but the `norm_item` do not in data_info
        with self.assertRaisesRegex(
                AssertionError,
                'The ground truth data info do not have the expected '
                'normalized factor'):
            nme_metric = KeypointNME(
                norm_mode='use_norm_item',
                norm_item='norm_item1',
                dataset_meta=self.coco_dataset_meta,
            )
            self._generate_data(norm_item='norm_item2')
            # raise AssertionError here
            _ = nme_metric(self.predictions, self.groundtruths)

        # test when norm_mode = 'keypoint_distance'
        # but the `dataset_meta` is None
        with self.assertRaisesRegex(
                AssertionError, 'When `norm_mode` is `keypoint_distance`, '
                '`dataset_meta` cannot be None.'):
            nme_metric = KeypointNME(
                norm_mode='keypoint_distance', keypoint_indices=[0, 1])

            self._generate_data()
            # raise AssertionError here
            _ = nme_metric(self.predictions, self.groundtruths)

        # test when norm_mode = 'keypoint_distance', `keypoint_indices` = None
        # but the dataset_name not in `DEFAULT_KEYPOINT_INDICES`
        with self.assertRaisesRegex(
                KeyError, 'can not find the keypoint_indices in '
                '`DEFAULT_KEYPOINT_INDICES`'):
            nme_metric = KeypointNME(
                norm_mode='keypoint_distance',
                keypoint_indices=None,
                dataset_meta=self.coco_dataset_meta)

            self._generate_data()
            # raise AssertionError here
            _ = nme_metric(self.predictions, self.groundtruths)

        # test when len(keypoint_indices) is not 2
        with self.assertRaisesRegex(
                AssertionError,
                'The keypoint indices used for normalization should be a pair.'
        ):
            nme_metric = KeypointNME(
                norm_mode='keypoint_distance',
                keypoint_indices=[0, 1, 2],
                dataset_meta=self.coco_dataset_meta)

            self._generate_data()
            # raise AssertionError here
            _ = nme_metric(self.predictions, self.groundtruths)

        # test when dataset does not contain the required keypoint
        with self.assertRaisesRegex(AssertionError,
                                    'dataset does not contain the required'):
            nme_metric = KeypointNME(
                norm_mode='keypoint_distance',
                keypoint_indices=[17, 18],
                dataset_meta=self.coco_dataset_meta)

            self._generate_data()
            # raise AssertionError here
            _ = nme_metric(self.predictions, self.groundtruths)
