# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from unittest import TestCase

from mmeval.metrics import JhmdbPCKAccuracy, MpiiPCKAccuracy, PCKAccuracy


class TestPCKAccuracy(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.batch_size = 8
        num_keypoints = 15
        self.predictions = []
        self.groundtruths = []

        for i in range(self.batch_size):
            keypoints = np.zeros((1, num_keypoints, 2))
            keypoints[0, i] = [0.5 * i, 0.5 * i]
            keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
            keypoints_visible[0, (2 * i) % 8] = False
            bbox_size = np.random.random((1, 2)) * 20 * i
            head_size = np.random.random((1, 2)) * 10 * i
            torso_size = np.random.random((1, 2)) * 15 * i

            prediction = {'coords': keypoints}
            groundtruth = {
                'coords': keypoints,
                'mask': keypoints_visible,
                'bbox_size': bbox_size,
                'head_size': head_size,
                'torso_size': torso_size
            }

            self.predictions.append(prediction)
            self.groundtruths.append(groundtruth)

    def test_init(self):
        """test metric init method."""
        # test invalid normalized_items
        with self.assertRaisesRegex(
                KeyError, "Should be one of 'bbox', 'head', 'torso'"):
            PCKAccuracy(norm_item='invalid')

    def test_evaluate(self):
        """test PCK accuracy evaluation metric."""
        # test normalized by 'bbox'
        pck_metric = PCKAccuracy(thr=0.5, norm_item='bbox')
        pck_results = pck_metric(self.predictions, self.groundtruths)
        target = {'PCK@0.5': 1.0}
        self.assertDictEqual(pck_results, target)

        # test normalized by 'head_size'
        pckh_metric = PCKAccuracy(thr=0.3, norm_item='head')
        pckh_results = pckh_metric(self.predictions, self.groundtruths)
        target = {'PCKh@0.3': 1.0}
        self.assertDictEqual(pckh_results, target)

        # test normalized by 'torso_size'
        tpck_metric = PCKAccuracy(thr=0.05, norm_item=['bbox', 'torso'])
        tpck_results = tpck_metric(self.predictions, self.groundtruths)
        self.assertIsInstance(tpck_results, dict)
        target = {
            'PCK@0.05': 1.0,
            'tPCK@0.05': 1.0,
        }
        self.assertDictEqual(tpck_results, target)


class TestMpiiAccuracy(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.batch_size = 8
        num_keypoints = 16
        self.predictions = []
        self.groundtruths = []

        for i in range(self.batch_size):
            keypoints = np.zeros((1, num_keypoints, 2))
            keypoints[0, i] = [0.5 * i, 0.5 * i]
            keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
            keypoints_visible[0, (2 * i) % 8] = False
            head_size = np.random.random((1, 2)) * 10 * i

            prediction = {'coords': keypoints}
            groundtruth = {
                'coords': keypoints + 1.0,
                'mask': keypoints_visible,
                'head_size': head_size,
            }

            self.predictions.append(prediction)
            self.groundtruths.append(groundtruth)

    def test_init(self):
        """test metric init method."""
        # test invalid normalized_items
        with self.assertRaisesRegex(
                KeyError, "Should be one of 'bbox', 'head', 'torso'"):
            MpiiPCKAccuracy(norm_item='invalid')

    def test_evaluate(self):
        """test PCK accuracy evaluation metric."""
        # test normalized by 'head_size'
        mpii_pckh_metric = MpiiPCKAccuracy(thr=0.3, norm_item='head')
        pckh_results = mpii_pckh_metric(self.predictions, self.groundtruths)
        target = {
            'Head': 100.0,
            'Shoulder': 100.0,
            'Elbow': 100.0,
            'Wrist': 100.0,
            'Hip': 100.0,
            'Knee': 100.0,
            'Ankle': 100.0,
            'PCKh': 100.0,
            'PCKh@0.1': 100.0,
        }
        self.assertDictEqual(pckh_results, target)


class TestJhmdbPCKAccuracy(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.batch_size = 8
        num_keypoints = 15
        self.predictions = []
        self.groundtruths = []

        for i in range(self.batch_size):
            keypoints = np.zeros((1, num_keypoints, 2))
            keypoints[0, i] = [0.5 * i, 0.5 * i]
            keypoints_visible = np.ones((1, num_keypoints)).astype(bool)
            # keypoints_visible[0, (2 * i) % 8] = False
            bbox_size = np.random.random((1, 2)) * 20 * i
            head_size = np.random.random((1, 2)) * 10 * i
            torso_size = np.random.random((1, 2)) * 15 * i

            prediction = {'coords': keypoints}
            groundtruth = {
                'coords': keypoints,
                'mask': keypoints_visible,
                'bbox_size': bbox_size,
                'head_size': head_size,
                'torso_size': torso_size
            }

            self.predictions.append(prediction)
            self.groundtruths.append(groundtruth)

    def test_init(self):
        """test metric init method."""
        # test invalid normalized_items
        with self.assertRaisesRegex(
                KeyError, "Should be one of 'bbox', 'head', 'torso'"):
            JhmdbPCKAccuracy(norm_item='invalid')

    def test_evaluate(self):
        """test PCK accuracy evaluation metric."""
        # test normalized by 'bbox_size'
        jhmdb_pck_metric = JhmdbPCKAccuracy(thr=0.5, norm_item='bbox')
        pckh_results = jhmdb_pck_metric(self.predictions, self.groundtruths)
        target = {
            'Head PCK': 1.0,
            'Sho PCK': 1.0,
            'Elb PCK': 1.0,
            'Wri PCK': 1.0,
            'Hip PCK': 1.0,
            'Knee PCK': 1.0,
            'Ank PCK': 1.0,
            'Mean PCK': 1.0,
        }
        self.assertDictEqual(pckh_results, target)
