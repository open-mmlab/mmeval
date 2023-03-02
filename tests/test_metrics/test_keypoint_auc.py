# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from unittest import TestCase

from mmeval.metrics import KeypointAUC


class TestKeypointAUC(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.output = np.zeros((1, 5, 2))
        self.target = np.zeros((1, 5, 2))
        # first channel
        self.output[0, 0] = [10, 4]
        self.target[0, 0] = [10, 0]
        # second channel
        self.output[0, 1] = [10, 18]
        self.target[0, 1] = [10, 10]
        # third channel
        self.output[0, 2] = [0, 0]
        self.target[0, 2] = [0, -1]
        # fourth channel
        self.output[0, 3] = [40, 40]
        self.target[0, 3] = [30, 30]
        # fifth channel
        self.output[0, 4] = [20, 10]
        self.target[0, 4] = [0, 10]

        self.keypoints_visible = np.array([[True, True, False, True, True]])

    def test_auc_evaluate(self):
        """test AUC evaluation metric."""
        # case 1: norm_factor=20, num_thrs=4
        auc_metric = KeypointAUC(norm_factor=20, num_thrs=4)
        target = {'AUC@4': 0.375}

        prediction = {'coords': self.output}
        groundtruth = {'coords': self.target, 'mask': self.keypoints_visible}
        predictions = [prediction]
        groundtruths = [groundtruth]

        auc_results = auc_metric(predictions, groundtruths)
        self.assertDictEqual(auc_results, target)

        # case 2: use ``add`` multiple times then ``compute``
        auc_metric._results = []
        preds1 = [{'coords': self.output[:3]}]
        preds2 = [{'coords': self.output[3:]}]
        gts1 = [{
            'coords': self.target[:3],
            'mask': self.keypoints_visible[:3]
        }]
        gts2 = [{
            'coords': self.target[3:],
            'mask': self.keypoints_visible[3:]
        }]

        auc_metric.add(preds1, gts1)
        auc_metric.add(preds2, gts2)

        auc_results = auc_metric.compute_metric(auc_metric._results)
        self.assertDictEqual(auc_results, target)
