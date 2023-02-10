# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from unittest import TestCase

from mmeval.metrics import KeypointAUC


class TestKeypointAUCandEPE(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        output = np.zeros((1, 5, 2))
        target = np.zeros((1, 5, 2))
        # first channel
        output[0, 0] = [10, 4]
        target[0, 0] = [10, 0]
        # second channel
        output[0, 1] = [10, 18]
        target[0, 1] = [10, 10]
        # third channel
        output[0, 2] = [0, 0]
        target[0, 2] = [0, -1]
        # fourth channel
        output[0, 3] = [40, 40]
        target[0, 3] = [30, 30]
        # fifth channel
        output[0, 4] = [20, 10]
        target[0, 4] = [0, 10]

        keypoints_visible = np.array([[True, True, False, True, True]])

        prediction = {'coords': output}
        groundtruth = {'coords': target, 'mask': keypoints_visible}

        self.predictions = [prediction]
        self.groundtruths = [groundtruth]

    def test_auc_evaluate(self):
        """test AUC evaluation metric."""
        auc_metric = KeypointAUC(norm_factor=20, num_thrs=4)
        auc_results = auc_metric(self.predictions, self.groundtruths)
        target = {'AUC@4': 0.375}
        self.assertDictEqual(auc_results, target)
