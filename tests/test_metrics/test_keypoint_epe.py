# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from unittest import TestCase

from mmeval.metrics import KeypointEndPointError


class TestKeypointEndPointError(TestCase):

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

    def test_epe_evaluate(self):
        """test EPE evaluation metric."""
        # case 1: test normal use case
        epe_metric = KeypointEndPointError()

        prediction = {'coords': self.output}
        groundtruth = {'coords': self.target, 'mask': self.keypoints_visible}
        predictions = [prediction]
        groundtruths = [groundtruth]

        epe_results = epe_metric(predictions, groundtruths)
        self.assertAlmostEqual(epe_results['EPE'], 11.5355339)

        # case 2: use ``add`` multiple times then ``compute``
        epe_metric._results = []
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

        epe_metric.add(preds1, gts1)
        epe_metric.add(preds2, gts2)

        epe_results = epe_metric.compute_metric(epe_metric._results)
        self.assertAlmostEqual(epe_results['EPE'], 11.5355339)
