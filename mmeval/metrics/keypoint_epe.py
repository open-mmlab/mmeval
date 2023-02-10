# Copyright (c) OpenMMLab. All rights reserved.
import logging
import numpy as np
from collections import OrderedDict
from typing import Dict, List

from mmeval.core.base_metric import BaseMetric
from .utils import calc_distances

logger = logging.getLogger(__name__)


def keypoint_epe_accuracy(pred: np.ndarray, gt: np.ndarray,
                          mask: np.ndarray) -> float:
    """Calculate the end-point error.

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        float: Average end-point error.
    """

    distances = calc_distances(
        pred, gt, mask,
        np.ones((pred.shape[0], pred.shape[2]), dtype=np.float32))
    distance_valid = distances[distances != -1]
    return distance_valid.sum() / max(1, len(distance_valid))


class KeypointEndPointError(BaseMetric):
    """EPE evaluation metric.

    Calculate the end-point error (EPE) of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)
    """

    def add(self, predictions: List[Dict], groundtruths: List[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of predictions and groundtruths and add the
        intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): Predictions from the model.

            groundtruths (Sequence[dict]): The ground truth labels.
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            self._results.append((prediction, groundtruth))

    def compute_metric(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # split gt and prediction list
        preds, gts = zip(*results)

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate([pred['coords'] for pred in preds])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([gt['coords'] for gt in gts])
        # mask: [N, K]
        mask = np.concatenate([gt['mask'] for gt in gts])

        logger.info(f'Evaluating {self.__class__.__name__}...')

        epe = keypoint_epe_accuracy(pred_coords, gt_coords, mask)

        metric_results: OrderedDict = OrderedDict()
        metric_results['EPE'] = epe

        return metric_results
