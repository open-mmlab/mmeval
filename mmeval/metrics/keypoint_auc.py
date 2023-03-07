# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List

from mmeval.core.base_metric import BaseMetric
from .pck_accuracy import keypoint_pck_accuracy


def keypoint_auc_accuracy(prediction: np.ndarray,
                          groundtruth: np.ndarray,
                          mask: np.ndarray,
                          norm_factor: float,
                          num_thrs: int = 20) -> float:
    """Calculate the Area under curve (AUC) of keypoint PCK accuracy.

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        prediction (np.ndarray[N, K, 2]): Predicted keypoint location.
        groundtruth (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (float): Normalization factor.
        num_thrs (int): number of thresholds to calculate auc.

    Returns:
        float: Area under curve (AUC) of keypoint PCK accuracy.
    """
    nor = np.tile(
        np.array([[norm_factor, norm_factor]]), (prediction.shape[0], 1))
    thrs = [1.0 * i / num_thrs for i in range(num_thrs)]
    avg_accs = []
    for thr in thrs:
        _, avg_acc, _ = keypoint_pck_accuracy(prediction, groundtruth, mask,
                                              thr, nor)
        avg_accs.append(avg_acc)

    auc = 0
    for i in range(num_thrs):
        auc += 1.0 / num_thrs * avg_accs[i]
    return auc


class KeypointAUC(BaseMetric):
    """AUC evaluation metric.

    Calculate the Area Under Curve (AUC) of keypoint PCK accuracy.

    By altering the threshold percentage in the calculation of PCK accuracy,
    AUC can be generated to further evaluate the pose estimation algorithms.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        norm_factor (float): AUC normalization factor, Defaults to 30 (pixels).
        num_thrs (int): Number of thresholds to calculate AUC. Defaults to 20.
        **kwargs: Keyword parameters passed to :class:`mmeval.BaseMetric`. Must
            include ``dataset_meta`` in order to compute the metric.

    Examples:

        >>> from mmeval import KeypointAUC
        >>> import numpy as np
        >>> auc_metric = KeypointAUC(norm_factor=20, num_thrs=4)
        >>> output = np.array([[[10.,  4.],
        ...     [10., 18.],
        ...     [ 0.,  0.],
        ...     [40., 40.],
        ...     [20., 10.]]])
        >>> target = np.array([[[10.,  0.],
        ...     [10., 10.],
        ...     [ 0., -1.],
        ...     [30., 30.],
        ...     [ 0., 10.]]])
        >>> keypoints_visible = np.array([[True, True, False, True, True]])
        >>> num_keypoints = 15
        >>> prediction = {'coords': output}
        >>> groundtruth = {'coords': target, 'mask': keypoints_visible}
        >>> predictions = [prediction]
        >>> groundtruths = [groundtruth]
        >>> auc_metric(predictions, groundtruths)
        OrderedDict([('AUC@4', 0.375)])
    """

    def __init__(self,
                 norm_factor: float = 30,
                 num_thrs: int = 20,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm_factor = norm_factor
        self.num_thrs = num_thrs

    def add(self, predictions: List[Dict], groundtruths: List[Dict]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of predictions and groundtruths and add the
        intermediate results to `self._results`.

        Args:
            predictions (Sequence[dict]): Predictions from the model.
                Each prediction dict has the following keys:

                - coords (np.ndarray, [1, K, D]): predicted keypoints
                  coordinates

            groundtruths (Sequence[dict]): The ground truth labels.
                Each groundtruth dict has the following keys:

                - coords (np.ndarray, [1, K, D]): ground truth keypoints
                  coordinates
                - mask (np.ndarray, [1, K]): ground truth keypoints_visible
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

        self.logger.info(f'Evaluating {self.__class__.__name__}...')

        auc = keypoint_auc_accuracy(pred_coords, gt_coords, mask,
                                    self.norm_factor, self.num_thrs)

        return {f'AUC@{self.num_thrs}': auc}
