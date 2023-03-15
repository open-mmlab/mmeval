# Copyright (c) OpenMMLab. All rights reserved.Dict
import cv2
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


class ConnectivityError(BaseMetric):
    """Connectivity error for evaluating alpha matte prediction.

    Args:
        step (float): Step of threshold when computing intersection between
            `alpha` and `pred_alpha`. Default to 0.1 .
        norm_const (int): Divide the result to reduce its magnitude.
            Defaults to 1000 .
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Note:
        The current implementation assumes the image / alpha / trimap
        a numpy array with pixel values ranging from 0 to 255.

        The pred_alpha should be masked by trimap before passing
        into this metric.

        The trimap is the most commonly used prior knowledge. As the
        name implies, trimap is a ternary graph and each pixel
        takes one of {0, 128, 255}, representing the foreground, the
        unknown and the background respectively.

    Examples:

        >>> from mmeval import ConnectivityError
        >>> import numpy as np
        >>>
        >>> connectivity_error = ConnectivityError()
        >>> pred_alpha = np.zeros((32, 32), dtype=np.uint8)
        >>> gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
        >>> trimap = np.zeros((32, 32), dtype=np.uint8)
        >>> trimap[:16, :16] = 128
        >>> trimap[16:, 16:] = 255
        >>> connectivity_error(pred_alpha, gt_alpha, trimap)
        {'connectivity_error': ...}
    """

    def __init__(self,
                 step: float = 0.1,
                 norm_const: int = 1000,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = step
        self.norm_const = norm_const

    def add(self, pred_alphas: Sequence[np.ndarray], gt_alphas: Sequence[np.ndarray], trimaps: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add ConnectivityError score of batch to ``self._results``

        Args:
            pred_alphas (Sequence[np.ndarray]): Predict the probability
                that pixels belong to the foreground.
            gt_alphas (Sequence[np.ndarray]): Probability that the actual
                pixel belongs to the foreground.
            trimaps (Sequence[np.ndarray]): Broadly speaking, the trimap
                consists of foreground and unknown region.
        """

        for pred_alpha, gt_alpha, trimap in zip(pred_alphas, gt_alphas,
                                                trimaps):
            assert pred_alpha.shape == gt_alpha.shape, 'The shape of ' \
                '`pred_alpha` and `gt_alpha` should be the same, but got: ' \
                f'{pred_alpha.shape} and {gt_alpha.shape}'

            thresh_steps = np.arange(0, 1 + self.step, self.step)
            round_down_map = -np.ones_like(gt_alpha)
            for i in range(1, len(thresh_steps)):
                gt_alpha_thresh = gt_alpha >= thresh_steps[i]
                pred_alpha_thresh = pred_alpha >= thresh_steps[i]
                intersection = gt_alpha_thresh & pred_alpha_thresh
                intersection = intersection.astype(np.uint8)

                # connected components
                _, output, stats, _ = cv2.connectedComponentsWithStats(
                    intersection, connectivity=4)
                # start from 1 in dim 0 to exclude background
                size = stats[1:, -1]

                # largest connected component of the intersection
                omega = np.zeros_like(gt_alpha)
                if len(size) != 0:
                    max_id = np.argmax(size)
                    # plus one to include background
                    omega[output == max_id + 1] = 1

                mask = (round_down_map == -1) & (omega == 0)
                round_down_map[mask] = thresh_steps[i - 1]

            round_down_map[round_down_map == -1] = 1

            gt_alpha_diff = gt_alpha - round_down_map
            pred_alpha_diff = pred_alpha - round_down_map
            # only calculate difference larger than or equal to 0.15
            gt_alpha_phi = 1 - gt_alpha_diff * (gt_alpha_diff >= 0.15)
            pred_alpha_phi = 1 - pred_alpha_diff * (pred_alpha_diff >= 0.15)

            connectivity_error = np.sum(
                np.abs(gt_alpha_phi - pred_alpha_phi) * (trimap == 128))

            # divide by norm_const to reduce the magnitude of the result
            connectivity_error /= self.norm_const

            self._results.append(connectivity_error)

    def compute_metric(self, results: List) -> Dict[str, float]:
        """Compute the ConnectivityError metric.

        Args:
            results (List): A list that consisting the ConnectivityError score.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed ConnectivityError metric.
            The keys are the names of the metrics,
            and the values are corresponding results.
        """

        return {'connectivity_error': float(np.array(results).mean())}
