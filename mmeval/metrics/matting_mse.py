# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


class MattingMeanSquaredError(BaseMetric):
    """Mean Squared Error metric for image matting.

    This metric computes the per-pixel squared error average across all
    pixels.
    i.e. mean((a-b)^2)

    Args:
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

        >>> from mmeval import MattingMeanSquaredError as MattingMSE
        >>> import numpy as np
        >>>
        >>> matting_mse = MattingMSE()
        >>> pred_alpha = np.zeros((32, 32), dtype=np.uint8)
        >>> gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
        >>> trimap = np.zeros((32, 32), dtype=np.uint8)
        >>> trimap[:16, :16] = 128
        >>> trimap[16:, 16:] = 255
        >>> matting_mse(pred_alpha, gt_alpha, trimap)  # doctest: +ELLIPSIS
        {'matting_mse': ...}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add(self, pred_alphas: Sequence[np.ndarray], gt_alphas: Sequence[np.ndarray], trimaps: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add MattingMeanSquaredError score of batch to ``self._results``

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

            weight_sum = (trimap == 128).sum()
            if weight_sum != 0:
                mse_result = ((pred_alpha - gt_alpha)**2).sum() / weight_sum
            else:
                mse_result = 0

            self._results.append(mse_result)

    def compute_metric(self, results: List) -> Dict[str, float]:
        """Compute the MattingMeanSquaredError metric.

        Args:
            results (List): A list that consisting the
                MattingMeanSquaredError score. This list has already
                been synced across all ranks.

        Returns:
            Dict[str, float]: The computed MattingMeanSquaredError metric.
            The keys are the names of the metrics,
            and the values are corresponding results.
        """

        return {'matting_mse': float(np.array(results).mean())}


# Keep the deprecated metric name as an alias.
# The deprecated Metric names will be removed in 1.0.0!
MattingMSE = MattingMeanSquaredError
