# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


class MattingSAD(BaseMetric):
    """Sum of Absolute Differences metric for image matting.

    This metric compute per-pixel absolute difference and sum across all
    pixels.
    i.e. sum(abs(a-b)) / norm_const

    Args:
        **kwargs:Keyword parameters passed to :class:`BaseMetric`.
        norm_const (int): Divide the result to reduce its magnitude.
            Default to 1000.

    Note:
        The current implementation assumes the image / alpha / trimap
        a numpy array with pixel values ranging from 0 to 255.

        The pred_alpha should be masked by trimap before passing
        into this metric.


    Examples:

        >>> from mmeval import MattingSAD
        >>> import numpy as np
        >>>
        >>> mattingsad = MattingSAD()
        >>> pred_alpha = np.zeros((32, 32), dtype=np.uint8)
        >>> gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
        >>> mattingsad(pred_alpha, gt_alpha)  # doctest: +ELLIPSIS
        {'MattingSAD': ...}
    """

    def __init__(self, norm_const=1000, **kwargs) -> None:
        self.norm_const = norm_const
        super().__init__(**kwargs)

    def add(self, pred_alphas: Sequence[np.ndarray], gt_alphas: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add SAD score of batch to ``self._results``

        Args:
            pred_alpha(Sequence[np.ndarray]): Pred_alpha data of predictions.
            ori_alpha(Sequence[np.ndarray]): Ori_alpha data of data_batch.
        """

        for pred_alpha, gt_alpha in zip(pred_alphas, gt_alphas):
            assert pred_alpha.shape == gt_alpha.shape, 'The shape of ' \
                '`pred_alpha` and `gt_alpha` should be the same, but got: ' \
                f'{pred_alpha.shape} and {gt_alpha.shape}'

            sad_sum = np.abs(pred_alpha - gt_alpha).sum() / self.norm_const

            self._results.append(sad_sum)

    def compute_metric(self, results: List) -> Dict[str, float]:
        """Compute the MattingMSE metric.

        Args:
            results (List): A list that consisting the MattingMSE score.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed MattingMSE metric.
            The keys are the names of the metrics,
            and the values are corresponding results.
        """

        return {'MattingSAD': float(np.array(results).mean())}
