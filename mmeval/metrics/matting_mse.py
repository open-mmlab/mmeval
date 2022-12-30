# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


def average(results, key):
    """Average of key in results(list[dict]).

    Args:
        results (list[dict]): A list of dict containing the necessary data.
        key (str): The key of target data.

    Returns:
        result: The average result.
    """

    total = 0
    n = 0
    for batch_result in results:
        batch_size = batch_result.get('batch_size', 1)
        total += batch_result[key] * batch_size
        n += batch_size

    return total / n


class MattingMSE(BaseMetric):
    """Mean Squared Error metric for image matting.

    This metric compute per-pixel squared error average across all
    pixels.
    i.e. mean((a-b)^2) / norm_const

    .. note::

        Current implementation assume image / alpha / trimap array in numpy
        format and with pixel value ranging from 0 to 255.

    .. note::

        pred_alpha should be masked by trimap before passing
        into this metric

    Default prefix: ''

    Args:
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
        norm_const (int): Divide the result to reduce its magnitude.
            Default to 1000.

    Metrics:
        - MattingMSE (float): Mean of Squared Error

    Examples:

        >>> from mmeval import MattingMSE
        >>> import numpy as np
        >>>
        >>> mattingmse = MattingMSE()
        >>> gts = np.random.randint(0, 255, size=(3, 32, 32))
        >>> preds = np.random.randint(0, 255, size=(3, 32, 32))
        >>> trimap = np.random.choice(a=(0,128,255), size=(3, 32, 32))
        >>> mattingmse(preds, gts, trimap)  # doctest: +ELLIPSIS
        {'MattingMSE': ...}
    """

    default_prefix = ''

    def __init__(self, norm_const=1000, **kwargs) -> None:
        self.norm_const = norm_const
        super().__init__(**kwargs)

    def add(self, pred_alphas: Sequence[np.ndarray], gt_alphas: Sequence[np.ndarray], trimaps: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add MattingMSE score of batch to ``self._results``

        Args:
        pred_alpha: Pred_alpha data of predictions.
        ori_alpha: Ori_alpha data of data_batch.
        ori_trimap: Ori_trimap data of data_batch.
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

            self._results.append({'mse': mse_result})

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
        mse = average(results, 'mse')

        return {'MattingMSE': mse}
