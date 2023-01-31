# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


class SumAbsoluteDifferences(BaseMetric):
    """Sum of Absolute Differences metric for image.

    This metric computes per-pixel absolute difference and sum across all
    pixels.
    i.e. sum(abs(a-b)) / norm_const

    Args:
        norm_const (int): Divide the result to reduce its magnitude.
            Default to 1000.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Note:
        The current implementation assumes the image a numpy array
        with pixel values ranging from 0 to 255.

    Examples:

        >>> from mmeval import SumAbsoluteDifferences as SAD
        >>> import numpy as np
        >>>
        >>> sad = SAD()
        >>> prediction = np.zeros((32, 32), dtype=np.uint8)
        >>> groundtruth = np.ones((32, 32), dtype=np.uint8) * 255
        >>> sad(prediction, groundtruth)  # doctest: +ELLIPSIS
        {'sad': ...}
    """

    def __init__(self, norm_const: int = 1000, **kwargs) -> None:
        super().__init__(**kwargs)
        self.norm_const = norm_const

    def add(self, predictions: Sequence[np.ndarray], groundtruths: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add SumAbsoluteDifferences score of batch to ``self._results``

        Args:
            predictions (Sequence[np.ndarray]): Sequence of predicted image.
            groundtruths (Sequence[np.ndarray]): Sequence of groundtruth image.
        """

        for prediction, groundtruth in zip(predictions, groundtruths):
            assert prediction.shape == groundtruth.shape, 'The shape of ' \
                '`prediction` and `groundtruth` should be the same, but got:' \
                f'{prediction.shape} and {groundtruth.shape}'

            sad_sum = np.abs(prediction - groundtruth).sum() / self.norm_const

            self._results.append(sad_sum)

    def compute_metric(self, results: List) -> Dict[str, float]:
        """Compute the SumAbsoluteDifferences metric.

        Args:
            results (List): A list that consisting the
                SumAbsoluteDifferences score. This list has already been
                synced across all ranks.

        Returns:
            Dict[str, float]: The computed SumAbsoluteDifferences metric.
            The keys are the names of the metrics,
            and the values are corresponding results.
        """

        return {'sad': float(np.array(results).mean())}


# Keep the deprecated metric name as an alias.
# The deprecated Metric names will be removed in 1.0.0!
SAD = SumAbsoluteDifferences
