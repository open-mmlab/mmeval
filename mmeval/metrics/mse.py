# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


class MSE(BaseMetric):
    """Mean Squared Error metric for image.

    mean((a-b)^2)

    Args:
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add(self, preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], masks: Sequence[np.ndarray] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add MSE score of batch to ``self._results``
        Args:
            preds (Sequence[np.ndarray]): Predictions of the model.
            gts (Sequence[np.ndarray]): The ground truth images.
            masks (Sequence[np.ndarray]): Mask images.
        """

        for i, data in enumerate(zip(preds, gts)):
            pred, gt = data
            assert gt.shape == pred.shape, (
                f'Image shapes are different: {gt.shape}, {pred.shape}.')
            if masks is None:
                self._results.append(self._compute_mse(gt, pred))
            else:
                self._results.append(self._compute_mse(gt, pred, masks[i]))

    def compute_metric(self, results: List[np.float32]) -> Dict[str, float]:
        """Compute the MSE metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.
        Args:
            results (List[np.float32]): A list that consisting the MSE score.
                This list has already been synced across all ranks.
        Returns:
            Dict[str, float]: The computed MSE metric.
        """

        return {'mse': float(np.array(results).mean())}

    @staticmethod
    def _compute_mse(gt: np.ndarray,
                     pred: np.ndarray,
                     mask: np.ndarray = None) -> np.float32:
        """Calculate MSE (Mean Squared Error).

        Args:
            gt (np.ndarray): Images with range [0, 255].
            pred (np.ndarray): Images with range [0, 255].
            mask (np.ndarray): Mask of evaluation.
        Returns:
            np.float32: MSE result.
        """

        gt = gt / 255.
        pred = pred / 255.

        diff = gt - pred
        diff *= diff

        if mask is not None:
            diff *= mask
            result = diff.sum() / mask.sum()
        else:
            result = diff.mean()

        return result
