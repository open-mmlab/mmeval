# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


class MAE(BaseMetric):
    """Mean Absolute Error metric for image.

    mean(abs(a-b))

    Args:
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> from mmeval import MAE
        >>> import numpy as np
        >>> mae = MAE()
        >>> preds = [np.ones((32, 32, 3))]
        >>> gts = [np.ones((32, 32, 3)) * 2]
        >>> mask = np.ones((32, 32, 3)) * 2
        >>> mask[:16] *= 0
        >>> mae(preds, gts, [mask])
        {'mae': 0.003921568627}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add(self, preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], masks: Sequence[np.ndarray] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add MAE score of batch to ``self._results``

        Args:
            preds (Sequence[np.ndarray]): Predictions of the model.
            gts (Sequence[np.ndarray]): The ground truth images.
            masks (Sequence[np.ndarray]): Mask images.
        """

        for i, (pred, gt) in enumerate(zip(preds, gts)):
            assert gt.shape == pred.shape, (
                f'Image shapes are different: {gt.shape}, {pred.shape}.')
            if masks is None:
                self._results.append(self._compute_mae(gt, pred))
            else:
                self._results.append(self._compute_mae(gt, pred, masks[i]))

    def compute_metric(self, results: List[np.float32]) -> Dict[str, float]:
        """Compute the MAE metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float32]): A list that consisting the MAE score.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed MAE metric.
        """

        return {'mae': float(np.array(results).mean())}

    @staticmethod
    def _compute_mae(gt: np.ndarray,
                     pred: np.ndarray,
                     mask: np.ndarray = None) -> np.float32:
        """Calculate MAE (Mean Absolute Error).

        Args:
            gt (np.ndarray): Images with range [0, 255].
            pred (np.ndarray): Images with range [0, 255].
            mask (np.ndarray): Mask of evaluation.

        Returns:
            np.float32: MAE result.
        """

        gt = gt / 255.
        pred = pred / 255.

        diff = gt - pred
        diff = abs(diff)

        if mask is not None:
            diff *= mask  # broadcast for channel dimension
            scale = np.prod(diff.shape) / np.prod(mask.shape)
            result = diff.sum() / (mask.sum() * scale + 1e-12)
        else:
            result = diff.mean()

        return result
