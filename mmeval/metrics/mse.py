# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Optional, Sequence

from mmeval.core import BaseMetric


class MSE(BaseMetric):
    """Mean Squared Error metric for image.

    Formula: mean((a-b)^2).

    Args:
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:
        >>> from mmeval import MSE
        >>> import numpy as np
        >>> mse = MSE()
        >>> preds = [np.ones((32, 32, 3))]
        >>> gts = [np.ones((32, 32, 3)) * 2]
        >>> mask = np.ones((32, 32, 3)) * 2
        >>> mask[:16] *= 0
        >>> mse(preds, gts, [mask])
        {'mse': 0.000015378700496}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add(self, predictions: Sequence[np.ndarray], groundtruths: Sequence[np.ndarray], masks: Optional[Sequence[np.ndarray]] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add MSE score of batch to ``self._results``

        Args:
            predictions (Sequence[np.ndarray]): Predictions of the model.
            groundtruths (Sequence[np.ndarray]): The ground truth images.
            masks (Sequence[np.ndarray], optional): Mask images.
        """

        for i, (prediction,
                groundtruth) in enumerate(zip(predictions, groundtruths)):
            assert groundtruth.shape == prediction.shape, (
                f'Image shapes are different: \
                    {groundtruth.shape}, {prediction.shape}.')
            if masks is None:
                self._results.append(self.compute_mse(groundtruth, prediction))
            else:
                self._results.append(
                    self.compute_mse(groundtruth, prediction, masks[i]))

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
    def compute_mse(groundtruth: np.ndarray,
                    prediction: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> np.float32:
        """Calculate MSE (Mean Squared Error).

        Args:
            groundtruth (np.ndarray): Images with range [0, 255].
            prediction (np.ndarray): Images with range [0, 255].
            mask (np.ndarray, optional): Mask of evaluation.

        Returns:
            np.float32: MSE result.
        """

        groundtruth = groundtruth / 255.
        prediction = prediction / 255.

        diff = groundtruth - prediction
        diff *= diff

        if mask is not None:
            diff *= mask
            result = diff.sum() / mask.sum()
        else:
            result = diff.mean()

        return result
