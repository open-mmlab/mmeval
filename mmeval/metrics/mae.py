# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Optional, Sequence

from mmeval.core import BaseMetric


class MeanAbsoluteError(BaseMetric):
    """Mean Absolute Error metric for image.

    Formula: mean(abs(a-b)).

    Args:
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import MeanAbsoluteError as MAE
        >>> import numpy as np
        >>>
        >>> mae = MAE()
        >>> gts = np.random.randint(0, 255, size=(3, 32, 32))
        >>> preds = np.random.randint(0, 255, size=(3, 32, 32))
        >>> mae(preds, gts)  # doctest: +ELLIPSIS
        {'mae': ...}

    Calculate MeanAbsoluteError between 2 images with mask:

        >>> img1 = np.ones((32, 32, 3))
        >>> img2 = np.ones((32, 32, 3)) * 2
        >>> mask = np.ones((32, 32, 3)) * 2
        >>> mask[:16] *= 0
        >>> MAE.compute_mae(img1, img2, mask)
        0.003921568627
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add(self, predictions: Sequence[np.ndarray], groundtruths: Sequence[np.ndarray], masks: Optional[Sequence[np.ndarray]] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add MeanAbsoluteError score of batch to ``self._results``

        Args:
            predictions (Sequence[np.ndarray]): Predictions of the model.
            groundtruths (Sequence[np.ndarray]): The ground truth images.
            masks (Sequence[np.ndarray], optional): Mask images.
                Defaults to None.
        """

        for i, (prediction,
                groundtruth) in enumerate(zip(predictions, groundtruths)):
            assert groundtruth.shape == prediction.shape, (
                f'Image shapes are different: \
                    {groundtruth.shape}, {prediction.shape}.')
            if masks is None:
                result = self.compute_mae(prediction, groundtruth)
            else:
                # when prediction is a image
                if len(prediction.shape) <= 3:
                    result = self.compute_mae(prediction, groundtruth,
                                              masks[i])
                # when prediction is a video
                else:
                    result_sum = 0
                    for j in range(prediction.shape[0]):
                        result_sum += self.compute_mae(prediction[j],
                                                       groundtruth[j],
                                                       masks[i][j])
                    result = result_sum / prediction.shape[0]
            self._results.append(result)

    def compute_metric(self, results: List[np.float32]) -> Dict[str, float]:
        """Compute the MeanAbsoluteError metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float32]): A list that consisting the
                MeanAbsoluteError score. This list has already been
                synced across all ranks.

        Returns:
            Dict[str, float]: The computed MeanAbsoluteError metric.
        """

        return {'mae': float(np.array(results).mean())}

    @staticmethod
    def compute_mae(prediction: np.ndarray,
                    groundtruth: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> np.float32:
        """Calculate MeanAbsoluteError (Mean Absolute Error).

        Args:
            prediction (np.ndarray): Images with range [0, 255].
            groundtruth (np.ndarray): Images with range [0, 255].
            mask (np.ndarray, optional): Mask of evaluation.

        Returns:
            np.float32: MeanAbsoluteError result.
        """

        prediction = prediction / 255.
        groundtruth = groundtruth / 255.

        diff = groundtruth - prediction
        diff = abs(diff)

        if mask is not None:
            diff *= mask  # broadcast for channel dimension
            scale = np.prod(diff.shape) / np.prod(mask.shape)
            result = diff.sum() / (mask.sum() * scale + 1e-12)
        else:
            result = diff.mean()

        return result


# Keep the deprecated metric name as an alias.
# The deprecated Metric names will be removed in 1.0.0!
MAE = MeanAbsoluteError
