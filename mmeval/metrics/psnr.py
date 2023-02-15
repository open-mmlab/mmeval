# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Optional, Sequence

from mmeval.core import BaseMetric
from .utils import reorder_and_crop


class PeakSignalNoiseRatio(BaseMetric):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PeakSignalNoiseRatio calculation.
            Defaults to 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Defaults to 'CHW'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Defaults to None.
        channel_order (str): The channel order of image. Defaults to 'rgb'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import PeakSignalNoiseRatio as PSNR
        >>> import numpy as np
        >>>
        >>> psnr = PSNR(input_order='CHW', convert_to='Y', channel_order='rgb')
        >>> gts = np.random.randint(0, 255, size=(3, 32, 32))
        >>> preds = np.random.randint(0, 255, size=(3, 32, 32))
        >>> psnr(preds, gts)  # doctest: +ELLIPSIS
        {'psnr': ...}

    Calculate PeakSignalNoiseRatio between 2 single channel images:

        >>> img1 = np.ones((32, 32))
        >>> img2 = np.ones((32, 32))
        >>> PSNR.compute_psnr(img1, img2)
        49.45272242415597
    """

    def __init__(self,
                 crop_border: int = 0,
                 input_order: str = 'CHW',
                 convert_to: Optional[str] = None,
                 channel_order: str = 'rgb',
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert input_order.upper() in [
            'CHW', 'HWC'
        ], (f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

        if convert_to is not None:
            assert convert_to.upper() == 'Y', (
                'Wrong color model. Supported values are "Y" and None.')
            assert channel_order.upper() in [
                'BGR', 'RGB'
            ], ('Only support `rgb2y` and `bgr2y`, but the channel_order '
                f'is {channel_order}')

        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to
        self.channel_order = channel_order

    def add(self, predictions: Sequence[np.ndarray], groundtruths: Sequence[np.ndarray], channel_order: Optional[str] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add PeakSignalNoiseRatio score of batch to ``self._results``

        Args:
            predictions (Sequence[np.ndarray]): Predictions of the model.
            groundtruths (Sequence[np.ndarray]): The ground truth images.
            channel_order (Optional[str]): The channel order of the input
                samples. If not passed, will set as :attr:`self.channel_order`.
                Defaults to None.
        """

        if channel_order is None:
            channel_order = self.channel_order

        for prediction, groundtruth in zip(predictions, groundtruths):
            assert groundtruth.shape == prediction.shape, (
                f'Image shapes are different: \
                    {groundtruth.shape}, {prediction.shape}.')
            groundtruth = reorder_and_crop(
                groundtruth,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=channel_order)
            prediction = reorder_and_crop(
                prediction,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=channel_order)

            if len(prediction.shape) == 3:
                prediction = np.expand_dims(prediction, axis=0)
                groundtruth = np.expand_dims(groundtruth, axis=0)
            _psnr_score = []
            for i in range(prediction.shape[0]):
                _psnr_score.append(
                    self.compute_psnr(prediction[i], groundtruth[i]))
            self._results.append(np.array(_psnr_score).mean())

    def compute_metric(self, results: List[np.float64]) -> Dict[str, float]:
        """Compute the PeakSignalNoiseRatio metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float64]): A list that consisting the
                PeakSignalNoiseRatio score. This list has already been
                synced across all ranks.

        Returns:
            Dict[str, float]: The computed PeakSignalNoiseRatio metric.
        """

        return {'psnr': float(np.array(results).mean())}

    @staticmethod
    def compute_psnr(prediction: np.ndarray,
                     groundtruth: np.ndarray) -> np.float64:
        """Calculate PeakSignalNoiseRatio (Peak Signal-to-Noise Ratio).

        Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            prediction (np.ndarray): Images with range [0, 255].
            groundtruth (np.ndarray): Images with range [0, 255].

        Returns:
            np.float64: PeakSignalNoiseRatio result.
        """

        mse_value = ((groundtruth - prediction)**2).mean()
        if mse_value == 0:
            result = float('inf')
        else:
            result = 20. * np.log10(255. / np.sqrt(mse_value))

        return result


# Keep the deprecated metric name as an alias.
# The deprecated Metric names will be removed in 1.0.0!
PSNR = PeakSignalNoiseRatio
