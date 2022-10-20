# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Optional, Sequence

from mmeval.core import BaseMetric
from .utils import img_transform


class SNR(BaseMetric):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.
        channel_order (str): The channel order of image. Default: 'rgb'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 crop_border: int = 0,
                 input_order: str = 'CHW',
                 convert_to: Optional[str] = None,
                 channel_order: str = 'rgb',
                 **kwargs):
        super().__init__(**kwargs)

        assert input_order.upper() in [
            'CHW', 'HWC'
        ], (f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
        self.input_order = input_order
        self.crop_border = crop_border

        if convert_to is not None:
            assert convert_to.upper() == 'Y', (
                'Wrong color model. Supported values are "Y" and None.')
            assert channel_order.upper() in [
                'BGR', 'RGB'
            ], ('Only support `rgb2y` and `bgr2y`, but the channel_order '
                f'is {channel_order}')
        self.channel_order = channel_order
        self.convert_to = convert_to

    def add(self, preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], channel_order: Optional[str] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add SNR score of batch to ``self._results``

        Args:
            preds (Sequence[np.ndarray]): Predictions of the model.
            gts (Sequence[np.ndarray]): The ground truth images.
            channel_order (Optional[str]): The channel order of the input
                samples. If not passed, will set as :attr:`self.channel_order`.
                Defaults to None.
        """
        channel_order = self.channel_order \
            if channel_order is None else channel_order
        for pred, gt in zip(preds, gts):
            assert gt.shape == pred.shape, (
                f'Image shapes are different: {gt.shape}, {pred.shape}.')
            gt = img_transform(
                gt,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=self.channel_order)
            pred = img_transform(
                pred,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=self.channel_order)

            self._results.append(self._compute_snr(gt, pred))

    def compute_metric(self, results: List[np.float64]) -> Dict[str, float]:
        """Compute the SNR metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float64]): A list that consisting the SNR score.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed SNR metric.
        """

        return {'snr': float(np.array(results).mean())}

    @staticmethod
    def _compute_snr(gt: np.ndarray, pred: np.ndarray) -> np.float64:
        """Calculate SNR (Signal-to-Noise Ratio).

        Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio

        Args:
            gt (np.ndarray): Images with range [0, 255].
            pred (np.ndarray): Images with range [0, 255].

        Returns:
            np.float64: SNR result.
        """
        signal = ((gt)**2).mean()
        noise = ((gt - pred)**2).mean()

        result = 10. * np.log10(signal / noise)

        return result
