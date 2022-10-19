# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Optional, Sequence

from mmeval.core import BaseMetric
from .utils import img_transform


class SNR(BaseMetric):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.
        channel_order (str): The channel order of image. Default: 'rgb'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 input_order: str = 'CHW',
                 channel_order: str = 'rgb',
                 convert_to: Optional[str] = None,
                 crop_border: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        # TODO: maybe a better solution for channel order
        self.channel_order = channel_order

        self.input_order = input_order
        self.convert_to = convert_to
        self.crop_border = crop_border

    def add(self, preds: Sequence[np.array], gts: Sequence[np.array]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add SNR score of batch to ``self._results``

        Args:
            preds (Sequence[np.array]): Predictions of the model.
            gts (Sequence[np.array]): The ground truth images.
        """
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
    def _compute_snr(gt: np.array, pred: np.array) -> np.flaot64:
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            gt (np.array): Images with range [0, 255].
            pred (np.array): Images with range [0, 255].

        Returns:
            np.float64: SNR result.
        """
        signal = ((gt)**2).mean()
        noise = ((gt - pred)**2).mean()

        result = 10. * np.log10(signal / noise)

        return result
