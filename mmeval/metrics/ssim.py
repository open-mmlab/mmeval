# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from mmeval.core import BaseMetric
from mmeval.utils import try_import
from .utils import img_transform

if TYPE_CHECKING:
    import cv2
else:
    cv2 = try_import('cv2')


class SSIM(BaseMetric):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

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
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # TODO: maybe a better solution for channel order
        self.channel_order = channel_order

        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to

    def add(self, preds: Sequence[np.array], gts: Sequence[np.array]) -> None:   # type: ignore # yapf: disable # noqa: E501
        """Add the SSIM score of the batch to ``self._results``.

        Args:
            preds (Sequence[np.array]): Predictions of the model.
            gts (Sequence[np.array]): The ground truth images.
        """
        for pred, gt in zip(preds, gts):
            assert pred.shape == gt.shape, (
                f'Image shapes are different: {pred.shape}, {gt.shape}.')
            pred = img_transform(
                pred,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=self.channel_order)
            gt = img_transform(
                gt,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=self.channel_order)
            self._results.append(self._compute_ssim(pred, gt))

    def compute_metric(self, results: List[np.float64]) -> Dict[str, float]:
        """Compute the SSIM metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float64]): A list that consisting the SSIM scores.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed SSIM metric.
        """

        return {'ssim': float(np.array(results).mean())}

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> np.float64:
        """Calculate SSIM (structural similarity).

        Ref:
        Image quality assessment: From error visibility to structural
        similarity

        The results are the same as that of the official released MATLAB code
        in https://ece.uwaterloo.ca/~z70wang/research/ssim/.

        For three-channel images, SSIM is calculated for each channel and then
        averaged.

        Args:
            img1 (np.ndarray): Images with range [0, 255] with order 'HWC'.
            img2 (np.ndarray): Images with range [0, 255] with order 'HWC'.

        Returns:
            np.float64: SSIM result.
        """

        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()
