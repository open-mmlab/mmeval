# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from typing import Dict, List, Optional, Sequence

from mmeval.core import BaseMetric
from .utils import reorder_and_crop


class StructuralSimilarity(BaseMetric):
    """Calculate StructuralSimilarity (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, StructuralSimilarity is calculated for each
    channel and then averaged.

    Args:
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PeakSignalNoiseRatio calculation.
            Defaults to 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Defaults to 'HWC'.
        convert_to (str, optional): Whether to convert the images to other
            color models. If None, the images are not altered. When computing
            for 'Y', the images are assumed to be in BGR order. Options are
            'Y' and None. Defaults to None.
        channel_order (str): The channel order of image. Choices are 'rgb' and
            'bgr'. Defaults to 'rgb'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import StructuralSimilarity as SSIM
        >>> import numpy as np
        >>>
        >>> ssim = SSIM(input_order='CHW', convert_to='Y', channel_order='rgb')
        >>> gts = np.random.randint(0, 255, size=(3, 32, 32))
        >>> preds = np.random.randint(0, 255, size=(3, 32, 32))
        >>> ssim(preds, gts)  # doctest: +ELLIPSIS
        {'ssim': ...}

    Calculate StructuralSimilarity between 2 single channel images:

        >>> img1 = np.ones((32, 32)) * 2
        >>> img2 = np.ones((32, 32))
        >>> SSIM.compute_ssim(img1, img2)
        0.913062377743969
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

    def add(self, predictions: Sequence[np.ndarray], groundtruths: Sequence[np.ndarray], channel_order: Optional[str] = None) -> None:   # type: ignore # yapf: disable # noqa: E501
        """Add the StructuralSimilarity score of the batch to
        ``self._results``.

        For three-channel images, StructuralSimilarity is calculated for
        each channel and then averaged.

        Args:
            predictions (Sequence[np.ndarray]): Predictions of the model.
            groundtruths (Sequence[np.ndarray]): The ground truth images.
            channel_order (Optional[str]): The channel order of the input
                samples. If not passed, will set as :attr:`self.channel_order`.
                Defaults to None.
        """
        channel_order = self.channel_order \
            if channel_order is None else channel_order

        for pred, gt in zip(predictions, groundtruths):
            assert pred.shape == gt.shape, (
                f'Image shapes are different: {pred.shape}, {gt.shape}.')
            pred = reorder_and_crop(
                pred,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=channel_order)
            gt = reorder_and_crop(
                gt,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to,
                channel_order=channel_order)

            if len(pred.shape) == 3:
                pred = np.expand_dims(pred, axis=0)
                gt = np.expand_dims(gt, axis=0)
            _ssim_score = []
            for i in range(pred.shape[0]):
                for j in range(pred.shape[3]):
                    _ssim_score.append(
                        self.compute_ssim(pred[i][..., j], gt[i][..., j]))
            self._results.append(np.array(_ssim_score).mean())

    def compute_metric(self, results: List[np.float64]) -> Dict[str, float]:
        """Compute the StructuralSimilarity metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float64]): A list that consisting the
                StructuralSimilarity scores. This list has already been
                synced across all ranks.

        Returns:
            Dict[str, float]: The computed StructuralSimilarity metric.
        """

        return {'ssim': float(np.array(results).mean())}

    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> np.float64:
        """Calculate StructuralSimilarity (structural similarity) between two
        single channel image.

        Ref:
        Image quality assessment: From error visibility to structural
        similarity

        The results are the same as that of the official released MATLAB code
        in https://ece.uwaterloo.ca/~z70wang/research/ssim/.

        Args:
            img1 (np.ndarray): Single channels Images with range [0, 255].
            img2 (np.ndarray): Single channels Images with range [0, 255].

        Returns:
            np.float64: StructuralSimilarity result.
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


# Keep the deprecated metric name as an alias.
# The deprecated Metric names will be removed in 1.0.0!
SSIM = StructuralSimilarity
