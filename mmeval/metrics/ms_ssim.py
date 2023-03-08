# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from scipy import signal
from typing import Dict, List, Sequence, Tuple

from mmeval.core import BaseMetric
from .utils.image_transforms import reorder_image


class MultiScaleStructureSimilarity(BaseMetric):
    """MS-SSIM (Multi-Scale Structure Similarity) metric.

    Ref:
    This class implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    PGGAN's implementation:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py

    Args:
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Defaults to 'HWC'.
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Defaults to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Defaults to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Defaults to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Defaults to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Defaults to 0.03.
        weights (List[float]): List of weights for each level. Defaults to
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]. Noted that the default
            weights don't sum to 1.0 but do match the paper / matlab code.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import MultiScaleStructureSimilarity as MS_SSIM
        >>> import numpy as np
        >>>
        >>> ms_ssim = MS_SSIM()
        >>> preds = [np.random.randint(0, 255, size=(3, 32, 32)) for _ in range(4)]  # noqa
        >>> ms_ssim(preds)  # doctest: +ELLIPSIS
        {'ms_ssim': ...}
    """

    def __init__(self,
                 input_order: str = 'CHW',
                 max_val: int = 255,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 weights: List[float] = [
                     0.0448, 0.2856, 0.3001, 0.2363, 0.1333
                 ],
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert input_order.upper() in [
            'CHW', 'HWC'
        ], (f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
        self.input_order = input_order

        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        self.weights = np.array(weights)

    def add(self, predictions: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add a bunch of images to calculate metric result.

        Args:
            predictions (Sequence[np.ndarray]): Predictions of the model. The
                number of elements in the Sequence must be divisible by 2, and
                the width and height of each element must be divisible by 2 **
                num_scale (`self.weights.size`). The channel order of each
                element should align with `self.input_order` and the range
                should be [0, 255].
        """

        num_samples = len(predictions)
        assert num_samples % 2 == 0, 'Number of samples must be divided by 2.'

        half1 = [
            reorder_image(pred, self.input_order) for pred in predictions[0::2]
        ]
        half2 = [
            reorder_image(pred, self.input_order) for pred in predictions[1::2]
        ]

        least_size = 2**self.weights.size
        assert all([
            sample.shape[0] % least_size == 0 for sample in half1
        ]), ('The height and width of each sample must be divisible by '
             f'{least_size} (2 ** len(self.weights.size)).')
        assert all([
            sample.shape[0] % least_size == 0 for sample in half2
        ]), ('The height and width of each sample must be divisible by '
             f'{least_size} (2 ** self.weights.size).')

        half1 = np.stack(half1, axis=0).astype(np.uint8)
        half2 = np.stack(half2, axis=0).astype(np.uint8)

        self._results += self.compute_ms_ssim(half1, half2)

    def compute_metric(self, results: List[np.float64]) -> Dict[str, float]:
        """Compute the MS-SSIM metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float64]): A list that consisting the PSNR score.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed PSNR metric.
        """
        return {'ms-ssim': float(np.array(results).mean())}

    def compute_ms_ssim(self, img1: np.array, img2: np.array) -> List[float]:
        """Calculate MS-SSIM (multi-scale structural similarity).

        Args:
            img1 (ndarray): Images with range [0, 255] and order "NHWC".
            img2 (ndarray): Images with range [0, 255] and order "NHWC".

        Returns:
            np.ndarray: MS-SSIM score between `img1` and `img2` of shape (N, ).
        """
        if img1.shape != img2.shape:
            raise RuntimeError(
                'Input images must have the same shape (%s vs. %s).' %
                (img1.shape, img2.shape))
        if img1.ndim != 4:
            raise RuntimeError(
                'Input images must have four dimensions, not %d' % img1.ndim)

        levels = self.weights.size
        im1, im2 = (x.astype(np.float32) for x in [img1, img2])
        mssim = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self._ssim_for_multi_scale(
                im1,
                im2,
                max_val=self.max_val,
                filter_size=self.filter_size,
                filter_sigma=self.filter_sigma,
                k1=self.k1,
                k2=self.k2)
            mssim.append(ssim)
            mcs.append(cs)
            im1, im2 = (self._hox_downsample(x) for x in [im1, im2])

        # Clip to zero. Otherwise we get NaNs.
        mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
        mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

        results = np.prod(
            mcs[:-1, :]**self.weights[:-1, np.newaxis], axis=0) * (
                mssim[-1, :]**self.weights[-1])
        return results.tolist()

    @staticmethod
    def _f_special_gauss(size: int, sigma: float) -> np.ndarray:
        r"""Return a circular symmetric gaussian kernel.

        Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/2504c3f3cb98ca58751610ad61fa1097313152bd/metrics/ms_ssim.py#L25-L36  # noqa

        Args:
            size (int): Size of Gaussian kernel.
            sigma (float): Standard deviation for Gaussian blur kernel.

        Returns:
            np.ndarray: Gaussian kernel.
        """
        radius = size // 2
        offset = 0.0
        start, stop = -radius, radius + 1
        if size % 2 == 0:
            offset = 0.5
            stop -= 1
        x, y = np.mgrid[offset + start:stop,  # type: ignore # noqa
                        offset + start:stop]  # type: ignore # noqa
        assert len(x) == size
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()

    @staticmethod
    def _hox_downsample(img: np.ndarray) -> np.ndarray:
        r"""Downsample images with factor equal to 0.5.

        Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/2504c3f3cb98ca58751610ad61fa1097313152bd/metrics/ms_ssim.py#L110-L111  # noqa

        Args:
            img (np.ndarray): Images with order "NHWC".

        Returns:
            np.ndarray: Downsampled images with order "NHWC".
        """
        return (img[:, 0::2, 0::2, :] + img[:, 1::2, 0::2, :] +
                img[:, 0::2, 1::2, :] + img[:, 1::2, 1::2, :]) * 0.25

    def _ssim_for_multi_scale(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            max_val: int = 255,
            filter_size: int = 11,
            filter_sigma: float = 1.5,
            k1: float = 0.01,
            k2: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate SSIM (structural similarity) and contrast sensitivity.

        Ref:
        Our implementation is based on PGGAN:
        https://github.com/tkarras/progressive_growing_of_gans/blob/2504c3f3cb98ca58751610ad61fa1097313152bd/metrics/ms_ssim.py#L38-L108  # noqa

        Args:
            img1 (np.ndarray): Images with range [0, 255] and order "NHWC".
            img2 (np.ndarray): Images with range [0, 255] and order "NHWC".
            max_val (int): the dynamic range of the images (i.e., the
                difference between the maximum the and minimum allowed
                values). Defaults to 255.
            filter_size (int): Size of blur kernel to use (will be reduced for
                small images). Defaults to 11.
            filter_sigma (float): Standard deviation for Gaussian blur kernel (
                will be reduced for small images). Defaults to 1.5.
            k1 (float): Constant used to maintain stability in the SSIM
                calculation (0.01 in the original paper). Defaults to 0.01.
            k2 (float): Constant used to maintain stability in the SSIM
                calculation (0.03 in the original paper). Defaults to 0.03.

        Returns:
            tuple: Pair containing the mean SSIM and contrast sensitivity
                between `img1` and `img2`.
        """
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        _, height, width, _ = img1.shape

        # Filter size can't be larger than height or width of images.
        size = min(filter_size, height, width)

        # Scale down sigma if a smaller filter size is used.
        sigma = size * filter_sigma / filter_size if filter_size else 0

        if filter_size:
            window = np.reshape(
                self._f_special_gauss(size, sigma), (1, size, size, 1))
            mu1 = signal.fftconvolve(img1, window, mode='valid')
            mu2 = signal.fftconvolve(img2, window, mode='valid')
            sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
            sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
            sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
        else:
            # Empty blur kernel so no need to convolve.
            mu1, mu2 = img1, img2
            sigma11 = img1 * img1
            sigma22 = img2 * img2
            sigma12 = img1 * img2

        mu11 = mu1 * mu1
        mu22 = mu2 * mu2
        mu12 = mu1 * mu2
        sigma11 -= mu11
        sigma22 -= mu22
        sigma12 -= mu12

        # Calculate intermediate values used by both ssim and cs_map.
        c1 = (k1 * max_val)**2
        c2 = (k2 * max_val)**2
        v1 = 2.0 * sigma12 + c2
        v2 = sigma11 + sigma22 + c2
        ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)),
                       axis=(1, 2, 3))  # Return for each image individually.
        cs = np.mean(v1 / v2, axis=(1, 2, 3))
        return ssim, cs
