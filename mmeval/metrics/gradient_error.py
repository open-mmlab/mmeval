# Copyright (c) OpenMMLab. All rights reserved.Dict
import cv2
import numpy as np
from typing import Dict, List, Sequence

from mmeval.core import BaseMetric


def gaussian(x: np.ndarray, sigma: float):
    """Gaussian function.

    Args:
        x (np.ndarray): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Returns:
        np.ndarray or scalar: Gaussian value of `x`.
    """

    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def dgaussian(x: np.ndarray, sigma: float):
    """Derivative of Gaussian function.

    Args:
        x (np.ndarray): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Returns:
        np.ndarray or scalar: Gradient of gaussian of `x`.
    """

    return -x * gaussian(x, sigma) / sigma**2


def gauss_filter(sigma: float, epsilon: float = 1e-2):
    """Gaussian Filter.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
        epsilon (float): Small value used when calculating kernel size.
            Default to 1e-2.

    Returns:
        tuple(np.ndarray, np.ndarray): Gaussian filter along x and y axis.
    """

    half_size = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
    size = int(2 * half_size + 1)

    # create filter in x axis
    filter_x = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            filter_x[i, j] = gaussian(i - half_size, sigma) * dgaussian(
                j - half_size, sigma)

    # normalize filter
    norm = np.sqrt((filter_x**2).sum())
    filter_x = filter_x / norm
    filter_y = np.transpose(filter_x)

    return filter_x, filter_y


def gauss_gradient(img: np.ndarray, sigma: float):
    """Gaussian gradient.

    Reference: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/
    submissions/8060/versions/2/previews/gaussgradient/gaussgradient.m/
    index.html

    Args:
        img (np.ndarray): Input image.
        sigma (float): Standard deviation of the gaussian kernel.

    Returns:
        np.ndarray: Gaussian gradient of input `img`.
    """

    filter_x, filter_y = gauss_filter(sigma)
    img_filtered_x = cv2.filter2D(
        img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
    img_filtered_y = cv2.filter2D(
        img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(img_filtered_x**2 + img_filtered_y**2)


class GradientError(BaseMetric):
    """Gradient error for evaluating alpha matte prediction.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
            Defaults to 1.4 .
        norm_const (int): Divide the result to reduce its magnitude.
            Defaults to 1000.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Note:
        The current implementation assumes the image / alpha / trimap
        a numpy array with pixel values ranging from 0 to 255.

        The pred_alpha should be masked by trimap before passing
        into this metric.

        The trimap is the most commonly used prior knowledge. As the
        name implies, trimap is a ternary graph and each pixel
        takes one of {0, 128, 255}, representing the foreground, the
        unknown and the background respectively.

    Examples:

        >>> from mmeval import GradientError
        >>> import numpy as np
        >>>
        >>> gradient_error = GradientError()
        >>> np.random.seed(0)
        >>> pred_alpha = np.random.randn(32, 32).astype('uint8')
        >>> gt_alpha = np.ones((32, 32), dtype=np.uint8) * 255
        >>> trimap = np.zeros((32, 32), dtype=np.uint8)
        >>> trimap[:16, :16] = 128
        >>> trimap[16:, 16:] = 255
        >>> gradient_error(pred_alpha, gt_alpha, trimap)  # doctest: +ELLIPSIS
        {'gradient_error': ...}
    """

    def __init__(self,
                 sigma: float = 1.4,
                 norm_const: int = 1000,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sigma = sigma
        self.norm_const = norm_const

    def add(self, pred_alphas: Sequence[np.ndarray], gt_alphas: Sequence[np.ndarray], trimaps: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add GradientError score of batch to ``self._results``

        Args:
            pred_alphas (Sequence[np.ndarray]): Predict the probability
                that pixels belong to the foreground.
            gt_alphas (Sequence[np.ndarray]): Probability that the actual
                pixel belongs to the foreground.
            trimaps (Sequence[np.ndarray]): Broadly speaking, the trimap
                consists of foreground and unknown region.
        """

        for pred_alpha, gt_alpha, trimap in zip(pred_alphas, gt_alphas,
                                                trimaps):
            assert pred_alpha.shape == gt_alpha.shape, 'The shape of ' \
                '`pred_alpha` and `gt_alpha` should be the same, but got: ' \
                f'{pred_alpha.shape} and {gt_alpha.shape}'

            gt_alpha_normed = np.zeros_like(gt_alpha)
            pred_alpha_normed = np.zeros_like(pred_alpha)

            cv2.normalize(gt_alpha, gt_alpha_normed, 1.0, 0.0, cv2.NORM_MINMAX)
            cv2.normalize(pred_alpha, pred_alpha_normed, 1.0, 0.0,
                          cv2.NORM_MINMAX)

            gt_alpha_grad = gauss_gradient(gt_alpha_normed, self.sigma)
            pred_alpha_grad = gauss_gradient(pred_alpha_normed, self.sigma)
            # this is the sum over n samples
            grad_loss = ((gt_alpha_grad - pred_alpha_grad)**2 *
                         (trimap == 128)).sum()

            # divide by self.norm_const to reduce the magnitude of the result
            grad_loss /= self.norm_const

            self._results.append(grad_loss)

    def compute_metric(self, results: List) -> Dict[str, float]:
        """Compute the GradientError metric.

        Args:
            results (List): A list that consisting the GradientError score.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed GradientError metric.
            The keys are the names of the metrics,
            and the values are corresponding results.
        """

        return {'gradient_error': float(np.array(results).mean())}
