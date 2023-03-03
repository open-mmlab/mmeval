# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import os
from scipy.ndimage import convolve
from scipy.special import gamma
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from mmeval.core import BaseMetric
from .utils import reorder_and_crop


class NaturalImageQualityEvaluator(BaseMetric):
    """Calculate Natural Image Quality Evaluator(NIQE) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Args:
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the NIQE calculation. Defaults to 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Defaults to 'CHW'.
        convert_to (str): Convert the images to other color models. Options are
            'y' and 'gray'. Defaults to 'gray'.
        channel_order (str): The channel order of image. Defaults to 'rgb'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import NaturalImageQualityEvaluator
        >>> import numpy as np
        >>>
        >>> niqe = NaturalImageQualityEvaluator()
        >>> preds = np.random.randint(0, 255, size=(3, 32, 32))
        >>> niqe(preds)  # doctest: +ELLIPSIS
        {'niqe': ...}
    """

    def __init__(self,
                 crop_border: int = 0,
                 input_order: str = 'CHW',
                 convert_to: str = 'gray',
                 channel_order: str = 'rgb',
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if input_order.upper() not in ['CHW', 'HWC']:
            raise ValueError(
                f'Wrong input_order {input_order}. Supported input_orders '
                "are 'HWC' and 'CHW'")
        if convert_to.lower() not in ['y', 'gray']:
            raise ValueError(
                'Only support gray image, '
                "``convert_to`` should be selected from ['y', 'gray']")

        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to
        self.channel_order = channel_order

        # we use the official params estimated from the pristine dataset.
        niqe_pris_params = np.load(
            os.path.join(
                os.path.dirname(__file__),
                '../extra_data/niqe_pris_params.npz'))
        self.mu_pris_param = niqe_pris_params['mu_pris_param']
        self.cov_pris_param = niqe_pris_params['cov_pris_param']
        self.gaussian_window = niqe_pris_params['gaussian_window']

    def add(self, predictions: Sequence[np.ndarray], channel_order: Optional[str] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add NIQE score of batch to ``self._results``

        Args:
            predictions (Sequence[np.ndarray]): Predictions of the model. Each
                prediction should be a 4D (as a video, not batches of images)
                or 3D (as an image) array.
            channel_order (Optional[str]): The channel order of the input
                samples. If not passed, will set as :attr:`self.channel_order`.
                Defaults to None.
        """

        if channel_order is None:
            channel_order = self.channel_order

        for prediction in predictions:

            if len(prediction.shape) <= 3:
                result = self.compute_niqe(prediction, channel_order,
                                           self.mu_pris_param,
                                           self.cov_pris_param,
                                           self.gaussian_window)
            else:
                result_sum = 0
                for i in range(prediction.shape[0]):
                    result_sum += self.compute_niqe(prediction[i],
                                                    channel_order,
                                                    self.mu_pris_param,
                                                    self.cov_pris_param,
                                                    self.gaussian_window)
                result = result_sum / prediction.shape[0]

            self._results.append(result)

    def compute_metric(self, results: List[np.float64]) -> Dict[str, float]:
        """Compute the NaturalImageQualityEvaluator metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[np.float64]): A list that consisting the NIQE score.
                This list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed NIQE metric.
        """

        return {'niqe': float(np.array(results).mean())}

    def compute_niqe(self,
                     prediction: np.ndarray,
                     channel_order: str,
                     mu_pris_param: np.ndarray,
                     cov_pris_param: np.ndarray,
                     gaussian_window: np.ndarray,
                     block_size_h: int = 96,
                     block_size_w: int = 96) -> np.float64:
        """Calculate NIQE (Natural Image Quality Evaluator) metric. We use the
        official params estimated from the pristine dataset. We use the
        recommended block size (96, 96) without overlaps.

        Note that we do not include block overlap height and width, since they
        are always 0 in the official implementation.

        For good performance, it is advisable by the official implementation to
        divide the distorted image into the same size patched as used for the
        construction of multivariate Gaussian model.

        Args:
            prediction (np.ndarray): Input image whose quality to be computed.
                Range [0, 255] with float type.
            channel_order (str): The channel order of image.
            mu_pris_param (np.ndarray): Mean of a pre-defined multivariate
                Gaussian model calculated on the pristine dataset.
            cov_pris_param (np.ndarray): Covariance of a pre-defined
                multivariate Gaussian model calculated on the pristine dataset.
            gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing
                the image.
            block_size_h (int): Height of the blocks in to which image is
                divided. Defaults to 96.
            block_size_w (int): Width of the blocks in to which image is
                divided. Defaults to 96.

        Returns:
            np.float64: NIQE result.
        """

        prediction = reorder_and_crop(
            prediction,
            crop_border=self.crop_border,
            input_order=self.input_order,
            convert_to=self.convert_to,
            channel_order=channel_order)

        prediction = np.squeeze(prediction)

        # round to follow official implementation
        prediction = prediction.round()

        # crop image
        h, w = prediction.shape
        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)
        img = prediction[0:num_block_h * block_size_h,
                         0:num_block_w * block_size_w]

        distparam = []  # dist param is actually the multiscale features
        for scale in (1, 2):  # perform on two scales (1, 2)
            mu = convolve(img, gaussian_window, mode='nearest')

            sigma = np.sqrt(
                np.abs(
                    convolve(np.square(img), gaussian_window, mode='nearest') -
                    np.square(mu)))
            # normalize, as in Eq. 1 in the paper
            img_normalized = (img - mu) / (sigma + 1)

            feat = []
            for idx_w in range(num_block_w):
                for idx_h in range(num_block_h):
                    # process each block
                    block = img_normalized[idx_h * block_size_h //
                                           scale:(idx_h + 1) * block_size_h //
                                           scale, idx_w * block_size_w //
                                           scale:(idx_w + 1) * block_size_w //
                                           scale]
                    feat.append(self.compute_feature(block))

            distparam.append(np.array(feat))

            # matlab-like bicubic downsample with anti-aliasing
            if scale == 1:
                img = self.matlab_resize(
                    img[:, :, np.newaxis] / 255., scale=0.5)[:, :, 0] * 255.

        distparam = np.concatenate(distparam, axis=1)

        # fit a MVG (multivariate Gaussian) model to distorted patch features
        mu_distparam = np.nanmean(distparam, axis=0)
        distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
        cov_distparam = np.cov(distparam_no_nan, rowvar=False)

        # compute niqe quality, Eq. 10 in the paper
        invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
        quality = np.matmul(
            np.matmul((mu_pris_param - mu_distparam), invcov_param),
            np.transpose(mu_pris_param - mu_distparam))

        result = np.squeeze(np.sqrt(quality))
        return result

    def compute_feature(self, block: np.ndarray) -> List:
        """Compute features.

        Args:
            block (np.ndarray): 2D Image block.

        Returns:
            List: Features with length of 18.
        """

        feat = []
        alpha, beta_l, beta_r = self.estimate_aggd_param(block)
        feat.extend([alpha, (beta_l + beta_r) / 2])

        # distortions disturb the fairly regular structure of natural images.
        # This deviation can be captured by analyzing the sample distribution
        # of the products of pairs of adjacent coefficients computed along
        # horizontal, vertical and diagonal orientations.
        shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
        for shift in shifts:
            shifted_block = np.roll(block, shift, axis=(0, 1))
            alpha, beta_l, beta_r = self.estimate_aggd_param(block *
                                                             shifted_block)
            mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
            feat.extend([alpha, mean, beta_l, beta_r])
        return feat

    def estimate_aggd_param(self, block: np.ndarray) -> Tuple:
        """Estimate AGGD (Asymmetric Generalized Gaussian Distribution)
        parameters.

        Args:
            block (np.ndarray): 2D Image block.

        Returns:
            Tuple: alpha(float), beta_l(float) and beta_r(float) for the AGGD
            distribution (Estimating parameters in Equation 7 in the paper).
        """

        block = block.flatten()
        gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
        gam_reciprocal = np.reciprocal(gam)
        r_gam = np.square(gamma(gam_reciprocal * 2)) / (
            gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

        left_std = np.sqrt(np.mean(block[block < 0]**2))
        right_std = np.sqrt(np.mean(block[block > 0]**2))
        gammahat = left_std / right_std
        rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
        rhatnorm = (rhat * (gammahat**3 + 1) *
                    (gammahat + 1)) / ((gammahat**2 + 1)**2)
        array_position = np.argmin((r_gam - rhatnorm)**2)

        alpha = gam[array_position]
        beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
        beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
        return (alpha, beta_l, beta_r)

    def matlab_resize(self, img: np.ndarray, scale: float) -> np.ndarray:
        """Resize an image to the required size.

        Args:
            img (np.ndarray): The original image.
            scale (float): The scale factor of the resize operation.

        Returns:
            np.ndarray: The resized image.
        """

        weights = {}
        indices = {}

        kernel_func = self._cubic
        kernel_width = 4.0

        # compute scale and output_size
        scale = float(scale)
        scale = [scale, scale]
        output_size = self.get_size_from_scale(img.shape, scale)

        # apply cubic interpolation along two dimensions
        order = np.argsort(np.array(scale))
        for k in range(2):
            key = (img.shape[k], output_size[k], scale[k], kernel_func,
                   kernel_width)
            weight, index = self.get_weights_indices(img.shape[k],
                                                     output_size[k], scale[k],
                                                     kernel_func, kernel_width)
            weights[key] = weight
            indices[key] = index

        output = np.copy(img)
        if output.ndim == 2:  # grayscale image
            output = output[:, :, np.newaxis]

        for k in range(2):
            dim = order[k]
            key = (img.shape[dim], output_size[dim], scale[dim], kernel_func,
                   kernel_width)
            output = self.resize_along_dim(output, weights[key], indices[key],
                                           dim)

        return output

    def get_size_from_scale(self, input_size: Tuple,
                            scale_factor: List[float]) -> List[int]:
        """Get the output size given input size and scale factor.

        Args:
            input_size (Tuple): The size of the input image.
            scale_factor (List[float]): The resize factor.

        Returns:
            List[int]: The size of the output image.
        """

        output_shape = [
            int(np.ceil(scale * shape))
            for (scale, shape) in zip(scale_factor, input_size)
        ]

        return output_shape

    def _cubic(self, x: np.ndarray) -> np.ndarray:
        """Cubic function.

        Args:
            x (np.ndarray): The distance from the center position.

        Returns:
            np.ndarray: The weight corresponding to a particular distance.
        """

        x = np.array(x, dtype=np.float32)
        x_abs = np.abs(x)
        x_abs_sq = x_abs**2
        x_abs_cu = x_abs_sq * x_abs

        # if |x| <= 1: y = 1.5|x|^3 - 2.5|x|^2 + 1
        # if 1 < |x| <= 2: -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2
        f = (1.5 * x_abs_cu - 2.5 * x_abs_sq + 1) * (x_abs <= 1) + (
            -0.5 * x_abs_cu + 2.5 * x_abs_sq - 4 * x_abs + 2) * ((1 < x_abs) &
                                                                 (x_abs <= 2))

        return f

    def get_weights_indices(
            self, input_length: int, output_length: int, scale: float,
            kernel: Callable,
            kernel_width: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get weights and indices for interpolation.

        Args:
            input_length (int): Length of the input sequence.
            output_length (int): Length of the output sequence.
            scale (float): Scale factor.
            kernel (Callable): The kernel used for resizing.
            kernel_width (float): The width of the kernel.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: The weights and the
            indices for interpolation.
        """

        if scale < 1:  # modified kernel for antialiasing

            def h(x):
                return scale * kernel(scale * x)

            kernel_width = 1.0 * kernel_width / scale
        else:
            h = kernel
            kernel_width = kernel_width

        # coordinates of output
        x = np.arange(1, output_length + 1).astype(np.float32)

        # coordinates of input
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = np.floor(u - kernel_width / 2)  # leftmost pixel
        p = int(np.ceil(kernel_width)) + 2  # maximum number of pixels

        # indices of input pixels
        ind = left[:, np.newaxis, ...] + np.arange(p)
        indices = ind.astype(np.int32)

        # weights of input pixels
        weights = h(u[:, np.newaxis, ...] - indices - 1)

        weights = weights / np.sum(weights, axis=1)[:, np.newaxis, ...]

        # remove all-zero columns
        aux = np.concatenate(
            (np.arange(input_length), np.arange(input_length - 1, -1,
                                                step=-1))).astype(np.int32)
        indices = aux[np.mod(indices, aux.size)]
        ind2store = np.nonzero(np.any(weights, axis=0))
        weights = weights[:, ind2store]
        indices = indices[:, ind2store]

        return weights, indices

    def resize_along_dim(self, img_in: np.ndarray, weights: np.ndarray,
                         indices: np.ndarray, dim: int) -> np.ndarray:
        """Resize along a specific dimension.

        Args:
            img_in (np.ndarray): The input image.
            weights (np.ndarray): The weights used for interpolation, computed
                from [get_weights_indices].
            indices (np.ndarray): The indices used for interpolation, computed
                from [get_weights_indices].
            dim (int): Which dimension to undergo interpolation.

        Returns:
            np.ndarray: Interpolated (along one dimension) image.
        """

        img_in = img_in.astype(np.float32)
        w_shape = weights.shape
        output_shape = list(img_in.shape)
        output_shape[dim] = w_shape[0]
        img_out = np.zeros(output_shape)

        if dim == 0:
            for i in range(w_shape[0]):
                w = weights[i, :][np.newaxis, ...]
                ind = indices[i, :]
                img_slice = img_in[ind, :]
                img_out[i] = np.sum(
                    np.squeeze(img_slice, axis=0) * w.T, axis=0)
        elif dim == 1:
            for i in range(w_shape[0]):
                w = weights[i, :][:, :, np.newaxis]
                ind = indices[i, :]
                img_slice = img_in[:, ind]
                img_out[:, i] = np.sum(
                    np.squeeze(img_slice, axis=1) * w.T, axis=1)

        if img_in.dtype == np.uint8:
            img_out = np.clip(img_out, 0, 255)
            return np.around(img_out).astype(np.uint8)
        else:
            return img_out
