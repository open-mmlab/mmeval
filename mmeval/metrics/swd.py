# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from mmeval.core import BaseMetric
from mmeval.utils import try_import

if TYPE_CHECKING:
    from scipy import ndimage
else:
    ndimage = try_import('scipy.ndimage')


class SlicedWassersteinDistance(BaseMetric):
    """SWD (Sliced Wasserstein distance) metric. We calculate the SWD of two
    sets of images in the following way. In every 'feed', we obtain the
    Laplacian pyramids of every images and extract patches from the Laplacian
    pyramids as descriptors. In 'summary', we normalize these descriptors along
    channel, and reshape them so that we can use these descriptors to represent
    the distribution of real/fake images. And we can calculate the sliced
    Wasserstein distance of the real and fake descriptors as the SWD of the
    real and fake images. Note that, as with the official implementation, we
    multiply the result by 10 to prevent the value from being too small and to
    facilitate comparison.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py # noqa

    Args:
        resolution (int): Resolution of the input images.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Defaults to 'HWC'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import SlicedWassersteinDistance as SWD
        >>> import numpy as np
        >>>
        >>> swd = SWD(resolution=64)
        >>> preds = [np.random.randint(0, 255, size=(3, 64, 64)) for _ in range(4)]  # noqa
        >>> gts = [np.random.randint(0, 255, size=(3, 64, 64)) for _ in range(4)]  # noqa
        >>> swd(preds, gts)  # doctest: +ELLIPSIS
        {'SWD/64': ..., 'SWD/32': ..., 'SWD/16': ..., 'SWD/8': ..., 'SWD/avg': ...}
    """

    def __init__(self,
                 resolution: int,
                 input_order: str = 'CHW',
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert input_order.upper() in [
            'CHW', 'HWC'
        ], (f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
        self.input_order = input_order

        self.nhood_size = 7  # height and width of the extracted patches
        self.nhoods_per_image = 128  # number of extracted patches per image
        self.dir_repeats = 4  # times of sampling directions
        self.dirs_per_repeat = 128  # number of directions per sampling

        self.resolution = resolution
        self._resolutions: List[Any] = []
        while resolution >= 16 and len(self._resolutions) < 4:
            self._resolutions.append(resolution)
            resolution //= 2
        self.n_pyramids = len(self._resolutions)
        self.gaussian_k = self.get_gaussian_kernel()

    @staticmethod
    def get_gaussian_kernel() -> np.ndarray:
        """Get Gaussian kernel.

        Returns:
            np.ndarray: Gaussian kernel.
        """
        kernel = np.array(
            [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
             [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], np.float32) / 256.0
        return kernel.reshape(1, 1, 5, 5)

    def add(self, predictions: Sequence[np.ndarray], groundtruths: Sequence[np.ndarray]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add processed feature of batch to ``self._results``

        Args:
            predictions (Sequence[np.ndarray]): Predictions of the model.
                The channel order of each element in Sequence should align with
                `self.input_order` and the range should be [0, 255].
            groundtruths (Sequence[np.ndarray]): The ground truth images.
                The channel order of each element in Sequence should align with
                `self.input_order` and the range should be [0, 255].
        """
        if self.input_order == 'HWC':
            predictions = [pred.transpose(2, 0, 1) for pred in predictions]
            groundtruths = [gt.transpose(2, 0, 1) for gt in groundtruths]
        preds = np.stack(predictions, axis=0)
        gts = np.stack(groundtruths, axis=0)

        # convert to [-1, 1]
        preds, gts = (preds - 127.5) / 127.5, (gts - 127.5) / 127.5

        # prepare feature pyramid
        fake_pyramid = self.laplacian_pyramid(preds, self.n_pyramids - 1,
                                              self.gaussian_k)
        real_pyramid = self.laplacian_pyramid(gts, self.n_pyramids - 1,
                                              self.gaussian_k)

        # init list to save fake desp and real desp
        if self._results == []:
            self._results.append([[] for _ in self._resolutions])
            self._results.append([[] for _ in self._resolutions])
        for lod, level in enumerate(fake_pyramid):
            desc = self.get_descriptors_for_minibatch(level, self.nhood_size,
                                                      self.nhoods_per_image)
            self._results[0][lod].append(desc)

        for lod, level in enumerate(real_pyramid):
            desc = self.get_descriptors_for_minibatch(level, self.nhood_size,
                                                      self.nhoods_per_image)
            self._results[1][lod].append(desc)

    def compute_metric(self,
                       results: List[List[np.ndarray]]) -> Dict[str, float]:
        """Compute the SWD metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (List[List[np.ndarray]]): A list of feature of different
                resolution extracted from real and fake images. This list has
                already been synced across all ranks.

        Returns: Dict[str, float]: The computed SWD metric.
        """
        results_fake, results_real = results
        fake_descs = [self.finalize_descriptors(d) for d in results_fake]
        real_descs = [self.finalize_descriptors(d) for d in results_real]

        distance = [
            self.compute_swd(dreal, dfake, self.dir_repeats,
                             self.dirs_per_repeat)
            for dreal, dfake in zip(real_descs, fake_descs)
        ]
        del real_descs
        del fake_descs
        # multiply by 10^3 refers to https://github.com/tkarras/progressive_growing_of_gans/blob/2504c3f3cb98ca58751610ad61fa1097313152bd/metrics/sliced_wasserstein.py#L132  # noqa
        distance = [d * 1e3 for d in distance]
        result = distance + [np.mean(distance)]
        return {
            f'SWD/{resolution}': d
            for resolution, d in zip(self._resolutions + ['avg'], result)
        }

    def compute_swd(self, distribution_a: np.array, distribution_b: np.array,
                    dir_repeats: int, dirs_per_repeat: int) -> np.ndarray:
        """Calculate SWD (SlicedWassersteinDistance).

        Args:
            distribution_a (np.ndarray): Distribution to compute sliced
                wasserstein distance.
            distribution_b (np.ndarray): Distribution to compute sliced
                wasserstein distance.
            dir_repeats (int): The number of projection times.
            dirs_per_repeat (int): The number of directions per projection.

        Returns:
            np.ndarray: SWD between `distribution_a` and `distribution_b`.
        """
        results = []
        for _ in range(dir_repeats):
            # (descriptor_component, direction)
            dirs = np.random.randn(distribution_a.shape[1], dirs_per_repeat)
            # normalize descriptor components for each direction
            dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True))
            dirs = dirs.astype(np.float32)
            # (neighborhood, direction)
            projA = np.matmul(distribution_a, dirs)
            projB = np.matmul(distribution_b, dirs)
            # sort neighborhood projections for each direction
            projA = np.sort(projA, axis=0)
            projB = np.sort(projB, axis=0)
            # pointwise wasserstein distances
            dists = np.abs(projA - projB)
            # average over neighborhoods and directions
            results.append(np.mean(dists))

        return np.mean(results)

    def laplacian_pyramid(self, original: np.ndarray, n_pyramids: int,
                          gaussian_k: np.ndarray) -> List[np.ndarray]:
        """Calculate Laplacian pyramid.

        Ref: https://github.com/koshian2/swd-pytorch/blob/master/swd.py

        Args:
            original (np.ndarray): Batch of Images with range [0, 1] and order
                "NCHW".
            n_pyramids (int): Levels of pyramids minus one.
            gaussian_k (np.ndarray): Gaussian kernel with shape (1, 1, 5, 5).

        Return:
            list[np.ndarray]. Laplacian pyramids of original.
        """
        # create gaussian pyramid
        pyramids = self.gaussian_pyramid(original, n_pyramids, gaussian_k)

        # pyramid up - diff
        laplacian = []
        for i in range(len(pyramids) - 1):
            diff = pyramids[i] - self.get_pyramid_layer(
                pyramids[i + 1], gaussian_k, 'up')
            laplacian.append(diff)
        # Add last gaussian pyramid
        laplacian.append(pyramids[len(pyramids) - 1])
        return laplacian

    def gaussian_pyramid(self, original: np.ndarray, n_pyramids: int,
                         gaussian_k: np.ndarray) -> List[np.ndarray]:
        """Get a group of gaussian pyramid.

        Args:
            original (np.ndarray): The input image.
            n_pyramids (int): The number of pyramids.
            gaussian_k (np.ndarray): The gaussian kernel.

        Returns:
            List[np.ndarray]: The list of output of gaussian pyramid.
        """
        x = original
        # pyramid down
        pyramids = [original]
        for _ in range(n_pyramids):
            x = self.get_pyramid_layer(x, gaussian_k)
            pyramids.append(x)
        return pyramids

    @staticmethod
    def get_pyramid_layer(image: np.ndarray,
                          gaussian_k: np.ndarray,
                          direction: str = 'down') -> np.ndarray:
        """Get the pyramid layer.

        Args:
            image (np.ndarray): Input image.
            gaussian_k (np.ndarray): Gaussian kernel.
            direction (str, optional): The direction of pyramid. Defaults to
                'down'.

        Returns:
            np.ndarray: The output of the pyramid.
        """
        shape = image.shape
        if direction == 'up':
            # nearest interpolation with scale_factor = 2
            res = np.zeros((shape[0], shape[1], shape[2] * 2, shape[3] * 2),
                           image.dtype)
            res[:, :, ::2, ::2] = image
            res[:, :, 1::2, 1::2] = image
            res[:, :, 1::2, ::2] = image
            res[:, :, ::2, 1::2] = image
            return ndimage.convolve(res, gaussian_k, mode='constant')
        else:
            return ndimage.convolve(
                image, gaussian_k, mode='constant')[:, :, ::2, ::2]

    def get_descriptors_for_minibatch(self, minibatch: np.ndarray,
                                      nhood_size: int,
                                      nhoods_per_image: int) -> np.ndarray:
        r"""Get descriptors of one level of pyramids.

        Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

        Args:
            minibatch (np.ndarray): Pyramids of one level with order "NCHW".
            nhood_size (int): Pixel neighborhood size.
            nhoods_per_image (int): The number of descriptors per image.

        Return:
            np.ndarray: Descriptors of images from one level batch.
        """
        shape = minibatch.shape  # (minibatch, channel, height, width)
        assert len(shape) == 4 and shape[1] == 3
        N = nhoods_per_image * shape[0]
        H = nhood_size // 2
        nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H + 1, -H:H + 1]
        img = nhood // nhoods_per_image
        x = x + np.random.randint(H, shape[3] - H, size=(N, 1, 1, 1))
        y = y + np.random.randint(H, shape[2] - H, size=(N, 1, 1, 1))
        idx = ((img * shape[1] + chan) * shape[2] + y) * shape[3] + x
        return minibatch.flat[idx]

    @staticmethod
    def finalize_descriptors(desc: List[np.ndarray]) -> np.ndarray:
        r"""Normalize and reshape descriptors.

        Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

        Args:
            desc (List[np.ndarray]): List of descriptors of one level.

        Return:
            np.ndarray: Descriptors after normalized along channel and flattened.
        """
        desc = np.concatenate(desc, axis=0)
        desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
        desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
        desc = desc.reshape(desc.shape[0], -1)
        return desc
