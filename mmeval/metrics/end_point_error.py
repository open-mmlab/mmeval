# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, overload

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.utils import try_import

if TYPE_CHECKING:
    import oneflow
    import oneflow as flow
    import torch
else:
    torch = try_import('torch')
    flow = try_import('oneflow')


class EndPointError(BaseMetric):
    """EndPointError evaluation metric.

    EndPointError is a widely used evaluation metric for optical flow
    estimation.

    This metric supports 3 kinds of inputs, i.e. `numpy.ndarray` and
    `torch.Tensor`, `oneflow.Tensor`, and the implementation for the
    calculation depends on the inputs type.

    Args:
        **kwargs: Keyword arguments passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import EndPointError
        >>> epe = EndPointError()

    Use NumPy implementation:

        >>> import numpy as np
        >>> predictions = np.array(
        ...     [[[10., 5.], [0.1, 3.]],
        ...     [[3., 15.2], [2.4, 4.5]]])
        >>> labels = np.array(
        ...     [[[10.1, 4.8], [10, 3.]],
        ...     [[6., 10.2], [2.0, 4.1]]])
        >>> valid_masks = np.array([[1., 1.], [1., 0.]])
        >>> epe(predictions, labels, valid_masks)
        {'EPE': 5.318186230865093}

    Use PyTorch implementation:

        >>> import torch
        >>> predictions = torch.Tensor(
        ...     [[[10., 5.], [0.1, 3.]],
        ...     [[3., 15.2], [2.4, 4.5]]])
        >>> labels = torch.Tensor(
        ...     [[[10.1, 4.8], [10, 3.]],
        ...     [[6., 10.2], [2.0, 4.1]]])
        >>> valid_masks = torch.Tensor([[1., 1.], [1., 0.]])
        >>> epe(predictions, labels, valid_masks)
        {'EPE': 5.3181863}

    Accumulate batch:

        >>> for i in range(10):
        ...     predictions = torch.randn(10, 10, 2)
        ...     labels = torch.randn(10, 10, 2)
        ...     epe.add(predictions, labels)
        >>> epe.compute()  # doctest: +SKIP
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add(self, predictions: Sequence, labels: Sequence, valid_masks: Optional[Sequence] = None) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of predictions and labels.

        Args:
            predictions (Sequence): Predicted sequence of flow map with
                shape (H, W, 2).
            labels (Sequence): The ground truth sequence of flow map with
                shape (H, W, 2).
            valid_masks (Sequence): The Sequence of valid mask for labels with
                shape (H, W). If it is None, this function will automatically
                generate a map filled with 1. Defaults to None.
        """

        for idx, (prediction, label) in enumerate(zip(predictions, labels)):
            assert prediction.shape == label.shape, 'The shape of ' \
                '`prediction` and `label` should be the same, but got: ' \
                f'{prediction.shape} and {label.shape}'

            assert prediction.shape[-1] == 2, 'The last dimension of ' \
                f'`prediction` should be 2, but got: {prediction.shape[-1]}'

            if valid_masks is not None:
                valid_mask = valid_masks[idx]
            else:
                valid_mask = None
            epe, num_valid_pixel = self.end_point_error_map(
                prediction, label, valid_mask)
            self._results.append((epe, num_valid_pixel))

    @overload  # type: ignore
    @dispatch
    def end_point_error_map(
            self,
            prediction: np.ndarray,
            label: np.ndarray,
            valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """Calculate end point error map.

        Args:
            prediction (np.ndarray): Prediction with shape (H, W, 2).
            label (np.ndarray): Ground truth with shape (H, W, 2).
            valid_mask (np.ndarray, optional): Valid mask with shape (H, W).

        Returns:
            Tuple: The mean of end point error and the numbers of valid labels.
        """
        if valid_mask is None:
            valid_mask = np.ones_like(prediction[..., 0])
        epe_map = np.sqrt(np.sum((prediction - label)**2, axis=-1))
        val = valid_mask.reshape(-1) >= 0.5
        epe = epe_map.reshape(-1)[val]
        return epe.mean(keepdims=True), int(val.sum())

    @overload
    @dispatch
    def end_point_error_map(  # type: ignore
        self,
        prediction: 'torch.Tensor',
        label: 'torch.Tensor',
        valid_mask: Optional['torch.Tensor'] = None
    ) -> Tuple[np.ndarray, int]:  # yapf: disable
        """Calculate end point error map.

        Args:
            prediction (torch.Tensor): Prediction with shape (H, W, 2).
            label (torch.Tensor): Ground truth with shape (H, W, 2).
            valid_mask (torch.Tensor, optional): Valid mask with shape (H, W).

        Returns:
            Tuple: The mean of end point error and the numbers of valid labels.
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(prediction[..., 0])
        epe_map = torch.sqrt(torch.sum((prediction - label)**2, dim=-1))
        val = valid_mask.reshape(-1) >= 0.5
        epe = epe_map.reshape(-1)[val]
        return epe.mean().cpu().numpy(), int(val.sum())

    @dispatch
    def end_point_error_map(
        self,
        prediction: 'oneflow.Tensor',
        label: 'oneflow.Tensor',
        valid_mask: Optional['oneflow.Tensor'] = None
    ) -> Tuple[np.ndarray, int]:
        """Calculate end point error map.

        Args:
            prediction (oneflow.Tensor): Prediction with shape (H, W, 2).
            label (oneflow.Tensor): Ground truth with shape (H, W, 2).
            valid_mask (oneflow.Tensor, optional): Valid mask with
                shape (H, W).

        Returns:
            Tuple: The mean of end point error and the numbers of valid labels.
        """
        if valid_mask is None:
            valid_mask = flow.ones_like(prediction[..., 0])
        epe_map = flow.sqrt(flow.sum((prediction - label)**2, dim=-1))
        val = valid_mask.reshape(-1) >= 0.5
        epe = epe_map.reshape(-1)[val]
        return epe.mean().cpu().numpy(), int(val.sum())

    def compute_metric(self, results: List[Tuple[np.ndarray, int]]) -> dict:
        """Compute the EndPointError metric.

        This method would be invoked in `BaseMetric.compute` after distributed
        synchronization.

        Args:
            results (List[np.ndarray]): This list has already been synced
                across all ranks. This is a list of `np.ndarray`, which is
                the end point error map between the prediction and the label.

        Returns:
            Dict: The computed metric, with following key:

            - EPE, the mean end point error of all pairs.
        """
        epe_overall = sum(res[0] * res[1] for res in results)
        valid_pixels = sum(res[1] for res in results)
        metric_results = {'EPE': epe_overall / valid_pixels}
        return metric_results
