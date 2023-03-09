# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import (TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple,
                    Union, overload)

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.metrics.utils import MultiLabelMixin, format_data
from mmeval.utils import try_import

if TYPE_CHECKING:
    import oneflow
    import oneflow as flow
    import torch
else:
    torch = try_import('torch')
    flow = try_import('oneflow')

NUMPY_IMPL_HINTS = Tuple[Union[np.ndarray, np.number], Union[np.ndarray,
                                                             np.number]]
TORCH_IMPL_HINTS = Tuple['torch.Tensor', 'torch.Tensor']
ONEFLOW_IMPL_HINTS = Tuple['oneflow.Tensor', 'oneflow.Tensor']
BUILTIN_IMPL_HINTS = Tuple[Union[int, Sequence[Union[int, float]]],
                           Union[int, Sequence[int]]]


def _average_precision_torch(preds: 'torch.Tensor', labels: 'torch.Tensor',
                             average) -> 'torch.Tensor':
    r"""Calculate the average precision for torch.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        preds (torch.Tensor): The model prediction with shape
            ``(N, num_classes)``.
        labels (torch.Tensor): The target of predictions with shape
            ``(N, num_classes)``.

    Returns:
        torch.Tensor: average precision result.
    """
    # sort examples along classes
    sorted_pred_inds = torch.argsort(preds, dim=0, descending=True)
    sorted_target = torch.gather(labels, 0, sorted_pred_inds)

    # get indexes when gt_true is positive
    pos_inds = sorted_target == 1

    # Calculate cumulative tp case numbers
    tps = torch.cumsum(pos_inds, 0)
    total_pos = tps[-1].clone()  # the last of tensor may change later

    # Calculate cumulative tp&fp(pred_poss) case numbers
    pred_pos_nums = torch.arange(1, len(sorted_target) + 1).to(preds.device)

    tps[torch.logical_not(pos_inds)] = 0
    precision = tps / pred_pos_nums.unsqueeze(-1).float()  # divide along rows
    ap = torch.sum(precision, 0) / torch.clamp(total_pos, min=1)

    if average == 'macro':
        return ap.mean() * 100.0
    else:
        return ap * 100


def _average_precision_oneflow(preds: 'oneflow.Tensor',
                               labels: 'oneflow.Tensor',
                               average) -> 'oneflow.Tensor':
    r"""Calculate the average precision for oneflow.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        preds (oneflow.Tensor): The model prediction with shape
            ``(N, num_classes)``.
        labels (oneflow.Tensor): The target of predictions with shape
            ``(N, num_classes)``.

    Returns:
        oneflow.Tensor: average precision result.
    """
    # sort examples along classes
    sorted_pred_inds = flow.argsort(preds, dim=0, descending=True)
    sorted_target = flow.gather(labels, 0, sorted_pred_inds)

    # get indexes when gt_true is positive
    pos_inds = sorted_target == 1

    # Calculate cumulative tp case numbers
    tps = flow.cumsum(pos_inds, 0)
    total_pos = tps[-1].clone()  # the last of tensor may change later

    # Calculate cumulative tp&fp(pred_poss) case numbers
    pred_pos_nums = flow.arange(1, len(sorted_target) + 1).to(preds.device)

    tps[flow.logical_not(pos_inds)] = 0
    precision = tps / pred_pos_nums.unsqueeze(-1).float()  # divide along rows
    ap = flow.sum(precision, 0) / flow.clamp(total_pos, min=1)

    if average == 'macro':
        return ap.mean() * 100.0
    else:
        return ap * 100


def _average_precision(preds: np.ndarray, labels: np.ndarray,
                       average) -> np.ndarray:
    r"""Calculate the average precision for numpy.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        preds (np.ndarray): The model prediction with shape
            ``(N, num_classes)``.
        labels (np.ndarray): The target of predictions with shape
            ``(N, num_classes)``.

    Returns:
        np.ndarray: average precision result.
    """
    # sort examples along classes
    sorted_pred_inds = np.argsort(-preds, axis=0)
    sorted_target = np.take_along_axis(labels, sorted_pred_inds, axis=0)

    # get indexes when gt_true is positive
    pos_inds = sorted_target == 1

    # Calculate cumulative tp case numbers
    tps = np.cumsum(pos_inds, 0)
    total_pos = tps[-1].copy()  # the last of tensor may change later

    # Calculate cumulative tp&fp(pred_poss) case numbers
    pred_pos_nums = np.arange(1, len(sorted_target) + 1)

    tps[np.logical_not(pos_inds)] = 0
    precision = np.divide(
        tps, np.expand_dims(pred_pos_nums, -1), dtype=np.float32)
    ap = np.divide(
        np.sum(precision, 0), np.clip(total_pos, 1, np.inf), dtype=np.float32)

    if average == 'macro':
        return ap.mean() * 100.0
    else:
        return ap * 100


class AveragePrecision(MultiLabelMixin, BaseMetric):
    """Calculate the average precision with respect of classes.

    Args:
        average (str, optional): The average method. It supports two modes:

            - `"macro"`: Calculate metrics for each category, and calculate
                the mean value over all categories.
            - `None`: Return scores of all categories.

        Defaults to "macro".

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    Examples:

        >>> from mmeval import AveragePrecision
        >>> average_precision = AveragePrecision()

    Use Builtin implementation with label-format labels:

        >>> preds = [[0.9, 0.8, 0.3, 0.2],
                     [0.1, 0.2, 0.2, 0.1],
                     [0.7, 0.5, 0.9, 0.3],
                     [0.8, 0.1, 0.1, 0.2]]
        >>> labels = [[0, 1], [1], [2], [0]]
        >>> average_precision(preds, labels)
        {'mAP': 70.833..}

    Use Builtin implementation with one-hot encoding labels:

        >>> preds = [[0.9, 0.8, 0.3, 0.2],
                      [0.1, 0.2, 0.2, 0.1],
                      [0.7, 0.5, 0.9, 0.3],
                      [0.8, 0.1, 0.1, 0.2]]
        >>> labels = [[1, 1, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0]]
        >>> average_precision(preds, labels)
        {'mAP': 70.833..}

    Use NumPy implementation with label-format labels:

        >>> import numpy as np
        >>> preds = np.array([[0.9, 0.8, 0.3, 0.2],
                              [0.1, 0.2, 0.2, 0.1],
                              [0.7, 0.5, 0.9, 0.3],
                              [0.8, 0.1, 0.1, 0.2]])
        >>> labels = [np.array([0, 1]), np.array([1]), np.array([2]), np.array([0])] # noqa
        >>> average_precision(preds, labels)
        {'mAP': 70.833..}

    Use PyTorch implementation with one-hot encoding labels::

        >>> import torch
        >>> preds = torch.Tensor([[0.9, 0.8, 0.3, 0.2],
                                  [0.1, 0.2, 0.2, 0.1],
                                  [0.7, 0.5, 0.9, 0.3],
                                  [0.8, 0.1, 0.1, 0.2]])
        >>> labels = torch.Tensor([[1, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [1, 0, 0, 0]])
        >>> average_precision(preds, labels)
        {'mAP': 70.833..}

    Computing with `None` average mode:

        >>> preds = np.array([[0.9, 0.8, 0.3, 0.2],
                              [0.1, 0.2, 0.2, 0.1],
                              [0.7, 0.5, 0.9, 0.3],
                              [0.8, 0.1, 0.1, 0.2]])
        >>> labels = [np.array([0, 1]), np.array([1]), np.array([2]), np.array([0])] # noqa
        >>> average_precision = AveragePrecision(average=None)
        >>> average_precision(preds, labels)
        {'AP_classwise': [100.0, 83.33, 100.00, 0.0]}  # rounded results

    Accumulate batch:

        >>> for i in range(10):
        ...     preds = torch.randint(0, 4, size=(100, 10))
        ...     labels = torch.randint(0, 4, size=(100, ))
        ...     average_precision.add(preds, labels)
        >>> average_precision.compute()  # doctest: +SKIP
    """

    def __init__(self, average: Optional[str] = 'macro', **kwargs) -> None:
        super().__init__(**kwargs)
        average_options = ['macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specify from {average_options}.'
        self.average = average
        self.pred_is_onehot = False

    def add(self, preds: Sequence, labels: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            preds (Sequence): Predictions from the model. It should
                be scores of every class (N, C).
            labels (Sequence): The ground truth labels. It should be (N, ) for
                label-format, or (N, C) for one-hot encoding.
        """
        for pred, target in zip(preds, labels):
            self._results.append((pred, target))

    def _format_metric_results(self, ap):
        """Format the given metric results into a dictionary.

        Args:
            ap (list): Results of average precision for each categories
                or the single marco result.

        Returns:
            dict: The formatted dictionary.
        """
        result_metrics = dict()

        if self.average is None:
            _result = ap[0].tolist()
            result_metrics['AP_classwise'] = [round(_r, 4) for _r in _result]
        else:
            result_metrics['mAP'] = round(ap[0].item(), 4)

        return result_metrics

    @overload
    @dispatch
    def _compute_metric(self, preds: Sequence['torch.Tensor'],
                        labels: Sequence['torch.Tensor']) -> List[List]:
        """A PyTorch implementation that computes the metric."""

        preds = torch.stack(preds)
        num_classes = preds.shape[1]
        labels = format_data(labels, num_classes, self._label_is_onehot).long()

        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        return _average_precision_torch(preds, labels, self.average)

    @overload  # type: ignore
    @dispatch
    def _compute_metric(  # type: ignore
            self, preds: Sequence['oneflow.Tensor'],
            labels: Sequence['oneflow.Tensor']) -> List[List]:
        """A OneFlow implementation that computes the metric."""

        preds = flow.stack(preds)
        num_classes = preds.shape[1]
        labels = format_data(labels, num_classes, self._label_is_onehot).long()

        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        return _average_precision_oneflow(preds, labels, self.average)

    @overload
    @dispatch
    def _compute_metric(
            self, preds: Sequence[Union[int, Sequence[Union[int, float]]]],
            labels: Sequence[Union[int, Sequence[int]]]) -> List[List]:
        """A Builtin implementation that computes the metric."""

        return self._compute_metric([np.array(pred) for pred in preds],
                                    [np.array(target) for target in labels])

    @dispatch
    def _compute_metric(
            self, preds: Sequence[Union[np.ndarray, np.number]],
            labels: Sequence[Union[np.ndarray, np.number]]) -> List[List]:
        """A NumPy implementation that computes the metric."""

        preds = np.stack(preds)
        num_classes = preds.shape[1]
        labels = format_data(labels, num_classes,
                             self._label_is_onehot).astype(np.int64)

        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        return _average_precision(preds, labels, self.average)

    def compute_metric(
        self, results: List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS,
                                  ONEFLOW_IMPL_HINTS, BUILTIN_IMPL_HINTS]]
    ) -> Dict[str, float]:
        """Compute the metric.

        Currently, there are 3 implementations of this method: NumPy and
        PyTorch and OneFlow. Which implementation to use is determined by the
        type of the calling parameters. e.g. `numpy.ndarray` or
        `torch.Tensor`, `oneflow.Tensor`.

        This method would be invoked in `BaseMetric.compute` after distributed
        synchronization.

        Args:
            results (List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS,
            ONEFLOW_IMPL_HINTS]]): A list of tuples that consisting the
            prediction and label. This list has already been synced across
            all ranks.

        Returns:
            Dict[str, float]: The computed metric.
        """
        preds = [res[0] for res in results]
        labels = [res[1] for res in results]
        assert self._pred_is_onehot is False, '`self._pred_is_onehot` should' \
            f'be `False` for {self.__class__.__name__}, because scores are' \
            'necessary for compute the metric.'
        metric_results = self._compute_metric(preds, labels)
        return self._format_metric_results(metric_results)
