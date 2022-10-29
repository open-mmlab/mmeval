# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import warnings
from typing import (TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple,
                    Union, overload)

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.utils import try_import
from .single_label import _precision_recall_f1_support

if TYPE_CHECKING:
    import torch
else:
    torch = try_import('torch')

NUMPY_IMPL_HINTS = Tuple[Union[np.ndarray, np.number], Union[np.ndarray,
                                                             np.number]]
TORCH_IMPL_HINTS = Tuple['torch.Tensor', 'torch.Tensor']
BUILTIN_IMPL_HINTS = Tuple[Union[int, Sequence[Union[int, float]]],
                           Union[int, Sequence[int]]]


def label_to_onehot(label: Union[np.ndarray, 'torch.Tensor'],
                    num_classes: int) -> Union[np.ndarray, 'torch.Tensor']:
    """Convert the label-format input to one-hot encodings.

    Args:
        label (torch.Tensor or np.ndarray): The label-format input.
            The format of item must be label-format.
        num_classes (int): The number of classes.

    Return:
        torch.Tensor or np.ndarray: The converted one-hot encodings.
    """
    if torch and isinstance(label, torch.Tensor):
        label = label.long()
        onehot = label.new_zeros((num_classes, ))
    else:
        label = label.astype(np.int64)
        onehot = np.zeros((num_classes, ), dtype=np.int64)
    assert label.max().item() < num_classes, \
        'Max index is out of `num_classes` {num_classes}'
    assert label.min().item() >= 0
    onehot[label] = 1
    return onehot


def format_data(data: Union[Sequence[Union[np.ndarray, 'torch.Tensor']],
                            np.ndarray, 'torch.Tensor'],
                num_classes: int,
                is_onehot: bool = False) -> Union[np.ndarray, 'torch.Tensor']:
    """Format data from different inputs such as prediction scores, label-
    format data and one-hot encodings into the same output shape of `(N,
    num_classes)`.

    Args:
        data (Union[Sequence[np.ndarray, 'torch.Tensor'], np.ndarray,
            'torch.Tensor']): The input data of prediction or labels.
        num_classes (int): The number of classes.
        is_onehot (bool): Whether the data is one-hot encodings.

    Return:
        torch.Tensor or np.ndarray: One-hot encodings or predict scores.
    """
    if torch and isinstance(data[0], torch.Tensor):
        stack_func = torch.stack
    elif isinstance(data[0], (np.ndarray, np.number)):
        stack_func = np.stack
    else:
        raise NotImplementedError(f'Data type of {type(data[0])}'
                                  'is not supported.')

    try:
        # try stack scores or one-hot indices directly
        formated_data = stack_func(data)
        # all assertions below is to find labels that are
        # raw indices which should be caught in exception
        # to convert to one-hot indices.
        #
        # 1. all raw indices has only 1 dims
        assert formated_data.ndim == 2
        # 2. all raw indices has the same dims
        assert formated_data.shape[1] == num_classes
        # 3. all raw indices has the same dims as num_classes
        # then max indices should greater than 1 for num_classes > 2
        assert formated_data.max() <= 1
        # 4. corner case, num_classes=2, then one-hot indices
        # and raw indices are undistinguishable, for instance:
        #   [[0, 1], [0, 1]] can be one-hot indices of 2 positives
        #   or raw indices of 4 positives.
        # Extra induction is needed.
        if num_classes == 2:
            warnings.warn('Ambiguous data detected, reckoned as scores'
                          ' or label-format data as defaults. Please set '
                          'parms related to `is_onehot` if use one-hot '
                          'encoding data to compute metrics.')
            assert is_onehot
    # Error corresponds to np, torch, stack_func respectively
    except (ValueError, RuntimeError, AssertionError):
        # convert label-format inputs to one-hot encodings
        formated_data = stack_func(
            [label_to_onehot(sample, num_classes) for sample in data])
    return formated_data


class MultiLabelMixin:
    """A Mixin for Multilabel Metrics to clarify whether the input is one-hot
    encodings or label-format inputs for corner case with minimal user
    awareness."""

    def __init__(self, *args, **kwargs) -> None:
        # pass arguments for multiple inheritances
        super().__init__(*args, **kwargs)  # type: ignore
        self._pred_is_onehot = False
        self._label_is_onehot = False

    @property
    def pred_is_onehot(self) -> bool:
        """Whether prediction is one-hot encodings.

        Only works for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        return self._pred_is_onehot

    @pred_is_onehot.setter
    def pred_is_onehot(self, is_onehot: bool):
        """Set a flag of whether prediction is one-hot encodings.

        Only works for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        self._pred_is_onehot = is_onehot

    @property
    def label_is_onehot(self) -> bool:
        """Whether label is one-hot encodings.

        Only works for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        return self._label_is_onehot

    @label_is_onehot.setter
    def label_is_onehot(self, is_onehot: bool):
        """Set a flag of whether label is one-hot encodings.

        Only works for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        self._label_is_onehot = is_onehot


class MultiLabelMetric(MultiLabelMixin, BaseMetric):
    """A collection of metrics for multi-label multi-class classification task
    based on confusion matrix.

    It includes precision, recall, f1-score and support.

    Args:
        num_classes (int): Number of classes. Needed for different inputs
            as extra check.
        thr (float, optional): Predictions with scores under the thresholds
            are considered as negative. Defaults to None.
        topk (int, optional): Predictions with the k-th highest scores are
            considered as positive. Defaults to None.
        items (Sequence[str]): The detailed metric items to evaluate. Here is
            the available options:

            - `"precision"`: The ratio tp / (tp + fp) where tp is the
                number of true positives and fp the number of false
                positives.
            - `"recall"`: The ratio tp / (tp + fn) where tp is the number
                of true positives and fn the number of false negatives.
            - `"f1-score"`: The f1-score is the harmonic mean of the
                precision and recall.
            - `"support"`: The total number of positive of each category
                in the target.

            Defaults to ('precision', 'recall', 'f1-score').
        average (str | None): The average method. It supports three average
            modes:

            - `"macro"`: Calculate metrics for each category, and calculate
                the mean value over all categories.
            - `"micro"`: Calculate metrics globally by counting the total
                true positives, false negatives and false positives.
            - `None`: Return scores of all categories.

            Defaults to "macro".

    .. note::
        MultiLabelMetric supports different kinds of inputs. Such as:
        1. Each sample has scores for every classes. (Only for predictions)
        2. Each sample has one-hot indices for every classes.
        3. Each sample has label-format indices.

    Examples:

        >>> from mmeval import MultiLabelMetric
        >>> multi_lable_metic = MultiLabelMetric(num_classes=4)

    Use Builtin implementation with raw indices:

        >>> preds = [[0], [1], [2], [0, 3]]
        >>> labels = [[0], [1, 2], [2], [3]]
        >>> multi_lable_metic(preds, labels)
        {'precision': 87.5, 'recall': 87.5, 'f1-score': 83.33}

    Use Builtin implementation with one-hot indices:

        >>> preds = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [1, 0, 0, 1]]
        >>> labels = [[1, 0, 0, 0],
                     [0, 1, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
        >>> multi_lable_metic(preds, labels)
        {'precision': 87.5, 'recall': 87.5, 'f1-score': 83.33}

    Use Builtin implementation with scores:

        >>> preds = [[0.9, 0.1, 0.2, 0.3],
                      [0.1, 0.8, 0.1, 0.1],
                      [0.4, 0.3, 0.7, 0.1],
                      [0.8, 0.1, 0.1, 0.9]]
        >>> labels = [[1, 0, 0, 0],
                     [0, 1, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
        >>> multi_lable_metic(preds, labels)
        {'precision': 87.5, 'recall': 87.5, 'f1-score': 83.33}

    Use NumPy implementation with raw indices:

        >>> import numpy as np
        >>> preds = [np.array([0]), np.array([1, 2]), np.array([2]), np.array([3])] # noqa
        >>> labels = [np.array([0]), np.array([1]), np.array([2]), np.array([0, 3])] # noqa
        >>> multi_lable_metic(preds, labels)
        {'precision': 87.5, 'recall': 87.5, 'f1-score': 83.33}

    Use PyTorch implementation:

        >>> import torch
        >>> preds = [torch.tensor([0]), torch.tensor([1, 2]), torch.tensor([2]), torch.tensor([3])] # noqa
        >>> labels = [torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), torch.tensor([0, 3])] # noqa
        >>> multi_lable_metic(preds, labels)
        {'precision': 87.5, 'recall': 87.5, 'f1-score': 83.33}

    Computing with `micro` average mode with `topk=2`:

        >>> preds = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.3, 0.4, 0.2, 0.1],
            [0.0, 0.0, 0.1, 0.9]])
        >>> labels = np.array([0, 1, 2, 3])
        >>> multi_lable_metic = MultiLabelMetric(4, average='micro', topk=2)
        >>> multi_lable_metic(preds, labels)
        {'precision_top2_micro': 37.5, 'recall_top2_micro': 75.0, 'f1-score_top2_micro': 50.0} # noqa

    Accumulate batch:

        >>> for i in range(10):
        ...     labels = torch.randint(0, 4, size=(100, ))
        ...     predicts = torch.randint(0, 4, size=(100, ))
        ...     multi_lable_metic.add(predicts, labels)
        >>> multi_lable_metic.compute()  # doctest: +SKIP
    """

    def __init__(self,
                 num_classes: int,
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 average: Optional[str] = 'macro',
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if thr is None and topk is None:
            thr = 0.5
            warnings.warn('Neither thr nor k is given, set thr as 0.5 by '
                          'default.')
        elif thr is not None and topk is not None:
            warnings.warn('Both thr and topk are given, '
                          'use threshold in favor of top-k.')

        self.thr = thr
        self.topk = topk

        for item in items:
            assert item in ['precision', 'recall', 'f1-score', 'support'], \
                f'The metric {item} is not supported by `MultiLabelMetric`,' \
                ' please specify from "precision", "recall", "f1-score" and ' \
                '"support".'
        self.items = tuple(items)

        average_options = ['micro', 'macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specify from {average_options}.'
        self.average = average
        self.num_classes = num_classes

    def add(self, predictions: Sequence, labels: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            predictions (Sequence): Predictions from the model. It can be
                labels (N, ), or scores of every class (N, C).
            labels (Sequence): The ground truth labels. It should be (N, ).
        """
        for pred, label in zip(predictions, labels):
            self._results.append((pred, label))

    def _format_metric_results(self, results: List) -> Dict:
        """Format the given metric results into a dictionary.

        Args:
            results (list): Results of precision, recall, f1 and support.

        Returns:
            dict: The formatted dictionary.
        """
        metrics = {}

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics

        if self.thr:
            suffix = '' if self.thr == 0.5 else f'_thr-{self.thr:.2f}'
            for k, v in pack_results(*results).items():
                metrics[k + suffix] = v
        else:
            for k, v in pack_results(*results).items():
                metrics[k + f'_top{self.topk}'] = v

        result_metrics = dict()
        for k, v in metrics.items():

            if self.average is None:
                result_metrics[k + '_classwise'] = v.tolist()
            elif self.average == 'micro':
                result_metrics[k + f'_{self.average}'] = v.item()
            else:
                result_metrics[k] = v.item()

        return result_metrics

    @overload
    @dispatch
    def _compute_metric(self, predictions: Sequence['torch.Tensor'],
                        labels: Sequence['torch.Tensor']) -> List:
        """A PyTorch implementation that computes the metric."""

        preds = format_data(predictions, self.num_classes,
                            self._pred_is_onehot)
        labels = format_data(labels, self.num_classes,
                             self._label_is_onehot).long()

        # cannot be raised in current implementation because
        # `and` method will guarantee the equal length.
        # However length check should remain somewhere.
        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        if self.thr is not None:
            # a label is predicted positive if larger than self.
            # work for index as well
            pos_inds = (preds >= self.thr).long()
        else:
            # top-k labels will be predicted positive for any example
            _, topk_indices = preds.topk(self.topk)
            pos_inds = torch.zeros_like(preds).scatter_(1, topk_indices, 1)
            pos_inds = pos_inds.long()

        return _precision_recall_f1_support(  # type: ignore
            pos_inds, labels, self.average)

    @overload
    @dispatch
    def _compute_metric(self, preds: Sequence[Union[int,
                                                    Sequence[Union[int,
                                                                   float]]]],
                        labels: Sequence[Union[int, Sequence[int]]]) -> List:
        """A Builtin implementation that computes the metric."""

        return self._compute_metric([np.array(pred) for pred in preds],
                                    [np.array(target) for target in labels])

    @dispatch
    def _compute_metric(
            self, preds: Sequence[Union[np.ndarray, np.number]],
            labels: Sequence[Union[np.ndarray, np.number]]) -> List:
        """A NumPy implementation that computes the metric."""

        preds = format_data(preds, self.num_classes, self._pred_is_onehot)
        labels = format_data(labels, self.num_classes,
                             self._label_is_onehot).astype(np.int64)

        # cannot be raised in current implementation because
        # `and` method will guarantee the equal length.
        # However length check should remain somewhere.
        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        if self.thr is not None:
            # a label is predicted positive if larger than self.
            # work for index as well
            pos_inds = (preds >= self.thr).astype(np.int64)
        else:
            # top-k labels will be predicted positive for any example
            topk_indices = np.argpartition(
                preds, -self.topk, axis=-1)[:, -self.topk:]  # type: ignore
            pos_inds = np.zeros(preds.shape, dtype=np.int64)
            np.put_along_axis(pos_inds, topk_indices, 1, axis=1)

        return _precision_recall_f1_support(  # type: ignore
            pos_inds, labels, self.average)

    def compute_metric(
        self, results: List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS,
                                  BUILTIN_IMPL_HINTS]]
    ) -> Dict[str, float]:
        """Compute the metric.

        Currently, there are 2 implementations of this method: NumPy and
        PyTorch. Which implementation to use is determined by the type of the
        calling parameters. e.g. `numpy.ndarray` or `torch.Tensor`.
        This method would be invoked in `BaseMetric.compute` after distributed
        synchronization.
        Args:
            results (List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS]]): A list
                of tuples that consisting the prediction and label. This list
                has already been synced across all ranks.
        Returns:
            Dict[str, float]: The computed metric.
        """
        preds = [res[0] for res in results]
        labels = [res[1] for res in results]
        metric_results = self._compute_metric(preds, labels)
        return self._format_metric_results(metric_results)


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

    def add(self, preds: Sequence, labels: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            preds (Sequence): Predictions from the model. It should
                be scores of every class (N, C).
            labels (Sequence): The ground truth labels. It should be (N, ).
        """
        for pred, target in zip(preds, labels):
            self._results.append((pred, target))

    def _format_metric_results(self, ap):
        """Format the given metric results into a dictionary.

        Args:
            results (list): Results of precision, recall, f1 and support.

        Returns:
            dict: The formatted dictionary.
        """
        result_metrics = dict()

        if self.average is None:
            result_metrics['AP_classwise'] = ap[0].tolist()
        else:
            result_metrics['mAP'] = ap[0].item()

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
                                  BUILTIN_IMPL_HINTS]]
    ) -> Dict[str, float]:
        """Compute the metric.

        Currently, there are 2 implementations of this method: NumPy and
        PyTorch. Which implementation to use is determined by the type of the
        calling parameters. e.g. `numpy.ndarray` or `torch.Tensor`.
        This method would be invoked in `BaseMetric.compute` after distributed
        synchronization.
        Args:
            results (List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS]]): A list
                of tuples that consisting the prediction and label. This list
                has already been synced across all ranks.
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
