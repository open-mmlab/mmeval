# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union, overload

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch

try:
    import torch
except ImportError:
    torch = None

NUMPY_IMPL_HINTS = Tuple[Union[np.ndarray, np.int64], Union[np.ndarray,
                                                            np.int64]]
TORCH_IMPL_HINTS = Tuple['torch.Tensor', 'torch.Tensor']
BUILTIN_IMPL_HINTS = Tuple[Union[int, Sequence[Union[int, float]]],
                           Union[int, Sequence[int]]]


def label_to_onehot(label: Union[np.ndarray, 'torch.Tensor'],
                    num_classes: int) -> Union[np.ndarray, 'torch.Tensor']:
    """Convert the label-format input to one-hot.

    Args:
        label (torch.Tensor): The label-format input. The format
            of item must be label-format.
        num_classes (int): The number of classes.

    Return:
        torch.Tensor: The converted results.
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


def _precision_recall_f1_support(pred_positive, gt_positive, average):
    """Calculate base classification task metrics, such as  precision, recall,
    f1_score, support. Inputs of `pred_positive` and `gt_positive` should be
    both `torch.tensor` with `torch.int64` dtype or `numpy.ndarray` with
    `numpy.int64` dtype. And should be both with shape of (M, N):

    - M: Number of samples.
    - N: Number of classes.
    """
    average_options = ['micro', 'macro', None]
    assert average in average_options, 'Invalid `average` argument, ' \
        f'please specify from {average_options}.'

    class_correct = (pred_positive & gt_positive)
    if average == 'micro':
        tp_sum = class_correct.sum()
        pred_sum = pred_positive.sum()
        gt_sum = gt_positive.sum()
    else:
        tp_sum = class_correct.sum(0)
        pred_sum = pred_positive.sum(0)
        gt_sum = gt_positive.sum(0)

    # in case torch is not supported
    if torch and isinstance(pred_sum, torch.Tensor):
        # use torch with torch.Tensor
        precision = tp_sum / torch.clamp(pred_sum, min=1).double() * 100
        recall = tp_sum / torch.clamp(gt_sum, min=1).double() * 100
        f1_score = 2 * precision * recall / torch.clamp(
            precision + recall, min=torch.finfo(torch.float32).eps)
    else:
        # use numpy with numpy.ndarray
        precision = tp_sum / np.clip(pred_sum, 1, np.inf) * 100
        recall = tp_sum / np.clip(gt_sum, 1, np.inf) * 100
        f1_score = 2 * precision * recall / np.clip(precision + recall,
                                                    np.finfo(np.float32).eps,
                                                    np.inf)

    # skip process float results by numpy
    if average in ['macro', 'micro'] and not isinstance(precision, float):
        precision = precision.mean(0)
        recall = recall.mean(0)
        f1_score = f1_score.mean(0)
        support = gt_sum.sum(0)
    else:
        support = gt_sum
    return precision, recall, f1_score, support


class MultiLabelMetric(BaseMetric):
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
        3. Each sample has raw indices.

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

        >>> labels = np.array([0, 1, 2, 3])
        >>> preds = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.3, 0.4, 0.2, 0.1],
            [0.0, 0.0, 0.1, 0.9]])
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

        self._pred_is_rawindex = False
        self._label_is_rawindex = False

    @property
    def pred_is_rawindex(self):
        """Whether use raw index for predictions."""
        warnings.warn(
            '`pred_is_rawindex` only works for corner case when '
            'num_classes=2 to distinguish one-hot indices or raw indices')
        return self._pred_is_rawindex

    @pred_is_rawindex.setter
    def pred_is_rawindex(self, is_rawindex):
        """Set a flag of whether use raw index for predictions."""
        warnings.warn(
            '`pred_is_rawindex` only works for corner case when '
            'num_classes=2 to distinguish one-hot indices or raw indices')
        self._pred_is_rawindex = is_rawindex

    @property
    def label_is_rawindex(self):
        """Whether use raw index for labels."""
        warnings.warn(
            '`label_is_rawindex` only works for corner case when '
            'num_classes=2 to distinguish one-hot indices or raw indices')
        return self._label_is_rawindex

    @label_is_rawindex.setter
    def label_is_rawindex(self, is_rawindex):
        """Set a flag of whether use raw index for labels."""
        warnings.warn(
            '`label_is_rawindex` only works for corner case when '
            'num_classes=2 to distinguish one-hot indices or raw indices')
        self._label_is_rawindex = is_rawindex

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

    def _get_label(self, labels, is_rawindex, stack_func):
        """Get labels with shape of (N, num_classes) from different inputs of
        scores/ one-hot indices/ raw indices."""
        try:
            # try stack scores or one-hot indices directly
            _labels = stack_func(labels)
            # all assertions below is to find labels that are
            # raw indices which should be caught in exception
            # to convert to one-hot indices.
            #
            # 1. all raw indices has only 1 dims
            assert _labels.ndim == 2
            # 2. all raw indices has the same dims
            assert _labels.shape[1] == self.num_classes
            # 3. all raw indices has the same dims as num_classes
            # then max indices should greater than 1 for num_classes > 2
            assert _labels.max() <= 1
            # 4. corner case, num_classes=2, then one-hot indices
            # and raw indices are undistinguishable, for instance:
            #   [[0, 1], [0, 1]] can be one-hot indices of 2 positives
            #   or raw indices of 4 positives.
            # Extra induction is needed.
            if self.num_classes == 2:
                warnings.warn(
                    'Ambiguous labels detected, reckoned as scores'
                    ' or one-hot indices as defaults. Please set '
                    '`self.pred_is_rawindex` and `self.label_is_rawindex`'
                    ' if use raw indices to compute metrics.')
                assert not is_rawindex
        # Error corresponds to np, torch, assert respectively
        except (ValueError, RuntimeError, AssertionError):
            # convert raw indices to one-hot indices
            _labels = stack_func(
                [label_to_onehot(label, self.num_classes) for label in labels])
        return _labels

    @overload
    @dispatch
    def _compute_metric(self, predictions: Sequence['torch.Tensor'],
                        labels: Sequence['torch.Tensor']) -> List[List]:
        """A PyTorch implementation that computes the metric."""

        preds = self._get_label(predictions, self._pred_is_rawindex,
                                torch.stack)
        gts = self._get_label(labels, self._label_is_rawindex,
                              torch.stack).long()

        # cannot be raised in current implementation because
        # `and` method will guarantee the equal length.
        # However length check should remain somewhere.
        assert preds.shape[0] == gts.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({gts.shape[0]}).'

        if self.thr is not None:
            # a label is predicted positive if larger than self.
            # work for index as well
            pos_inds = (preds >= self.thr).long()
        else:
            # top-k gts will be predicted positive for any example
            _, topk_indices = preds.topk(self.topk)
            pos_inds = torch.zeros_like(preds).scatter_(1, topk_indices, 1)
            pos_inds = pos_inds.long()

        return _precision_recall_f1_support(pos_inds, gts, self.average)

    @overload
    @dispatch
    def _compute_metric(
            self, predictions: Sequence[Union[int, Sequence[Union[int,
                                                                  float]]]],
            labels: Sequence[Union[int, Sequence[int]]]) -> List[List]:
        """A Builtin implementation that computes the metric."""

        return self._compute_metric([np.array(pred) for pred in predictions],
                                    [np.array(gt) for gt in labels])

    @dispatch
    def _compute_metric(
            self, predictions: Sequence[Union[np.ndarray, np.int64]],
            labels: Sequence[Union[np.ndarray, np.int64]]) -> List[List]:
        """A NumPy implementation that computes the metric."""

        preds = self._get_label(predictions, self._pred_is_rawindex, np.stack)
        gts = self._get_label(labels, self._label_is_rawindex,
                              np.stack).astype(np.int64)

        # cannot be raised in current implementation because
        # `and` method will guarantee the equal length.
        # However length check should remain somewhere.
        assert preds.shape[0] == gts.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({gts.shape[0]}).'

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

        return _precision_recall_f1_support(pos_inds, gts, self.average)

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
        predictions = [res[0] for res in results]
        labels = [res[1] for res in results]
        metric_results = self._compute_metric(predictions, labels)
        return self._format_metric_results(metric_results)
