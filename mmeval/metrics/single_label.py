# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple,
                    Union, overload)

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.utils import try_import

if TYPE_CHECKING:
    import torch
    import torch.nn.functional as F
else:
    torch = try_import('torch')
    F = try_import('torch.nn.functional')

NUMPY_IMPL_HINTS = Tuple[Union[np.ndarray, np.number], np.number]
TORCH_IMPL_HINTS = Tuple['torch.Tensor', 'torch.Tensor']
BUILTIN_IMPL_HINTS = Tuple[Union[int, Sequence[Union[int, float]]],
                           Union[int, Sequence[int]]]


def _precision_recall_f1_support(pred_positive: Union[np.ndarray,
                                                      'torch.Tensor'],
                                 gt_positive: Union[np.ndarray,
                                                    'torch.Tensor'],
                                 average: Optional[str]) -> Tuple:
    """Calculate base classification task metrics, such as precision, recall,
    f1_score, support.

    Args:
        pred_positive (Union[np.ndarray, 'torch.Tensor']): A tensor or
            np.ndarray that indicates the one-hot mapping of positive
            labels in prediction.
        gt_positive (Union[np.ndarray, 'torch.Tensor']): A tensor or
            np.ndarray that indicates the one-hot mapping of positive
            labels in ground truth.
            of tuples that consisting the prediction and label. This list
            has already been synced across all ranks.
        average (str, optional): The average method. If None, the scores
            for each class are returned. And it supports two average modes:

            - `"macro"`: Calculate metrics for each category, and calculate
              the mean value over all categories.
            - `"micro"`: Calculate metrics globally by counting the total
              true positives, false negatives and false positives.

    Returns:
        Tuple: The results of precision, recall, f1_score, and support
        respectively, and the data type depends on the inputs and the
        average type.

    Notes:
        Inputs of `pred_positive` and `gt_positive` should be both
        `torch.tensor` with `torch.int64` dtype or `numpy.ndarray`
        with `numpy.int64` dtype. And should be both with shape of (M, N):
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
        precision = tp_sum / torch.clamp(pred_sum, min=1).float() * 100
        recall = tp_sum / torch.clamp(gt_sum, min=1).float() * 100
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


class SingleLabelMetric(BaseMetric):
    """A collection of metrics for single-label multi-class classification task
    based on confusion matrix.

    It includes precision, recall, f1-score, and support. Comparing with
    :class:`Accuracy`, these metrics don't support topk, but supports
    various average modes.

    Args:
        thrs (Sequence[float | None] | float | None): Predictions with scores
            under the thresholds are considered negative. None means no
            thresholds. Defaults to 0.
        items (Sequence[str]): The detailed metric items to evaluate. Here is
            the available options:

            - `"precision"`: The ratio tp / (tp + fp) where tp is the
              number of true positives and fp the number of false
              positives.
            - `"recall"`: The ratio tp / (tp + fn) where tp is the number
              of true positives and fn the number of false negatives.
            - `"f1-score"`: The f1-score is the harmonic mean of the
              precision and recall.
            - `"support"`: The total number of occurrences of each category
              in the target.

            Defaults to ('precision', 'recall', 'f1-score').
        average (str, optional): The average method. If None, the scores
            for each class are returned. And it supports two average modes:

            - `"macro"`: Calculate metrics for each category, and calculate
              the mean value over all categories.
            - `"micro"`: Calculate metrics globally by counting the total
              true positives, false negatives and false positives.

            Defaults to "macro".
        num_classes (int, optional): Number of classes, only need for predictions
            without scores. Defaults to None.
        **kwargs: Keyword arguments passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import SingleLabelMetric
        >>> single_lable_metic = SingleLabelMetric(num_classes=4)

    Use NumPy implementation:

        >>> import numpy as np
        >>> preds = np.asarray([0, 2, 1, 3])
        >>> labels = np.asarray([0, 1, 2, 3])
        >>> single_lable_metic(preds, labels)
        {'precision': 50.0, 'recall': 50.0, 'f1-score': 50.0}

    Use PyTorch implementation:

        >>> import torch
        >>> preds = torch.Tensor([0, 2, 1, 3])
        >>> labels = torch.Tensor([0, 1, 2, 3])
        >>> single_lable_metic(preds, labels)
        {'precision': 50.0, 'recall': 50.0, 'f1-score': 50.0}

    Computing with `micro` average mode:

        >>> preds = np.asarray([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.3, 0.4, 0.2, 0.1],
            [0.0, 0.0, 0.1, 0.9]])
        >>> labels = np.asarray([0, 1, 2, 3])
        >>> single_lable_metic = SingleLabelMetric(average='micro')
        >>> single_lable_metic(preds, labels)
        {'precision_micro': 50.0, 'recall_micro': 50.0, 'f1-score_micro': 50.0} # noqa

    Accumulate batch:

        >>> for i in range(10):
        ...     preds = torch.randint(0, 4, size=(100, ))
        ...     labels = torch.randint(0, 4, size=(100, ))
        ...     single_lable_metic.add(preds, labels)
        >>> single_lable_metic.compute()  # doctest: +SKIP
    """

    def __init__(self,
                 thrs: Union[float, Sequence[Optional[float]], None] = 0.,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 average: Optional[str] = 'macro',
                 num_classes: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs, )
        else:
            self.thrs = tuple(thrs)  # type: ignore

        for item in items:
            assert item in ['precision', 'recall', 'f1-score', 'support'], \
                f'The metric {item} is not supported by `SingleLabelMetric`,' \
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

    def _format_metric_results(self, results: Sequence) -> Dict:
        """Format the given metric results into a dictionary.

        Args:
            results (Sequence): A list of per topk and thrs metrics.

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

        if isinstance(results[0], tuple):
            # for predictions with scores
            multi_thrs = len(self.thrs) > 1
            for i, thr in enumerate(self.thrs):
                if multi_thrs:
                    suffix = '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                else:
                    suffix = ''

                for k, v in pack_results(*results[i]).items():
                    metrics[k + suffix] = v
        else:
            metrics = pack_results(*results)

        result_metrics = dict()
        for k, v in metrics.items():

            if self.average is None:
                result_metrics[k + '_classwise'] = v.tolist()
            elif self.average == 'micro':
                result_metrics[k + f'_{self.average}'] = v.item()
            else:
                result_metrics[k] = v.item()

        return result_metrics

    @overload  # type: ignore
    @dispatch
    def _compute_metric(self, predictions: Sequence['torch.Tensor'],
                        labels: Sequence['torch.Tensor']) -> List[Any]:
        """A PyTorch implementation that computes the accuracy metric."""
        preds = torch.stack(predictions)
        labels = torch.stack(labels)

        # cannot be raised in current implementation because
        # `add` method will guarantee the equal length.
        # However length check should remain somewhere.
        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        if preds.ndim == 1:
            assert self.num_classes is not None, \
                'Please specify `num_classes` in `self` if the `preds`'\
                'is labels instead of scores.'
            gt_positive = F.one_hot(labels.flatten().to(torch.int64),
                                    self.num_classes)
            pred_positive = F.one_hot(preds.to(torch.int64), self.num_classes)
            return _precision_recall_f1_support(  # type: ignore
                pred_positive, gt_positive, self.average)
        else:
            # For pred score, calculate on all thresholds.
            num_classes = preds.shape[1]
            if self.num_classes is not None:
                assert num_classes == self.num_classes, \
                    'Number of classes does not match between preds' \
                    f'({num_classes}) and `self` ({self.num_classes}).'
            pred_score, pred_label = torch.topk(preds, k=1)
            pred_score = pred_score.flatten()
            pred_label = pred_label.flatten()

            gt_positive = F.one_hot(labels.flatten().to(torch.int64),
                                    num_classes)

            results = []
            for thr in self.thrs:
                pred_positive = F.one_hot(
                    pred_label.to(torch.int64), num_classes)
                if thr is not None:
                    pred_positive[pred_score <= thr] = 0
                results.append(
                    _precision_recall_f1_support(pred_positive, gt_positive,
                                                 self.average))

            return results

    @overload
    @dispatch
    def _compute_metric(
            self, predictions: Sequence[Union[int, Sequence[Union[int,
                                                                  float]]]],
            labels: Sequence[Union[int, Sequence[int]]]) -> List[Any]:
        """A Builtin implementation that computes the metric."""

        return self._compute_metric([np.array(pred) for pred in predictions],
                                    [np.int64(label) for label in labels])

    @dispatch
    def _compute_metric(self, predictions: Sequence[Union[np.ndarray,
                                                          np.number]],
                        labels: Sequence[np.number]) -> List[Any]:
        """A NumPy implementation that computes the metric."""
        preds = np.stack(predictions)
        labels = np.stack(labels)

        # cannot be raised in current implementation because
        # `add` method will guarantee the equal length.
        # However length check should remain somewhere.
        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        if preds.ndim == 1:
            assert self.num_classes is not None, \
                'Please specify `num_classes` in `self` if the `preds`'\
                'is labels instead of scores.'
            gt_positive = np.eye(self.num_classes, dtype=np.int64)[labels]

            pred_positive = np.eye(self.num_classes, dtype=np.int64)[preds]

            return _precision_recall_f1_support(  # type: ignore
                pred_positive, gt_positive, self.average)
        else:
            # For pred score, calculate on all thresholds.
            num_classes = preds.shape[1]
            if self.num_classes is not None:
                assert num_classes == self.num_classes, \
                    'Number of classes does not match between preds' \
                    f'({num_classes}) and `self` ({self.num_classes}).'
            pred_score = preds.max(axis=1)
            pred_label = preds.argmax(axis=1)

            gt_positive = np.eye(num_classes, dtype=np.int64)[labels]

            results = []
            for thr in self.thrs:
                pred_positive = np.eye(num_classes, dtype=np.int64)[pred_label]
                if thr is not None:
                    pred_positive[pred_score <= thr] = 0
                results.append(
                    _precision_recall_f1_support(pred_positive, gt_positive,
                                                 self.average))

            return results

    def compute_metric(
        self, results: List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS,
                                  BUILTIN_IMPL_HINTS]]
    ) -> Dict[str, float]:
        """Compute the accuracy metric.

        Currently, there are 2 actual implementations of this method: NumPy and
        PyTorch. Which implementation to use is determined by the type of the
        calling parameters. e.g. `numpy.ndarray` or `torch.Tensor`.

        Builtin type of data will be converted to `numpy.ndarray` for default
        implementation.

        This method would be invoked in `BaseMetric.compute` after distributed
        synchronization.

        Args:
            results (List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS,
                BUILTIN_IMPL_HINTS]]): A list
                of tuples that consisting the prediction and label. This list
                has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed accuracy metric.
        """
        predictions = [res[0] for res in results]
        labels = [res[1] for res in results]
        metric_results = self._compute_metric(predictions, labels)
        return self._format_metric_results(metric_results)
