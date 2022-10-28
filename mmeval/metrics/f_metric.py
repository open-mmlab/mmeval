# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union, overload

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.utils import try_import

if TYPE_CHECKING:
    import torch
else:
    torch = try_import('torch')


class F1Metric(BaseMetric):
    """Compute F1 scores.

    Args:
        num_classes (int): Number of labels.
        mode (str or list[str]): There are 2 options:

            - 'micro': Calculate metrics globally by counting the total true
              positives, false negatives and false positives.
            - 'macro': Calculate metrics for each label, and find their
              unweighted mean.

            If mode is a list, then metrics in mode will be calculated
            separately. Defaults to 'micro'.
        cared_classes (list[int]): The indices of the labels participated in
            the metric computing. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        ignored_classes (list[int]): The index set of labels that are ignored
            when computing metrics. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        **kwargs: Keyword arguments passed to :class:`BaseMetric`.

    Warning:
        Only non-negative integer labels are involved in computing. All
        negative ground truth labels will be ignored.

    Examples:

        >>> from mmeval import F1Metric
        >>> f1 = F1Metric(num_classes=5, mode=['macro', 'micro'])

    Use NumPy implementation:

        >>> import numpy as np
        >>> labels = np.asarray([0, 1, 4])
        >>> preds = np.asarray([0, 1, 2])
        >>> f1(preds, labels)
        {'macro_f1': 0.4,
         'micro_f1': 0.6666666666666666}

    Use PyTorch implementation:

        >>> import torch
        >>> labels = torch.Tensor([0, 1, 4])
        >>> preds = torch.Tensor([0, 1, 2])
        >>> f1(preds, labels)
        {'macro_f1': 0.4,
         'micro_f1': 0.6666666666666666}

    Accumulate batch:

        >>> for i in range(10):
        ...     labels = torch.randint(0, 4, size=(20, ))
        ...     predicts = torch.randint(0, 4, size=(20, ))
        ...     f1.add(predicts, labels)
        >>> f1.compute()  # doctest: +SKIP
    """

    def __init__(self,
                 num_classes: int,
                 mode: Union[str, Sequence[str]] = 'micro',
                 cared_classes: Sequence[int] = [],
                 ignored_classes: Sequence[int] = [],
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert isinstance(num_classes, int)
        assert isinstance(cared_classes, (list, tuple))
        assert isinstance(ignored_classes, (list, tuple))
        assert isinstance(mode, (list, str))
        assert not (len(cared_classes) > 0 and len(ignored_classes) > 0), \
            'cared_classes and ignored_classes cannot be both non-empty'

        if isinstance(mode, str):
            mode = [mode]
        assert set(mode).issubset({'micro', 'macro'})
        self.mode = mode

        if len(cared_classes) > 0:
            assert min(cared_classes) >= 0 and \
                max(cared_classes) < num_classes, \
                'cared_classes must be a subset of [0, num_classes)'
            self.cared_labels = sorted(cared_classes)
        elif len(ignored_classes) > 0:
            assert min(ignored_classes) >= 0 and \
                max(ignored_classes) < num_classes, \
                'ignored_classes must be a subset of [0, num_classes)'
            self.cared_labels = sorted(
                set(range(num_classes)) - set(ignored_classes))
        else:
            self.cared_labels = list(range(num_classes))
        self.cared_labels = np.array(self.cared_labels, dtype=np.int64)
        self.num_classes = num_classes

    def add(self, predictions: Sequence[Union[Sequence[int], np.ndarray]], labels: Sequence[Union[Sequence[int], np.ndarray]]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Calculate the following 2 stuff from the inputs and store them in
        ``self._results``:

        - prediction: prediction labels.
        - label: ground truth labels.

        Args:
            predictions (Sequence[Sequence[int] or np.ndarray]): A batch
                of sequences of non-negative integer labels.
            labels (Sequence[Sequence[int] or np.ndarray]): A batch of
                sequences of non-negative integer labels.
        """
        for prediction, label in zip(predictions, labels):
            self._results.append((prediction, label))

    @overload  # type: ignore
    @dispatch
    def _compute_tp_fp_fn(self, predictions: Sequence['torch.Tensor'],
                          labels: Sequence['torch.Tensor']) -> tuple:
        """Compute tp, fp and fn from predictions and labels."""
        preds = torch.cat(predictions).long().flatten().cpu()
        gts = torch.cat(labels).long().flatten().cpu()

        assert preds.max() < self.num_classes
        assert gts.max() < self.num_classes

        cared_labels = preds.new_tensor(self.cared_labels, dtype=torch.long)

        hits = (preds == gts)[None, :]
        preds_per_label = cared_labels[:, None] == preds[None, :]
        gts_per_label = cared_labels[:, None] == gts[None, :]

        tp = (hits * preds_per_label).cpu().numpy().astype(float)
        fp = (~hits * preds_per_label).cpu().numpy().astype(float)
        fn = (~hits * gts_per_label).cpu().numpy().astype(float)
        return tp, fp, fn

    @dispatch
    def _compute_tp_fp_fn(self, predictions: Sequence[Union[np.ndarray, int]],
                          labels: Sequence[Union[np.ndarray, int]]) -> tuple:
        """Compute tp, fp and fn from predictions and labels."""
        preds = np.concatenate(predictions, axis=0).astype(np.int64).flatten()
        gts = np.concatenate(labels, axis=0).astype(np.int64).flatten()

        assert preds.max() < self.num_classes  # type: ignore
        assert gts.max() < self.num_classes  # type: ignore

        hits = np.equal(preds, gts)[None, :]
        preds_per_label = np.equal(self.cared_labels[:, None], preds[None, :])  # type: ignore # yapf: disable # noqa: E501
        gts_per_label = np.equal(self.cared_labels[:, None], gts[None, :])  # type: ignore # yapf: disable # noqa: E501

        tp = (hits * preds_per_label).astype(float)
        fp = ((1 - hits) * preds_per_label).astype(float)
        fn = ((1 - hits) * gts_per_label).astype(float)
        return tp, fp, fn

    def compute_metric(
            self, results: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[(ndarray, ndarray)]): The processed results of each
                batch.

        Returns:
            dict[str, float]: The f1 scores. The keys are the names of the
            metrics, and the values are corresponding results. Possible
            keys are 'micro_f1' and 'macro_f1'.
        """

        preds, gts = zip(*results)

        tp, fp, fn = self._compute_tp_fp_fn(preds, gts)

        result = {}
        if 'macro' in self.mode:
            result['macro_f1'] = self._compute_f1(
                tp.sum(-1), fp.sum(-1), fn.sum(-1))
        if 'micro' in self.mode:
            result['micro_f1'] = self._compute_f1(tp.sum(), fp.sum(), fn.sum())

        return result

    def _compute_f1(self, tp: np.ndarray, fp: np.ndarray,
                    fn: np.ndarray) -> float:
        """Compute the F1-score based on the true positives, false positives
        and false negatives.

        Args:
            tp (np.ndarray): The true positives.
            fp (np.ndarray): The false positives.
            fn (np.ndarray): The false negatives.

        Returns:
            float: The F1-score.
        """
        precision = tp / (tp + fp).clip(min=1e-8)
        recall = tp / (tp + fn).clip(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clip(min=1e-8)
        return float(f1.mean())
