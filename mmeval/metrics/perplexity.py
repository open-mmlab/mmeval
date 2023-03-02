# Copyright (c) OpenMMLab. All rights reserved.
# This class is modified from `torchmetrics
# <https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/perplexity.py>`_.
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union, overload

from mmeval import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.utils import is_list_of, try_import

if TYPE_CHECKING:
    import oneflow
    import oneflow as flow
    import paddle
    import tensorflow
    import tensorflow as tf
    import torch
else:
    paddle = try_import('paddle')
    torch = try_import('torch')
    tf = try_import('tensorflow')
    flow = try_import('oneflow')


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute the softmax function.

    Args:
        x (numpy.ndarray): The inputs.

    Returns:
        numpy.ndarray: The outputs after softmax.
    """
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


_CHECK_HINTS = Union[np.ndarray, 'tensorflow.Tensor', 'torch.Tensor',
                     'paddle.Tensor', 'oneflow.Tensor']


def _check_inputs_shape(pred: _CHECK_HINTS, target: _CHECK_HINTS) -> None:
    """Check the shape of inputs.

    Args:
        pred (numpy.ndarray | tensorflow.Tensor |
            torch.Tensor | paddle.Tensor | oneflow.Tensor): The prediction.
        target (numpy.ndarray | tensorflow.Tensor |
            torch.Tensor | paddle.Tensor | oneflow.Tensor): The target.
    """
    if pred.ndim != 2:
        raise ValueError(
            'Input tensor `pred` is expected to have 2 dimensions, '
            f'[seq_len, vocab_size], but got {pred.ndim}.')
    if target.ndim != 1:
        raise ValueError(
            'Input tensor `target` is expected to have 1 dimensions, '
            f'[seq_len, ], but got {target.ndim}.')
    if pred.shape[:1] != target.shape:
        raise ValueError('Input tensors `pred` and `target` are expected '
                         'to have equaling first dimensions, [seq_len, ], '
                         f'but got {pred.shape[:1]} and {target.shape}.')


class Perplexity(BaseMetric):
    """Perplexity measures how well a language model predicts a text sample.

    It is commonly used as a metric for evaluating the quality of a
    language model. It is defined as 2 to the power of the cross-entropy
    loss of the model (or the negative log-likelihood of the sample).

    Args:
        ignore_labels (int or list[int], optional): Integer specifying a
            target class to ignore. If given, this class index does not
            contribute to the returned score. Defaults to None.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import Perplexity
        >>> import numpy as np
        >>>
        >>> preds = np.random.rand(2, 4, 2)
        >>> targets = np.random.randint(low=0, high=2, size=(2, 4))
        >>> metric = Perplexity()
        >>> result = metric(preds, targets)  # doctest: +ELLIPSIS
        {'perplexity': ...}
    """

    def __init__(self,
                 ignore_labels: Union[int, List[int], None] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(ignore_labels, int):
            ignore_labels = [ignore_labels]
        if ignore_labels is not None:
            if not is_list_of(ignore_labels, int):
                raise ValueError(
                    'Argument `ignore_labels` expected to be '
                    f'`None`, `int`, or`List[int]`, but got {ignore_labels}')
            ignore_labels = list(set(ignore_labels))
        self.ignore_labels = ignore_labels

    def add(self, predictions: Sequence, targets: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
            predictions (Sequence): Probabilities assigned to each token in
                a sequence with shape [batch_size, seq_len, vocab_size].
            targets (Sequence): Ground truth values with a
                shape [batch_size, seq_len].
        """
        for prediction, target in zip(predictions, targets):
            _check_inputs_shape(prediction, target)
            total_probs, count = self._compute_perplexity(prediction, target)
            self._results.append((total_probs, count))

    @overload
    @dispatch
    def _compute_perplexity(  # type: ignore
            self, prediction: 'torch.Tensor',
            target: 'torch.Tensor') -> Tuple[float, int]:
        """Compute the perplexity with PyTorch.

        Args:
            prediction (torch.Tensor | Sequence): Prediction from the model.
                Same as ``self.add``.
            target (torch.Tensor | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            Tuple (float, int): include the value of the total and count.
        """
        probs = torch.nn.functional.softmax(prediction, dim=1)
        if self.ignore_labels is not None:
            mask = torch.ones_like(target, dtype=bool)
            for ignore_label in self.ignore_labels:
                mask &= target.ne(ignore_label)
            target = torch.masked_fill(target, ~mask, 0)
            probs = probs.index_select(dim=1, index=target).diagonal()[mask]
        else:
            probs = probs.index_select(dim=1, index=target).diagonal()
        total_probs = -probs.log().sum()
        count = torch.tensor(probs.size()[0])
        return total_probs.item(), count.item()

    @overload
    @dispatch
    def _compute_perplexity(  # type: ignore
            self, prediction: 'oneflow.Tensor',
            target: 'oneflow.Tensor') -> Tuple[float, int]:
        """Compute the perplexity with OneFlow.

        Args:
            prediction (oneflow.Tensor | Sequence): Prediction from the model.
                Same as ``self.add``.
            target (oneflow.Tensor | Sequence): The ground truth labels.
                Same as ``self.add``.

        Returns:
            Tuple (float, int): include the value of the total and count.
        """
        probs = flow.nn.functional.softmax(prediction, dim=1)
        if self.ignore_labels is not None:
            mask = flow.ones_like(target)
            for ignore_label in self.ignore_labels:
                mask &= target.ne(ignore_label)
            target = flow.masked_fill(target, ~mask, 0)
            probs = probs.index_select(dim=1, index=target).diagonal()[mask]
        else:
            probs = probs.index_select(dim=1, index=target).diagonal()
        total = -probs.log().sum()
        count = flow.tensor(probs.size()[0])
        result = (total.item(), count.item())
        return result

    @overload
    @dispatch
    def _compute_perplexity(  # type: ignore
            self, prediction: 'tensorflow.Tensor',
            target: 'tensorflow.Tensor') -> Tuple[float, int]:
        """Compute the perplexity with TensorFlow.

        Args:
            prediction (tensorflow.Tensor | Sequence): Prediction from
                the model. Same as ``self.add``.
            target (tensorflow.Tensor | Sequence): The ground truth labels.
                Same as ``self.add``.

        Returns:
            Tuple (float, int): include the value of the total and count.
        """
        probs = tf.nn.softmax(prediction, axis=1)
        if self.ignore_labels is not None:
            mask = tf.ones_like(target, dtype=bool)
            for ignore_label in self.ignore_labels:
                mask &= tf.not_equal(target, ignore_label)
            target = tf.where(~mask, 0, target)

            probs = tf.gather(probs, target, axis=1)
            probs = tf.linalg.tensor_diag_part(probs)[mask]
        else:
            probs = tf.gather(probs, target, axis=1)
            probs = tf.linalg.tensor_diag_part(probs)
        probs = tf.math.log(probs)
        total = -tf.math.reduce_sum(probs)
        count = tf.shape(probs)[0]
        result = (total.numpy().item(), count.numpy().item())
        return result

    @overload
    @dispatch
    def _compute_perplexity(  # type: ignore
            self, prediction: 'paddle.Tensor',
            target: 'paddle.Tensor') -> Tuple[float, int]:
        """Compute the perplexity with PaddlePaddle.

        Args:
            prediction (paddle.Tensor | Sequence): Prediction from the model.
                Same as ``self.add``.
            target (paddle.Tensor | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            Tuple (float, int): include the value of the total and count.
        """
        probs = paddle.nn.functional.softmax(prediction, axis=1)
        if self.ignore_labels is not None:
            mask = paddle.ones_like(target, dtype=bool)
            for ignore_label in self.ignore_labels:
                compare_label = paddle.full(
                    target.shape, ignore_label, dtype=target.dtype)
                mask &= target.not_equal(compare_label)
            replace = paddle.zeros_like(target)
            target = paddle.where(~mask, replace, target)
            probs = probs.index_select(axis=1, index=target).diagonal()[mask]
        else:
            probs = probs.index_select(axis=1, index=target).diagonal()
        total = -probs.log().sum(axis=0)
        count = paddle.to_tensor(probs.shape[0])
        result = (total.item(), count.item())
        return result

    @dispatch
    def _compute_perplexity(self, prediction: np.ndarray,
                            target: np.ndarray) -> Tuple[float, int]:
        """Compute the perplexity with NumPy.

        Args:
            prediction (np.ndarray | Sequence): Prediction from the model.
                Same as ``self.add``.
            target (np.ndarray | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            Tuple (float, int): include the value of the total and count.
        """
        probs = softmax(prediction)
        if self.ignore_labels is not None:
            mask = np.ones_like(target, dtype=bool)
            for ignore_label in self.ignore_labels:
                mask &= np.not_equal(target, ignore_label)
            target = np.ma.array(target, mask=~mask, fill_value=0).filled()
            probs = np.take(probs, target, axis=1).diagonal()[mask]
        else:
            probs = np.take(probs, target, axis=1).diagonal()
        total = -np.sum(np.log(probs))
        count = probs.size
        result = (total.item(), count)
        return result

    def compute_metric(self, results: List[Tuple[float, int]]) \
            -> Dict[str, float]:
        """Compute the perplexity metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (list): A list that consisting the total and count. This
                list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed perplexity metric.
        """
        total = 0.0
        count = 0
        for result in results:
            total += result[0]
            count += result[1]
        output = np.exp(total / count) if count != 0 else 0
        perplexity = {'perplexity': float(output)}
        return perplexity
