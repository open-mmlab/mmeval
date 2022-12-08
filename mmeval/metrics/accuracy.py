# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from typing import (TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union, overload)

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.utils import try_import

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
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
    jnp = try_import('jax.numpy')
    jax = try_import('jax')
    flow = try_import('oneflow')


@overload
@dispatch
def _is_scalar(obj: np.number):  # type: ignore
    """Check if the ``numpy.number`` is a scalar number."""
    return True


@overload
@dispatch
def _is_scalar(obj: Union[np.ndarray,  # type: ignore
                          'torch.Tensor', 'oneflow.Tensor',
                          'tensorflow.Tensor']):
    """Check if a ``np.ndarray`` | ``torch.Tensor`` | ``oneflow.Tensor``

    |``tensorflow.Tensor`` is a scalar.
    """
    return obj.ndim == 0


@dispatch
def _is_scalar(obj):
    """Check if an object is a scalar."""
    try:
        float(obj)  # type: ignore
        return True
    except Exception:
        return False


def _torch_topk(inputs: 'torch.Tensor',
                k: int,
                dim: Optional[int] = None) -> Tuple:
    """Invoke the PyTorch topk."""
    return inputs.topk(k, dim=dim)


def _oneflow_topk(inputs: 'oneflow.Tensor',
                  k: int,
                  dim: Optional[int] = None) -> Tuple:
    """Invoke the OneFlow topk."""
    return inputs.topk(k, dim=dim)


def _numpy_topk(inputs: np.ndarray,
                k: int,
                axis: Optional[int] = None) -> Tuple:
    """A implementation of numpy top-k.

    This implementation returns the values and indices of the k largest
    elements along a given axis.

    Args:
        inputs (numpy.ndarray): The input numpy array.
        k (int): The k in `top-k`.
        axis (int, optional): The axis to sort along.

    Returns:
        tuple: The values and indices of the k largest elements.

    Note:
        If PyTorch/OneFlow is available, the ``_torch_topk`` or
        ``_oneflow_topk`` would be used.
    """
    if torch is not None:
        values, indices = _torch_topk(torch.from_numpy(inputs), k, dim=axis)
        return values.numpy(), indices.numpy()

    if flow is not None:
        values, indices = _oneflow_topk(flow.from_numpy(inputs), k, dim=axis)
        return values.numpy(), indices.numpy()

    indices = np.argsort(inputs * -1.0, axis=axis)
    indices = np.take(indices, np.arange(k), axis=axis)
    values = np.take_along_axis(inputs, indices, axis=axis)
    return values, indices


def _jnp_topk(inputs: 'jax.Array',
              k: int,
              axis: Optional[int] = None) -> Tuple:
    """A implementation of jax.Array top-k.

    This implementation returns the values and indices of the k largest
    elements along a given axis.

    Args:
        inputs (jax.Array): The input jax Array.
        k (int): The k in `top-k`.
        axis (int, optional): The axis to sort along.

    Returns:
        tuple: The values and indices of the k largest elements.
    """
    if axis is None:
        return jax.lax.top_k(inputs, k)

    indices = jnp.argsort(inputs * -1.0, axis=axis)
    indices = jnp.take(indices, jnp.arange(k), axis=axis)
    values = jnp.take_along_axis(inputs, indices, axis=axis)
    return values, indices


class Accuracy(BaseMetric):
    """Top-k accuracy evaluation metric.

    This metric computes the accuracy based on the given topk and thresholds.

    Currently, this metric supports 5 kinds of inputs, i.e. ``numpy.ndarray``,
    ``torch.Tensor``, ``oneflow.Tensor``, ``tensorflow.Tensor`` and
    ``paddle.Tensor``, and the implementation for the calculation depends on
    the inputs type.

    Args:
        topk (int | Sequence[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Sequence[float | None] | float | None): Predictions with scores
            under the thresholds are considered negative. None means no
            thresholds. Defaults to 0.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import Accuracy
        >>> accuracy = Accuracy()

    Use NumPy implementation:

        >>> import numpy as np
        >>> labels = np.asarray([0, 1, 2, 3])
        >>> preds = np.asarray([0, 2, 1, 3])
        >>> accuracy(preds, labels)
        {'top1': 0.5}

    Use PyTorch implementation:

        >>> import torch
        >>> labels = torch.Tensor([0, 1, 2, 3])
        >>> preds = torch.Tensor([0, 2, 1, 3])
        >>> accuracy(preds, labels)
        {'top1': 0.5}

    Computing top-k accuracy with specified threold:

        >>> labels = np.asarray([0, 1, 2, 3])
        >>> preds = np.asarray([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.3, 0.4, 0.2, 0.1],
            [0.0, 0.0, 0.1, 0.9]])
        >>> accuracy = Accuracy(topk=(1, 2, 3))
        >>> accuracy(preds, labels)
        {'top1': 0.5, 'top2': 0.75, 'top3': 1.0}
        >>> accuracy = Accuracy(topk=2, thrs=(0.1, 0.5))
        >>> accuracy(preds, labels)
        {'top2_thr-0.10': 0.75, 'top2_thr-0.50': 0.5}

    Accumulate batch:

        >>> for i in range(10):
        ...     labels = torch.randint(0, 4, size=(100, ))
        ...     predicts = torch.randint(0, 4, size=(100, ))
        ...     accuracy.add(predicts, labels)
        >>> accuracy.compute()  # doctest: +SKIP
    """

    def __init__(self,
                 topk: Union[int, Sequence[int]] = (1, ),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(topk, int):
            self.topk = (topk, )
        else:
            self.topk = tuple(topk)  # type: ignore
        self.maxk = max(self.topk)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs, )
        else:
            self.thrs = tuple(thrs)  # type: ignore

    def add(self, predictions: Sequence, labels: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
            predictions (Sequence): Predictions from the model. It can be
                labels (N, ), or scores of every class (N, C).
            labels (Sequence): The ground truth labels. It should be (N, ).
        """
        corrects = self._compute_corrects(predictions, labels)
        for correct in corrects:
            self._results.append(correct)

    @overload  # type: ignore
    @dispatch
    def _compute_corrects(
        self, predictions: Union['torch.Tensor', Sequence['torch.Tensor']],
        labels: Union['torch.Tensor',
                      Sequence['torch.Tensor']]) -> 'torch.Tensor':
        """Compute the correct number of per topk and threshold with PyTorch.

        Args:
            prediction (torch.Tensor | Sequence): Predictions from the model.
                Same as ``self.add``.
            labels (torch.Tensor | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            torch.Tensor: Correct number with the following 2 shapes.

            - (N, ): If the ``predictions`` is a label tensor instead of score.
              Only return a top-1 correct tensor, and ignore the argument
              ``topk`` and ``thrs``.
            - (N, num_topk, num_thr): If the ``prediction`` is a score tensor
              (number of dimensions is 2). Return the correct number on each
              ``topk`` and ``thrs``.
        """
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.stack(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.stack(labels)

        if predictions.ndim == 1:
            corrects = (predictions.int() == labels)
            return corrects.float()

        pred_scores, pred_label = _torch_topk(predictions, self.maxk, dim=1)
        pred_label = pred_label.t()

        corrects = (pred_label == labels.view(1, -1).expand_as(pred_label))

        # compute the corrects corresponding to all topk and thrs per sample
        corrects_per_sample = torch.zeros(
            (len(predictions), len(self.topk), len(self.thrs)))
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                # Only prediction socres larger than thr are counted as correct
                if thr is not None:
                    thr_corrects = corrects & (pred_scores.t() > thr)
                else:
                    thr_corrects = corrects
                corrects_per_sample[:, i, j] = thr_corrects[:k].sum(
                    0, keepdim=True).float()
        return corrects_per_sample

    @overload  # type: ignore
    @dispatch
    def _compute_corrects(  # type: ignore
        self, predictions: Union['oneflow.Tensor', Sequence['oneflow.Tensor']],
        labels: Union['oneflow.Tensor',
                      Sequence['oneflow.Tensor']]) -> 'oneflow.Tensor':
        """Compute the correct number of per topk and threshold with OneFlow.

        Args:
            prediction (oneflow.Tensor | Sequence): Predictions from the model.
                Same as ``self.add``.
            labels (oneflow.Tensor | Sequence): The ground truth labels.
                Same as ``self.add``.

        Returns:
            oneflow.Tensor: Correct number with the following 2 shapes.

            - (N, ): If the ``predictions`` is a label tensor instead of score.
              Only return a top-1 correct tensor, and ignore the argument
              ``topk`` and ``thrs``.
            - (N, num_topk, num_thr): If the ``prediction`` is a score tensor
              (number of dimensions is 2). Return the correct number on each
              ``topk`` and ``thrs``.
        """
        if not isinstance(predictions, flow.Tensor):
            predictions = flow.stack(predictions)
        if not isinstance(labels, flow.Tensor):
            labels = flow.stack(labels)

        if predictions.ndim == 1:
            corrects = (predictions.int() == labels)
            return corrects.float()

        pred_scores, pred_label = _oneflow_topk(predictions, self.maxk, dim=1)
        pred_label = pred_label.t()

        corrects = (pred_label == labels.view(1, -1).expand_as(pred_label))

        # compute the corrects corresponding to all topk and thrs per sample
        corrects_per_sample = flow.zeros(
            (len(predictions), len(self.topk), len(self.thrs)))
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                # Only prediction socres larger than thr are counted as correct
                if thr is not None:
                    thr_corrects = corrects & (pred_scores.t() > thr)
                else:
                    thr_corrects = corrects
                corrects_per_sample[:, i, j] = thr_corrects[:k].sum(
                    0, keepdim=False).float()
        return corrects_per_sample

    @overload  # type: ignore
    @dispatch
    def _compute_corrects(  # type: ignore
        self, predictions: Union['tensorflow.Tensor',
                                 Sequence['tensorflow.Tensor']],
        labels: Union['tensorflow.Tensor',
                      Sequence['tensorflow.Tensor']]) -> 'tensorflow.Tensor':
        """Compute the correct number of per topk and threshold with
        TensorFlow.

        Args:
            prediction (tensorflow.Tensor | Sequence): Predictions from the
                model. Same as ``self.add``.
            labels (tensorflow.Tensor | Sequence): The ground truth labels.
                Same as ``self.add``.

        Returns:
            tensorflow.Tensor: Correct number with the following 2 shapes.

            - (N, ): If the ``predictions`` is a label tensor instead of score.
              Only return a top-1 correct tensor, and ignore the argument
              ``topk`` and ``thrs``.
            - (N, num_topk, num_thr): If the ``prediction`` is a score tensor
              (number of dimensions is 2). Return the correct number on each
              ``topk`` and ``thrs``.
        """
        if not isinstance(predictions, tf.Tensor):
            predictions = tf.stack(predictions)
        if not isinstance(labels, tf.Tensor):
            labels = tf.stack(labels)

        if predictions.ndim == 1:
            corrects = (tf.cast(predictions, labels.dtype) == labels)
            return tf.cast(corrects, tf.float64)

        pred_scores, pred_label = tf.math.top_k(predictions, self.maxk)
        pred_label = tf.transpose(pred_label)

        # broadcast `label` to the shape of `pred_label`
        labels = tf.broadcast_to(tf.reshape(labels, (1, -1)), pred_label.shape)
        # compute correct tensor
        corrects = (tf.cast(pred_label, labels.dtype) == labels)

        # compute the corrects corresponding to all topk and thrs per sample
        # NOTE: We should use a `tf.Variable` so that we can assign value.
        corrects_per_sample = tf.Variable(
            tf.zeros((len(predictions), len(self.topk), len(self.thrs)),
                     tf.int32))
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                # Only prediction socres larger than thr are counted as correct
                if thr is not None:
                    thr_corrects = corrects & (tf.transpose(pred_scores) > thr)
                else:
                    thr_corrects = corrects
                corrects_per_sample[:, i, j].assign(
                    tf.reduce_sum(tf.cast(thr_corrects[:k], tf.int32), axis=0))
        return corrects_per_sample.value()

    @overload  # type: ignore
    @dispatch
    def _compute_corrects(  # type: ignore
        self, predictions: Union['paddle.Tensor', Sequence['paddle.Tensor']],
        labels: Union['paddle.Tensor',
                      Sequence['paddle.Tensor']]) -> 'paddle.Tensor':
        """Compute the correct number of per topk and threshold with Paddle.

        Args:
            prediction (paddle.Tensor | Sequence): Predictions from the model.
                Same as ``self.add``.
            labels (paddle.Tensor | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            paddle.Tensor: Correct number with the following 2 shapes.

            - (N, ): If the ``predictions`` is a label tensor instead of score.
              Only return a top-1 correct tensor, and ignore the argument
              ``topk`` and ``thrs``.
            - (N, num_topk, num_thr): If the ``prediction`` is a score tensor
              (number of dimensions is 2). Return the correct number on each
              ``topk`` and ``thrs``.
        """
        if not isinstance(predictions, paddle.Tensor):
            predictions = paddle.stack(predictions)
        if not isinstance(labels, paddle.Tensor):
            labels = paddle.stack(labels)

        if predictions.ndim == 1:
            corrects = (predictions.cast(labels.dtype) == labels)
            return corrects.cast('float64')

        pred_scores, pred_label = paddle.topk(predictions, self.maxk)
        pred_label = pred_label.t()

        corrects = (
            pred_label == labels.reshape((1, -1)).expand_as(pred_label))

        # compute the corrects corresponding to all topk and thrs per sample
        # NOTE: The data type of `corrects_per_sample` should be 'float64',
        # otherwise will got wrong results when the shape of input is large.
        corrects_per_sample = paddle.zeros(
            (len(predictions), len(self.topk), len(self.thrs)), 'float64')
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                # Only prediction socres larger than thr are counted as correct
                if thr is not None:
                    thr_corrects = corrects & (pred_scores.t() > thr)
                else:
                    thr_corrects = corrects
                # NOTE: The `keepdim` should be True, otherwise will got
                # negative number.
                corrects_per_sample[:, i, j] = thr_corrects[:k].sum(
                    0, keepdim=False).cast('float64')
        return corrects_per_sample

    @overload
    @dispatch
    def _compute_corrects(  # type: ignore
            self, predictions: Union['jax.Array', Sequence['jax.Array']],
            labels: Union['jax.Array', Sequence['jax.Array']]) -> 'jax.Array':
        """Compute the correct number of per topk and threshold with JAX.

        Args:
            prediction (jax.Array | Sequence): Predictions from the model.
                Same as ``self.add``.
            labels (jax.Array | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            jax.Array: Correct number with the following 2 shapes.

            - (N, ): If the ``predictions`` is a label array instead of score.
              Only return a top-1 correct array, and ignore the argument
              ``topk`` and ``thrs``.
            - (N, num_topk, num_thr): If the ``prediction`` is a score array
              (number of dimensions is 2). Return the correct number on each
              ``topk`` and ``thrs``.
        """
        if not isinstance(predictions, jnp.ndarray):
            predictions = jnp.stack(predictions)
        if not isinstance(labels, jnp.ndarray):
            labels = jnp.stack(labels)

        if predictions.ndim == 1:
            corrects = (predictions == labels)
            return corrects.astype(jnp.int32)

        pred_scores, pred_label = _jnp_topk(predictions, self.maxk, axis=1)
        pred_label = pred_label.T
        # broadcast `label` to the shape of `pred_label`
        labels = jnp.broadcast_to(labels.reshape(1, -1), pred_label.shape)
        # compute correct array
        corrects = (pred_label == labels)

        # compute the corrects corresponding to all topk and thrs per sample
        corrects_per_sample = jnp.zeros(
            (len(predictions), len(self.topk), len(self.thrs)))

        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                # Only prediction socres larger than thr are counted as correct
                if thr is not None:
                    thr_corrects = corrects & (pred_scores.T > thr)
                else:
                    thr_corrects = corrects
                corrects_per_sample = corrects_per_sample.at[:, i, j].set(
                    thr_corrects[:k].sum(0,
                                         keepdims=True).astype(jnp.int32)[0])

        return corrects_per_sample

    @dispatch
    def _compute_corrects(
            self, predictions: Union[np.ndarray, Sequence[np.ndarray]],
            labels: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        """Compute the correct number of per topk and threshold with NumPy.

        Args:
            prediction (numpy.ndarray | Sequence): Predictions from the model.
                Same as ``self.add``.
            labels (numpy.ndarray | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            numpy.ndarray: Correct number with the following 2 shapes.

            - (N, ): If the ``predictions`` is a label array instead of score.
              Only return a top-1 correct array, and ignore the argument
              ``topk`` and ``thrs``.
            - (N, num_topk, num_thr): If the ``prediction`` is a score array
              (number of dimensions is 2). Return the correct number on each
              ``topk`` and ``thrs``.
        """
        if not isinstance(predictions, np.ndarray):
            predictions = np.stack(predictions)
        if not isinstance(labels, np.ndarray):
            labels = np.stack(labels)

        if predictions.ndim == 1:
            corrects = (predictions == labels)
            return corrects.astype(np.int32)

        pred_scores, pred_label = _numpy_topk(predictions, self.maxk, axis=1)
        pred_label = pred_label.T

        # broadcast `label` to the shape of `pred_label`
        labels = np.broadcast_to(labels.reshape(1, -1), pred_label.shape)
        # compute correct array
        corrects = (pred_label == labels)

        # compute the corrects corresponding to all topk and thrs per sample
        corrects_per_sample = np.zeros(
            (len(predictions), len(self.topk), len(self.thrs)))
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                # Only prediction socres larger than thr are counted as correct
                if thr is not None:
                    thr_corrects = corrects & (pred_scores.T > thr)
                else:
                    thr_corrects = corrects
                corrects_per_sample[:, i, j] = thr_corrects[:k].sum(
                    0, keepdims=True).astype(np.int32)
        return corrects_per_sample

    def compute_metric(
        self, results: List[Union[Iterable,
                                  Union[np.number, 'torch.Tensor',
                                        'tensorflow.Tensor', 'paddle.Tensor',
                                        'jax.Array', 'flow.Tensor']]]
    ) -> Dict[str, float]:
        """Compute the accuracy metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (list): A list that consisting the correct numbers. This
                list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed accuracy metric.
        """
        if _is_scalar(results[0]):
            return {'top1': float(sum(results) / len(results))}  # type: ignore

        metric_results = {}
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                corrects = [result[i][j] for result in results]  # type: ignore
                acc = float(sum(corrects) / len(corrects))
                name = f'top{k}'
                if len(self.thrs) > 1:
                    name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                metric_results[name] = acc
        return metric_results
