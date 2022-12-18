# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, overload

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
    jax = try_import('jax')
    jnp = try_import('jax.numpy')
    paddle = try_import('paddle')
    torch = try_import('torch')
    tf = try_import('tensorflow')
    flow = try_import('oneflow')


class MeanIoU(BaseMetric):
    """MeanIoU evaluation metric.

    MeanIoU is a widely used evaluation metric for image semantic segmentation.

    In addition to mean iou, it will also compute and return accuracy, mean
    accuracy, mean dice, mean precision, mean recall and mean f-score.

    This metric supports 6 kinds of inputs, i.e. ``numpy.ndarray``,
    ``torch.Tensor``, ``oneflow.Tensor``, ``tensorflow.Tensor``,
    ``paddle.Tensor``and``jax.Array``, and the implementation for
    the calculation depends on the inputs type.

    Args:
        num_classes (int, optional): The number of classes. If None, it will be
            obtained from the 'num_classes' or 'classes' field in
            `self.dataset_meta`. Defaults to None.
        ignore_index (int, optional): Index that will be ignored in evaluation.
            Defaults to 255.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Defaults to None.
        beta (int, optional): Determines the weight of recall in the F-score.
            Defaults to 1.
        classwise_results (bool, optional): Whether to return the computed
            results of each class. Defaults to False.
        **kwargs: Keyword arguments passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import MeanIoU
        >>> miou = MeanIoU(num_classes=4)

    Use NumPy implementation:

        >>> import numpy as np
        >>> labels = np.asarray([[[0, 1, 1], [2, 3, 2]]])
        >>> preds = np.asarray([[[0, 2, 1], [1, 3, 2]]])
        >>> miou(preds, labels)
        {'aAcc': 0.6666666666666666,
         'mIoU': 0.6666666666666666,
         'mAcc': 0.75,
         'mDice': 0.75,
         'mPrecision': 0.75,
         'mRecall': 0.75,
         'mFscore': 0.75,
         'kappa': 0.5384615384615384}

    Use PyTorch implementation:

        >>> import torch
        >>> labels = torch.Tensor([[[0, 1, 1], [2, 3, 2]]])
        >>> preds = torch.Tensor([[[0, 2, 1], [1, 3, 2]]])
        >>> miou(preds, labels)
        {'aAcc': 0.6666666666666666,
         'mIoU': 0.6666666666666666,
         'mAcc': 0.75,
         'mDice': 0.75,
         'mPrecision': 0.75,
         'mRecall': 0.75,
         'mFscore': 0.75,
         'kappa': 0.5384615384615384}

    Accumulate batch:

        >>> for i in range(10):
        ...     labels = torch.randint(0, 4, size=(100, 10, 10))
        ...     predicts = torch.randint(0, 4, size=(100, 10, 10))
        ...     miou.add(predicts, labels)
        >>> miou.compute()  # doctest: +SKIP
    """

    def __init__(self,
                 num_classes: Optional[int] = None,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 classwise_results: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._num_classes = num_classes
        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.classwise_results = classwise_results

    @property
    def num_classes(self) -> int:
        """Returns the number of classes.

        The number of classes should be set during initialization, otherwise it
        will be obtained from the 'classes' or 'num_classes' field in
        ``self.dataset_meta``.

        Raises:
            RuntimeError: If the num_classes is not set.

        Returns:
            int: The number of classes.
        """
        if self._num_classes is not None:
            return self._num_classes
        if self.dataset_meta and 'num_classes' in self.dataset_meta:
            self._num_classes = self.dataset_meta['num_classes']
        elif self.dataset_meta and 'classes' in self.dataset_meta:
            self._num_classes = len(self.dataset_meta['classes'])
        else:
            raise RuntimeError(
                'The `num_claases` is required, and not found in '
                f'dataset_meta: {self.dataset_meta}')
        return self._num_classes

    def add(self, predictions: Sequence, labels: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Calculate the following 3 stuff from the inputs and store them in
        ``self._results``:

        - num_tp_per_class: the number of true positive per-class.
        - num_gts_per_class: the number of ground truth per-class.
        - num_preds_per_class: the number of predicition per-class.

        Args:
            predictions (Sequence): A sequence of the predicted segmentation
                mask.
            labels (Sequence): A sequence of the segmentation mask labels.
        """
        for prediction, label in zip(predictions, labels):
            assert prediction.shape == label.shape, 'The shape of ' \
                '`prediction` and `label` should be the same, but got: ' \
                f'{prediction.shape} and {label.shape}'
            # We assert the prediction and label should be a segmentation mask.
            assert len(prediction.shape) == 2, 'The dimension of ' \
                f'`prediction` should be 2, but got shape: {prediction.shape}'
            # Store the intermediate result used to calculate IoU.
            confusion_matrix = self.compute_confusion_matrix(
                prediction, label, self.num_classes)
            num_tp_per_class = np.diag(confusion_matrix)
            num_gts_per_class = confusion_matrix.sum(1)
            num_preds_per_class = confusion_matrix.sum(0)
            self._results.append(
                (num_tp_per_class, num_gts_per_class, num_preds_per_class), )

    @overload  # type: ignore
    @dispatch
    def compute_confusion_matrix(self, prediction: np.ndarray,
                                 label: np.ndarray,
                                 num_classes: int) -> np.ndarray:
        """Compute confusion matrix with NumPy.

        Args:
            prediction (numpy.ndarray): The predicition.
            label (numpy.ndarray): The ground truth.
            num_classes (int): The number of classes.

        Returns:
            numpy.ndarray: The computed confusion matrix.
        """
        mask = (label != self.ignore_index)
        prediction, label = prediction[mask], label[mask]
        confusion_matrix_1d = np.bincount(
            num_classes * label + prediction, minlength=num_classes**2)
        confusion_matrix = confusion_matrix_1d.reshape(num_classes,
                                                       num_classes)
        return confusion_matrix

    @overload  # type: ignore
    @dispatch
    def compute_confusion_matrix(  # type: ignore
            self, prediction: 'torch.Tensor', label: 'torch.Tensor',
            num_classes: int) -> np.ndarray:
        """Compute confusion matrix with PyTorch.

        Args:
            prediction (torch.Tensor): The predicition.
            label (torch.Tensor): The ground truth.
            num_classes (int): The number of classes.

        Returns:
            numpy.ndarray: The computed confusion matrix.
        """
        mask = (label != self.ignore_index)
        prediction, label = prediction[mask], label[mask]
        confusion_matrix_1d = torch.bincount(
            num_classes * label + prediction, minlength=num_classes**2)
        confusion_matrix = confusion_matrix_1d.reshape(num_classes,
                                                       num_classes)
        return confusion_matrix.cpu().numpy()

    @overload  # type: ignore
    @dispatch
    def compute_confusion_matrix(  # type: ignore
            self, prediction: 'jax.Array', label: 'jax.Array',
            num_classes: int) -> np.ndarray:
        """Compute confusion matrix with JAX.

        Args:
            prediction (jax.Array): The predicition.
            label (jax.Array): The ground truth.
            num_classes (int): The number of classes.

        Returns:
            numpy.ndarray: The computed confusion matrix.
        """
        mask = (label != self.ignore_index)
        prediction, label = prediction[mask], label[mask]
        confusion_matrix_1d = jnp.bincount(
            num_classes * label + prediction, minlength=num_classes**2)
        confusion_matrix = confusion_matrix_1d.reshape(num_classes,
                                                       num_classes)
        return np.asarray(confusion_matrix)

    @overload  # type: ignore
    @dispatch
    def compute_confusion_matrix(  # type: ignore
            self, prediction: 'oneflow.Tensor', label: 'oneflow.Tensor',
            num_classes: int) -> np.ndarray:
        """Compute confusion matrix with OneFlow.

        Args:
            prediction (oneflow.Tensor): The predicition.
            label (oneflow.Tensor): The ground truth.
            num_classes (int): The number of classes.

        Returns:
            numpy.ndarray: The computed confusion matrix.
        """
        mask = (label != self.ignore_index)
        prediction, label = prediction[mask], label[mask]
        confusion_matrix_1d = flow.bincount(
            num_classes * label + prediction, minlength=num_classes**2)
        confusion_matrix = confusion_matrix_1d.reshape(num_classes,
                                                       num_classes)
        return confusion_matrix.cpu().numpy()

    @overload
    @dispatch
    def compute_confusion_matrix(  # type: ignore
            self, prediction: 'paddle.Tensor', label: 'paddle.Tensor',
            num_classes: int) -> np.ndarray:
        """Compute confusion matrix with Paddle.

        Args:
            prediction (paddle.Tensor): The predicition.
            label (paddle.Tensor): The ground truth.
            num_classes (int): The number of classes.

        Returns:
            numpy.ndarray: The computed confusion matrix.
        """
        mask = (label != self.ignore_index)
        prediction, label = prediction[mask], label[mask]
        # NOTE: Since the `paddle.bincount` has bug on the CUDA device, we use
        # the `np.bincount` instead. Once the bug is fixed, we will use
        # `paddle.bincount`.
        # For more see at: https://github.com/PaddlePaddle/Paddle/issues/46978
        confusion_matrix_1d = np.bincount(
            num_classes * label + prediction, minlength=num_classes**2)
        confusion_matrix = confusion_matrix_1d.reshape(
            (num_classes, num_classes))
        return confusion_matrix

    @dispatch
    def compute_confusion_matrix(  # type: ignore
            self, prediction: 'tensorflow.Tensor', label: 'tensorflow.Tensor',
            num_classes: int) -> np.ndarray:
        """Compute confusion matrix with TensorFlow.

        Args:
            prediction (tensorflow.Tensor): The predicition.
            label (tensorflow.Tensor): The ground truth.
            num_classes (int): The number of classes.

        Returns:
            numpy.ndarray: The computed confusion matrix.
        """
        mask = (label != self.ignore_index)
        prediction, label = prediction[mask], label[mask]
        confusion_matrix_1d = tf.math.bincount(
            tf.cast(num_classes * label + prediction, tf.int32),
            minlength=num_classes**2)
        confusion_matrix = tf.reshape(confusion_matrix_1d,
                                      (num_classes, num_classes))
        return confusion_matrix.numpy()

    def compute_metric(
        self,
        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict:
        """Compute the MeanIoU metric.

        This method would be invoked in `BaseMetric.compute` after distributed
        synchronization.

        Args:
            results (List[tuple]): This list has already been synced across all
                ranks. This is a list of tuple, and each tuple has the
                following elements:

                - (List[numpy.ndarray]): Each element in the list is the number
                  of true positive per-class on a sample.
                - (List[numpy.ndarray]): Each element in the list is the number
                  of ground truth per-class on a sample.
                - (List[numpy.ndarray]): Each element in the list is the number
                  of predicition per-class on a sample.

        Returns:
            Dict: The computed metric, with following keys:

            - aAcc, the overall accuracy, namely pixel accuracy.
            - mIoU, the mean Intersection-Over-Union (IoU) for all classes.
            - mAcc, the mean accuracy for all classes, namely mean pixel
            accuracy.
            - mDice, the mean dice coefficient for all claases.
            - mPrecision, the mean precision for all classes.
            - mRecall, the mean recall for all classes.
            - mFscore, the mean f-score for all classes.
            - kappa, the Cohen's kappa coefficient.
            - classwise_result, the evaluate results of each classes.
            This would be returned if ``self.classwise_result`` is True.
        """
        # Gather the `num_tp_per_class` from batches results.
        num_tp_per_class: np.ndarray = sum(res[0] for res in results)
        # Gather the `num_gts_per_class` from batches results.
        num_gts_per_class: np.ndarray = sum(res[1] for res in results)
        # Gather the `num_preds_per_class` from batches results.
        num_preds_per_class: np.ndarray = sum(res[2] for res in results)

        # Computing overall accuracy.
        overall_acc = num_tp_per_class.sum() / num_gts_per_class.sum()

        # compute iou per class
        union = num_preds_per_class + num_gts_per_class - num_tp_per_class
        iou = num_tp_per_class / union

        # compute accuracy per class
        accuracy = num_tp_per_class / num_gts_per_class

        # compute dice per class
        dice = 2 * num_tp_per_class / (num_preds_per_class + num_gts_per_class)

        # compute precision, recall and f-score per class
        precision = num_tp_per_class / num_preds_per_class
        recall = num_tp_per_class / num_gts_per_class
        f_score = (1 + self.beta**2) * (precision * recall) / (
            (self.beta**2 * precision) + recall)

        # compute kappa coefficient
        po = num_tp_per_class.sum() / num_gts_per_class.sum()
        pe = (num_gts_per_class * num_preds_per_class).sum() / (
            num_gts_per_class.sum()**2)
        kappa = (po - pe) / (1 - pe)

        def _mean(values: np.ndarray):
            if self.nan_to_num is not None:
                values = np.nan_to_num(values, nan=self.nan_to_num)
            return np.nanmean(values)

        metric_results = {
            'aAcc': overall_acc,
            'mIoU': _mean(iou),
            'mAcc': _mean(accuracy),
            'mDice': _mean(dice),
            'mPrecision': _mean(precision),
            'mRecall': _mean(recall),
            'mFscore': _mean(f_score),
            'kappa': kappa,
        }

        # Add the class-wise metric results to the returned results.
        if self.classwise_results:
            metric_results['classwise_results'] = {
                'IoU': iou,
                'Acc': accuracy,
                'Dice': dice,
                'Precision': precision,
                'Recall': recall,
                'Fscore': f_score,
            }
        return metric_results
