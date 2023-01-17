# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import warnings
from typing import TYPE_CHECKING, Sequence, Tuple, Union

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


def label_to_onehot(
        label: Union[np.ndarray, 'torch.Tensor',
                     'oneflow.Tensor'], num_classes: int
) -> Union[np.ndarray, 'torch.Tensor', 'oneflow.Tensor']:
    """Convert the label-format input to one-hot encodings.

    Args:
        label (torch.Tensor or oneflow.Tensor or np.ndarray):
            The label-format input. The format of item must be label-format.
        num_classes (int): The number of classes.

    Return:
        torch.Tensor or oneflow.Tensor or np.ndarray:
        The converted one-hot encodings.
    """
    if torch and isinstance(label, torch.Tensor):
        label = label.long()
        onehot = label.new_zeros((num_classes, ))
    elif flow and isinstance(label, flow.Tensor):
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


def format_data(
    data: Union[Sequence[Union[np.ndarray, 'torch.Tensor', 'oneflow.Tensor']],
                np.ndarray, 'torch.Tensor', 'oneflow.Tensor'],
    num_classes: int,
    is_onehot: bool = False
) -> Union[np.ndarray, 'torch.Tensor', 'oneflow.Tensor']:
    """Format data from different inputs such as prediction scores, label-
    format data and one-hot encodings into the same output shape of `(N,
    num_classes)`.

    Args:
        data (Union[Sequence[np.ndarray, 'torch.Tensor', 'oneflow.Tensor'],
        np.ndarray, 'torch.Tensor', 'oneflow.Tensor']):
            The input data of prediction or labels.
        num_classes (int): The number of classes.
        is_onehot (bool): Whether the data is one-hot encodings.

    Return:
        torch.Tensor or oneflow.Tensor or np.ndarray:
        One-hot encodings or predict scores.
    """
    if torch and isinstance(data[0], torch.Tensor):
        stack_func = torch.stack
    elif flow and isinstance(data[0], flow.Tensor):
        stack_func = flow.stack
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
    # Error corresponds to np, torch, oneflow, stack_func respectively
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
