# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import warnings
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

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
    is_onehot: Optional[bool] = None
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
            If `None`, this will be automatically inducted.
            Defaults to `None`.

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

    def _induct_is_onehot(inferred_data):
        """Conduct the input data format."""
        shapes = {d.shape for d in inferred_data}
        if len(shapes) == 1:
            # stack scores or one-hot indices directly if have same shapes
            cand_formated_data = stack_func(inferred_data)
            # all the conditions below is to find whether labels that are
            # raw indices which should be converted to one-hot indices.
            # 1. one-hot indices should has 2 dims;
            # 2. one-hot indices should has num_classes as the second dim;
            # 3. one-hot indices values should always smaller than 2.
            if cand_formated_data.ndim == 2 \
                and cand_formated_data.shape[1] == num_classes \
                    and cand_formated_data.max() <= 1:
                if num_classes > 2:
                    return True, cand_formated_data
                elif num_classes == 2:
                    # 4. corner case, num_classes=2, then one-hot indices
                    # and raw indices are undistinguishable, for instance:
                    #   [[0, 1], [0, 1]] can be one-hot indices of 2 positives
                    #   or raw indices of 4 positives.
                    # Extra induction is needed.
                    warnings.warn(
                        'Ambiguous data detected, reckoned as scores'
                        ' or label-format data as defaults. Please set '
                        'parms related to `is_onehot` to `True` if '
                        'use one-hot encoding data to compute metrics.')
                    return False, None
                else:
                    raise ValueError(
                        'num_classes should greater than 2 in multi label'
                        'metrics.')
        return False, None

    formated_data = None
    if is_onehot is None:
        is_onehot, formated_data = _induct_is_onehot(data)

    if not is_onehot:
        # convert label-format inputs to one-hot encodings
        formated_data = stack_func(
            [label_to_onehot(sample, num_classes) for sample in data])
    elif is_onehot and formated_data is None:
        # directly stack data if `is_onehot` is set to True without induction
        formated_data = stack_func(data)

    return formated_data


class MultiLabelMixin:
    """A Mixin for Multilabel Metrics to clarify whether the input is one-hot
    encodings or label-format inputs for corner case with minimal user
    awareness."""

    def __init__(self, *args, **kwargs) -> None:
        # pass arguments for multiple inheritances
        super().__init__(*args, **kwargs)  # type: ignore
        self._pred_is_onehot: Optional[bool] = None
        self._label_is_onehot: Optional[bool] = None

    @property
    def pred_is_onehot(self) -> Optional[bool]:
        """Whether prediction is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        return self._pred_is_onehot

    @pred_is_onehot.setter
    def pred_is_onehot(self, is_onehot: Optional[bool]):
        """Set a flag of whether prediction is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        self._pred_is_onehot = is_onehot

    @property
    def label_is_onehot(self) -> Optional[bool]:
        """Whether label is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        return self._label_is_onehot

    @label_is_onehot.setter
    def label_is_onehot(self, is_onehot: Optional[bool]):
        """Set a flag of whether label is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        self._label_is_onehot = is_onehot
