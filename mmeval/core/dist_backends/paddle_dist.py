# Copyright (c) OpenMMLab. All rights reserved.

import io
import numpy as np
import pickle
from typing import TYPE_CHECKING, Any, List, Tuple, TypeVar, Union

from mmeval.utils import try_import
from .base_backend import TensorBaseDistBackend

if TYPE_CHECKING:
    import paddle
    import paddle.distributed as paddle_dist
else:
    paddle = try_import('paddle')
    paddle_dist = try_import('paddle.distributed')

Tensor = TypeVar('Tensor', bound='paddle.Tensor')


class PaddleDist(TensorBaseDistBackend):
    """A distributed communication backend for paddle.distributed."""

    def __init__(self) -> None:
        super().__init__()
        if paddle is None:
            raise ImportError(f'For availability of {self.__class__.__name__},'
                              ' please install paddle first.')

    @property
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
            initialized, otherwise returns False.
        """
        # NOTE: The `paddle_dist.parallel.parallel_helper._is_parallel_ctx_initialized()`  # noqa: E501
        # API is not work when init parallel env with gloo backend. So we just
        # return True if use gloo backend (CPU only).
        place = paddle.fluid.framework._current_expected_place()
        if isinstance(place, paddle.fluid.core.CPUPlace):
            return True
        return paddle_dist.parallel.parallel_helper._is_parallel_ctx_initialized()  # noqa: E501 # yapf: disable

    @property
    def rank(self) -> int:
        """Returns the rank index of the current process group."""
        return paddle_dist.get_rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        return paddle_dist.get_world_size()

    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a tensor via `pickle.dumps`.

        Modified from: https://github.com/PaddlePaddle/Paddle/blob/
        264ad2055fdeb6c9cfce0ef2f0bd38641aae00a4/python/paddle/distributed/
        collective.py#L1069

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            Tuple: A tuple of the tensor converted from given object and the
            tensor size.
        """
        _pickler = pickle.Pickler
        f = io.BytesIO()
        _pickler(f).dump(obj)
        data = np.frombuffer(f.getvalue(), dtype=np.uint8)
        obj_tensor = paddle.to_tensor(data)
        # NOTE: Many ops in paddle are not implemented for 'uint8'.
        # So we cast to 'int32' here.
        # TODO: We should remove this data type cast once all ops that we used
        #  have been implemented for 'uint8'.
        obj_tensor = paddle.cast(obj_tensor, 'int32')
        return obj_tensor, obj_tensor.numel()

    def _tensor_to_object(self, tensor: Tensor,
                          tensor_size: Union[int, Tensor]) -> Any:
        """Convert the given Tensor to a object via `pickle.loads`.

        Modified from: https://github.com/PaddlePaddle/Paddle/blob/
        264ad2055fdeb6c9cfce0ef2f0bd38641aae00a4/python/paddle/distributed/
        collective.py#L1078

        Args:
            tenosr (Tensor): A tensor-like data.
            tensor_size (int or Tensor): The tensor size of the given Tensor to
                be converted object.

        Returns:
            Any: The object converted from the given tensor.
        """
        # NOTE: Since we cast tensor from 'unit8' to 'int32', we should cast
        # back to 'uint8'.
        tensor = paddle.cast(tensor, 'uint8')
        _unpickler = pickle.Unpickler
        return _unpickler(io.BytesIO(tensor.numpy()[:tensor_size])).load()

    def _pad_tensor(self, tensor: Tensor,
                    max_size: Union[int, Tensor]) -> Tensor:  # yapf: disable
        """Padding the given tensor to the given size.

        Modified from: https://github.com/PaddlePaddle/Paddle/blob/
        264ad2055fdeb6c9cfce0ef2f0bd38641aae00a4/python/paddle/distributed/
        collective.py#L1129

        Args:
            tensor (Tensor): A tensor-like data to be padded.
            max_size (int or Tensor): The max tensor size that for tensor
                padding.

        Returns:
            Tensor: The padded tensor.
        """
        numpy_data = tensor.numpy()
        padded_numpy_data = np.resize(numpy_data, [int(max_size)])
        padded_tensor = paddle.to_tensor(padded_numpy_data)
        return padded_tensor

    def _all_gather(self, tensor: Tensor) -> List[Tensor]:
        """All gather the given tensor.

        Args:
            tensor (Tensor): The tensor for all gather.

        Returns:
            list: A list of the gathered tensor.
        """
        # NOTE: The value of world size should be >=2 when invoking
        # `paddle_dist.all_gather`.
        if self.world_size < 2:
            return [tensor]
        tensor_list: List[Tensor] = []
        paddle_dist.all_gather(tensor_list, tensor)
        return tensor_list

    def _broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
        """Broadcast the given object from the source rank.

        Args:
            tensor (Tensor): The tensor for broadcast.
            src (int): The source rank index.

        Returns:
            Tensor: The broadcast tensor.
        """
        # NOTE: The value of world size should be >=2 when invoke
        # `paddle_dist.broadcast`.
        if self.world_size < 2:
            return tensor
        paddle_dist.broadcast(tensor, src=src)
        return tensor
