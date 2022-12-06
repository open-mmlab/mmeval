# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pickle
from typing import TYPE_CHECKING, Any, List, Tuple, TypeVar, Union

from mmeval.utils import try_import
from .base_backend import TensorBaseDistBackend

if TYPE_CHECKING:
    import oneflow
    import oneflow as flow
    import oneflow.framework.check_point_v2 as check_point_v2
else:
    flow = try_import('oneflow')
    check_point_v2 = try_import('oneflow.framework.check_point_v2')

Tensor = TypeVar('Tensor', bound='oneflow.Tensor')


class OneFlowDist(TensorBaseDistBackend):
    """A distributed communication backend for oneflow."""

    def __init__(self) -> None:
        super().__init__()
        if flow is None:
            raise ImportError(f'For availability of {self.__class__.__name__},'
                              ' please install oneflow first.')

    @property
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
            initialized, otherwise returns False.
        """
        try:
            flow.env.get_world_size()
            is_init = True
        except ValueError:
            is_init = False
        return is_init

    @property
    def rank(self) -> int:
        """Returns the rank index of the current process group."""
        return flow.env.get_rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        return flow.env.get_world_size()

    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            Tuple: A tuple of the tensor converted from given object and the
            tensor size.
        """
        buffer = pickle.dumps(obj)
        storage = np.frombuffer(buffer, dtype=np.int8)
        obj_tensor = flow.tensor(storage)
        obj_size_tensor = flow.tensor([obj_tensor.numel()])
        return obj_tensor, obj_size_tensor

    def _tensor_to_object(self, tensor: Tensor,
                          tensor_size: Union[int, Tensor]) -> Any:
        """Convert the given Tensor to a object via `pickle.loads`.

        Args:
            tenosr (Tensor): A tensor-like data.
            tensor_size (int or Tensor): The tensor size of the given Tensor to
                be convert object.

        Returns:
            Any: The object converted from the given tensor.
        """
        size = int(tensor_size)
        buffer = tensor.cpu().numpy().tobytes()[:size]
        obj = pickle.loads(buffer)
        return obj

    def _pad_tensor(self, tensor: Tensor,
                    max_size: Union[int, Tensor]) -> Tensor:  # yapf: disable
        """Padding the given tensor to the given size.

        Args:
            tensor (Tensor): A tensor-like data to be padded.
            max_size (int or Tensor): The max tensor size that for tensor
                padding.

        Returns:
            Tensor: The padded tensor.
        """
        max_size = int(max_size)
        padding = flow.zeros((max_size - tensor.numel(), ),
                             dtype=flow.int8,
                             device=tensor.device)
        tensor = flow.cat((tensor, padding), dim=0)
        return tensor

    def _all_gather(self, tensor: Tensor) -> List[Tensor]:
        """All gather the given tensor.

        Args:
            tensor (Tensor): The tensor for all gather.

        Returns:
            list: A list of the gathered tensor.
        """
        tensor_list = [
            flow.empty_like(tensor).to(tensor.device)
            for _ in range(self.world_size)
        ]
        flow.comm.all_gather(tensor_list, tensor)
        return tensor_list

    def _broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
        """Broadcast the given object from the source rank.

        Args:
            tensor (Tensor): The tensor for broadcast.
            src (int): The source rank index.

        Returns:
            Tensor: The broadcast tensor.
        """
        flow.comm.broadcast(tensor, src=src)
        return tensor
