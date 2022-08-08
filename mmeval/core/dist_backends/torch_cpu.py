# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pickle
import torch
import torch.distributed as torch_dist
from typing import Any, List, Tuple, TypeVar, Union

from mmeval.core.dist_backends.base_dist import TensorBaseDistributed

Tensor = TypeVar('Tensor', bound=torch.Tensor)


class TorchCPUDistributed(TensorBaseDistributed):
    """A cpu distributed communication backend for torch.distributed."""

    @property
    def is_dist_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
                initialized, else False.
        """
        return torch_dist.is_available() and torch_dist.is_initialized()

    @property
    def rank_id(self) -> int:
        """Returns the rank index of the current process group."""
        return torch_dist.get_rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        return torch_dist.get_world_size()

    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            Tuple: A tuple of the tensor converted from given object and the
                tensor size.
        """
        buffer = pickle.dumps(obj)
        obj_tensor = torch.tensor(np.frombuffer(buffer, dtype=np.int8))
        obj_size_tensor = torch.tensor(len(buffer), dtype=torch.long)
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
        buffer = tensor.numpy().tobytes()[:tensor_size]
        obj = pickle.loads(buffer)
        return obj

    def _pad_tensor(self, tensor: Tensor, max_size: Union[int,
                                                          Tensor]) -> Tensor:
        """Padding the given tensor to the given size with 0.

        Args:
            tensor (Tensor): A tensor-like data to be padded.
            max_size (int or Tensor): The max tensor size that for tensor
                padding.

        Returns:
            Tensor: The padded tensor.
        """
        padding = torch.ones(max_size - tensor.size()[0], dtype=tensor.dtype)
        padding = padding.to(tensor.device)
        padded_tensor = torch.cat([tensor, padding], axis=0)
        return padded_tensor

    def _all_gather(self, tensor: Tensor) -> List[Tensor]:
        """All gather the given tensor.

        Args:
            tensor (Tensor): The tensor for all gather.

        Returns:
            list: A list of the gathered tensor.
        """
        global_tensor_list = [
            torch.empty_like(tensor).to(tensor.device)
            for _ in range(self.world_size)
        ]
        torch_dist.all_gather(global_tensor_list, tensor, group=None)
        return global_tensor_list

    def _broadcast(self, tensor: Tensor, src: int) -> Tensor:
        """Broadcast the given object from the source rank.

        Args:
            tensor (Tensor): The tensor for broadcast.
            src (int): The source rank index.

        Returns:
            Tensor: The broadcast tensor.
        """
        torch_dist.broadcast(tensor, src=src, group=None)
        return tensor
