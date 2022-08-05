# Copyright (c) OpenMMLab. All rights reserved.

import torch
from typing import Any, Tuple, TypeVar, Union

from mmeval.core.dist_backends.torch_cpu import TorchCPUDistributed

Tensor = TypeVar('Tensor', bound=torch.Tensor)


class TorchCUDADistributed(TorchCPUDistributed):
    """A cuda distributed communication backend for torch.distributed."""

    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a cuda tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            Tuple: A tuple of the tensor converted from given object and the
                tensor size.
        """
        # Add type annotation make mypy happy
        obj_tensor: Tensor
        obj_size_tensor: Tensor
        obj_tensor, obj_size_tensor = super()._object_to_tensor(obj)
        return obj_tensor.cuda(), obj_size_tensor.cuda()

    def _tensor_to_object(self, tensor: Tensor,
                          tensor_size: Union[int, Tensor]) -> Any:
        """Convert the given cuda tensor to a object via `pickle.loads`.

        Args:
            tenosr (Tensor): A cuda tensor.
            tensor_size (int or Tensor): The tensor size of the given Tensor to
                be convert object.

        Returns:
            Any: The object converted from the given cuda tensor.
        """
        return super()._tensor_to_object(tensor.detach().cpu(), tensor_size)
