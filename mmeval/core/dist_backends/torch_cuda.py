# Copyright (c) OpenMMLab. All rights reserved.

from typing import TYPE_CHECKING, Any, Tuple, TypeVar, Union

from mmeval.utils import try_import
from .torch_cpu import TorchCPUDist

if TYPE_CHECKING:
    import torch
else:
    torch = try_import('torch')

Tensor = TypeVar('Tensor', bound='torch.Tensor')


class TorchCUDADist(TorchCPUDist):
    """A cuda distributed communication backend for torch.distributed."""

    def __init__(self) -> None:
        super().__init__()
        if torch is None:
            raise ImportError(f'For availability of {self.__class__.__name__},'
                              ' please install pytorch first.')
        if not torch.distributed.is_nccl_available():
            raise RuntimeError(
                f'For availability of {self.__class__.__name__},'
                ' make sure torch.distributed.is_nccl_available().')

    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a cuda tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            tuple: A tuple of the tensor converted from given object and the
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
