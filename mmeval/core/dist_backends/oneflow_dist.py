# Copyright (c) OpenMMLab. All rights reserved.

import pickle
import numpy as np
from typing import TYPE_CHECKING, Any, List, Tuple, TypeVar, Union

from mmeval.utils import try_import
from .base_backend import BaseDistBackend

if TYPE_CHECKING:
    import oneflow
    import oneflow as flow
    from oneflow.framework.check_point_v2 import _broadcast_py_object
else:
    flow = try_import('oneflow')
    from oneflow.framework.check_point_v2 import _broadcast_py_object

Tensor = TypeVar('Tensor', bound='oneflow.Tensor')

    
class OneFlowDist(BaseDistBackend):
    """A cuda distributed communication backend for oneflow."""

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
        return flow.env.get_local_rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        return flow.env.get_world_size()

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

    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process.

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """
        if isinstance(obj, flow.Tensor):
            return self._all_gather(obj)

        obj_list = []
        for src in range(flow.env.get_world_size()):
            obj_list.append(_broadcast_py_object(obj, src))
        return obj_list

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """
        if isinstance(obj, flow.Tensor):
            return self._broadcast(obj, src)

        return _broadcast_py_object(obj, src)