# Copyright (c) OpenMMLab. All rights reserved.

from typing import TYPE_CHECKING, Any, List, TypeVar

from mmeval.utils import try_import
from .base_backend import BaseDistBackend

if TYPE_CHECKING:
    import oneflow
    import oneflow as flow
    import oneflow.framework.check_point_v2 as check_point_v2
else:
    flow = try_import('oneflow')
    check_point_v2 = try_import('oneflow.framework.check_point_v2')

Tensor = TypeVar('Tensor', bound='oneflow.Tensor')


class OneFlowDist(BaseDistBackend):
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

    def _all_gather_py_obj(self, obj: Any) -> List[Any]:
        """All gather the given python object.

        Args:
            obj (Any): The python object for all gather.

        Returns:
            list: A list of the gathered objects.
        """
        obj_list = []
        for src in range(flow.env.get_world_size()):
            new_obj = check_point_v2._broadcast_py_object(obj, src)
            obj_list.append(new_obj)
        return obj_list

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
            shape = list(obj.shape)
            shapes = self._all_gather_py_obj(shape)
            flag = all(s == shapes[0] for s in shapes[1:])
            if flag:
                return self._all_gather(obj)
            nps = self._all_gather_py_obj(obj.numpy())
            return [flow.tensor(np) for np in nps]

        return self._all_gather_py_obj(obj)

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

        return check_point_v2._broadcast_py_object(obj, src)
