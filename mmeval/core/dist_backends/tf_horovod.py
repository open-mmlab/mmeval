# Copyright (c) OpenMMLab. All rights reserved.

from typing import TYPE_CHECKING, Any, List

from mmeval.utils import try_import
from .base_backend import BaseDistBackend

if TYPE_CHECKING:
    import horovod.tensorflow as hvd
else:
    hvd = try_import('horovod.tensorflow')


class TFHorovodDist(BaseDistBackend):
    """A distributed communication backend for horovod.tensorflow."""

    def __init__(self) -> None:
        super().__init__()
        if hvd is None:
            raise ImportError(f'For availability of {self.__class__.__name__},'
                              ' please install horovod with tensorflow first.')

    @property
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
            initialized, otherwise returns False.
        """
        try:
            hvd.size()
            is_init = True
        except ValueError:
            is_init = False
        return is_init

    @property
    def rank(self) -> int:
        """Returns the rank index of the current process group."""
        return hvd.rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        return hvd.size()

    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process..

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """
        return hvd.allgather_object(obj)

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """
        return hvd.broadcast_object(obj, root_rank=src)
