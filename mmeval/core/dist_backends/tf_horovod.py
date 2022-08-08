# Copyright (c) OpenMMLab. All rights reserved.

import horovod.tensorflow as hvd
from typing import Any, List

from mmeval.core.dist_backends.base_dist import BaseDistributed


class TFHorovodDistributed(BaseDistributed):
    """A distributed communication backend for horovod.tensorflow."""

    @property
    def is_dist_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.
        
        Returns:
            bool: Returns True if the distributed environment has been 
                initialized, else False.
        """
        try:
            hvd.size()
            is_init = True
        except ValueError:
            is_init = False 
        return is_init

    @property
    def rank_id(self) -> int:
        """Returns the rank index of the current process group."""
        return hvd.rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        return hvd.size()

    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        return as a list.

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """
        return hvd.allgather_object(obj)

    def broadcast_object(self, obj: Any, src: int) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """
        return hvd.broadcast_object(obj, root_rank=src)
