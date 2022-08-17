# Copyright (c) OpenMMLab. All rights reserved.

import os
from typing import Any, List

from mmeval.core.dist_backends.base_backend import BaseDistBackend

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class MPI4PyDist(BaseDistBackend):
    """A distributed communication backend for mpi4py."""

    def __init__(self) -> None:
        super().__init__()
        assert MPI is not None, \
            f'For availability of {self.__class__.__name__},' \
            ' please install mpi4py first.'

    @property
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
                initialized, otherwise returns False.
        """
        return 'OMPI_COMM_WORLD_SIZE' in os.environ

    @property
    def rank(self) -> int:
        """Returns the rank index of the current process group."""
        comm = MPI.COMM_WORLD
        return comm.Get_rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        comm = MPI.COMM_WORLD
        return comm.Get_size()

    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process..

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """
        comm = MPI.COMM_WORLD
        return comm.allgather(obj)

    def broadcast_object(self, obj: Any, src: int) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """
        comm = MPI.COMM_WORLD
        return comm.bcast(obj, root=src)
