# Copyright (c) OpenMMLab. All rights reserved.

from mpi4py import MPI

from mmeval.core.dist_backends.base_dist import BaseDistributed


class MPI4PyDistributed(BaseDistributed):

    @property
    def rank_id(self) -> int:
        comm = MPI.COMM_WORLD
        return comm.Get_rank()
    
    @property
    def world_size(self) -> int:
        comm = MPI.COMM_WORLD
        return comm.Get_size()
    
    def all_gather_object(self, obj):
        comm = MPI.COMM_WORLD
        return comm.allgather(obj)

    def broadcast_object(self, obj, src):
        comm = MPI.COMM_WORLD
        return comm.bcast(obj, root=src)