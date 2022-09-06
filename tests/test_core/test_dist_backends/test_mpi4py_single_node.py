# Copyright (c) OpenMMLab. All rights reserved.

import os
import pytest

# check if current process is launch via mpirun
# eg. `mpirun -np 2 pytest -v --capture=no --with-mpi tests/test_core/dist_backends`  # noqa: E501
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') == '0':
    pytest.skip(allow_module_level=True)

from mmeval.core.dist_backends.mpi4py import MPI4PyDist


def _create_obj_list(world_size):
    obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        obj['data'] = [i for i in range(idx)]
        obj_list.append(obj)
    return obj_list


@pytest.mark.mpi
def test_mpi4py_all_gather_fn():

    dist_comm = MPI4PyDist()
    assert dist_comm.is_initialized

    rank = dist_comm.rank
    world_size = dist_comm.world_size

    obj_list = _create_obj_list(world_size)
    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')

    assert gather_obj_list == obj_list


@pytest.mark.mpi
def test_mpi4py_broadcast_fn():

    dist_comm = MPI4PyDist()
    assert dist_comm.is_initialized

    rank = dist_comm.rank

    rank_0_obj = {'rank': 0}

    if rank == 0:
        obj = rank_0_obj
    else:
        obj = None

    print(f'rank {rank}, obj {obj}')
    broadcast_obj = dist_comm.broadcast_object(obj, src=0)
    print(f'rank {rank}, broadcast_obj {broadcast_obj}')

    assert broadcast_obj == rank_0_obj


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no', '--with-mpi'])
