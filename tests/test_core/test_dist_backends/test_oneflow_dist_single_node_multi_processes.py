# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import os
import pytest
import sys

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

from mmeval.core.dist_backends.oneflow_dist import OneFlowDist

flow = pytest.importorskip('oneflow')


def equal(a, b):
    if isinstance(a, dict):
        return all(equal(a[k], b[k]) for k in a.keys())
    elif isinstance(a, (list, tuple)):
        return all(equal(ai, bi) for ai, bi in zip(a, b))
    elif isinstance(a, (int, float, bool, str)):
        return a == b
    elif isinstance(a, flow.Tensor):
        return np.all(a.numpy() == b.numpy())
    else:
        return False


def _create_obj_list(rank, world_size, device):
    obj_list = []
    for idx in range(world_size):
        rank = idx + 1
        obj = dict()
        obj['rank'] = idx
        obj['ranks'] = list(range(world_size))
        obj['world_size'] = world_size
        obj['data'] = [
            flow.tensor([rank * 1.0, rank * 2.0, rank * 3.0], device=device)
        ]
        obj_list.append(obj)
    return obj_list


def _oneflow_dist_all_gather_fn(rank, world_size, device):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    obj_list = _create_obj_list(rank, world_size, device)

    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')
    assert equal(gather_obj_list, obj_list)


def _oneflow_dist_broadcast_fn(rank, world_size, device):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    obj_list = _create_obj_list(rank, world_size, device)

    local_obj = obj_list[rank]

    print(f'rank {rank}, obj {local_obj}')
    broadcast_obj = dist_comm.broadcast_object(local_obj, src=0)
    print(f'rank {rank}, broadcast_obj {broadcast_obj}')

    assert equal(broadcast_obj, obj_list[0])


if __name__ == '__main__':
    fn = sys.argv[2]
    device = sys.argv[4]
    assert fn in ['all_gather', 'broadcast']
    assert device in ['cpu', 'cuda']
    rank = flow.env.get_local_rank()
    world_size = flow.env.get_world_size()

    device = f'{device}:{rank}'
    if fn == 'all_gather':
        _oneflow_dist_all_gather_fn(rank, world_size, device)
    else:
        _oneflow_dist_broadcast_fn(rank, world_size, device)
