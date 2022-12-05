# Copyright (c) OpenMMLab. All rights reserved.

import os, sys
import pytest
import numpy as np

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

from mmeval.core.dist_backends.oneflow_dist import OneFlowDist

flow = pytest.importorskip('oneflow')

def equal(a, b):
    if isinstance(a, dict):
        return all(equal(a[k], b[k]) for k in a.keys())
    elif isinstance(a, (list, tuple)):
        return all(equal(ai,bi) for ai,bi in zip(a,b))
    elif isinstance(a, (int, float, bool, str)):
        return a==b
    elif isinstance(a, flow.Tensor):
        return np.all(a.numpy()==b.numpy())
    else:
        return False

def _create_obj_list(world_size):
    obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        obj['data'] = [i for i in range(idx + 1)]
        obj_list.append(obj)
    return obj_list

def _create_tensor_list(rank, world_size, device="cpu"):
    obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        idx += 1
        obj['data'] = [flow.tensor([idx * 1.0, idx * 2.0, idx * 3.0], device=device)]
        obj_list.append(obj)
    return obj_list

def _oneflow_dist_all_gather_fn(rank, world_size, device):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    # cpu
    obj_list = _create_obj_list(world_size)

    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')
    assert equal(gather_obj_list, obj_list)

    # cuda
    obj_list = _create_tensor_list(rank, world_size, device)
    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')
    
    assert equal(gather_obj_list, obj_list), f"{gather_obj_list}, {obj_list}"


def _oneflow_dist_broadcast_fn(rank, world_size, device):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    # cpu
    rank_0_obj = {'rank': 0}

    if rank == 0:
        obj = rank_0_obj
    else:
        obj = None

    print(f'rank {rank}, obj {obj}')
    broadcast_obj = dist_comm.broadcast_object(obj, src=0)
    print(f'rank {rank}, broadcast_obj {broadcast_obj}')

    assert equal(broadcast_obj, rank_0_obj)

    # cuda
    rank_0_obj = flow.randn(3,4).to(device)

    print(f'rank {rank}, obj {rank_0_obj}')
    broadcast_obj = dist_comm.broadcast_object(rank_0_obj, src=0)
    print(f'rank {rank}, broadcast_obj {broadcast_obj}')

    assert equal(broadcast_obj, rank_0_obj)


if __name__ == "__main__":
    fn = sys.argv[2]
    device = sys.argv[4]
    assert fn in ["all_gather", "broadcast"]
    assert device in ["cpu", "cuda"]
    rank = flow.env.get_local_rank()
    world_size = flow.env.get_world_size()

    device = f"{device}:{rank}"
    if fn=="all_gather":
        _oneflow_dist_all_gather_fn(rank, world_size, device)
    else:
        _oneflow_dist_broadcast_fn(rank, world_size, device)

