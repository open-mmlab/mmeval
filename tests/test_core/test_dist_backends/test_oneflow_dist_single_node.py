# Copyright (c) OpenMMLab. All rights reserved.

import multiprocessing as mp
import numpy as np
import os
import pytest

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


def _create_obj_list(rank, world_size):
    obj_list = []
    for idx in range(world_size):
        rank = idx + 1
        obj = dict()
        obj['rank'] = idx
        obj['ranks'] = list(range(world_size))
        obj['world_size'] = world_size
        obj['data'] = [flow.tensor([rank * 1.0, rank * 2.0, rank * 3.0])]
        obj_list.append(obj)
    return obj_list


def _oneflow_dist_all_gather_fn(rank, world_size):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    obj_list = _create_obj_list(rank, world_size)

    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')
    assert equal(gather_obj_list, obj_list)


def _oneflow_dist_broadcast_fn(rank, world_size):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    obj_list = _create_obj_list(rank, world_size)

    local_obj = obj_list[rank]

    print(f'rank {rank}, obj {local_obj}')
    broadcast_obj = dist_comm.broadcast_object(local_obj, src=0)
    print(f'rank {rank}, broadcast_obj {broadcast_obj}')

    assert equal(broadcast_obj, obj_list[0])


def _init_oneflow_dist(local_rank, world_size, port):
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = '127.0.0.1'


def _reset_dist_env():
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'


def _launch_dist_fn(target_fn, process_num, comm_port):
    ctx = mp.get_context('spawn')
    process_list = []
    for rank in range(process_num):
        _init_oneflow_dist(rank, process_num, comm_port)
        p = ctx.Process(target=target_fn, args=(rank, process_num))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    # reset the env variable to prevent getting stuck when importing oneflow
    _reset_dist_env()


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        (1, 2350),
        (2, 2350),
        (4, 2350),
    ])
def test_broadcast_object(process_num, comm_port):
    _launch_dist_fn(_oneflow_dist_broadcast_fn, process_num, comm_port)


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        (1, 2350),
        (2, 2350),
        (4, 2350),
    ])
def test_all_gather_object(process_num, comm_port):
    _launch_dist_fn(_oneflow_dist_all_gather_fn, process_num, comm_port)


if __name__ == '__main__':
    pytest.main([__file__, '-vvv', '--capture=no'])
