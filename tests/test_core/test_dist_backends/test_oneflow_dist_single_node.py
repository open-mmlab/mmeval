# Copyright (c) OpenMMLab. All rights reserved.

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
        return flow.all(a == b)
    else:
        return False


def _create_obj_list(world_size):
    obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        obj['data'] = [i for i in range(idx)]
        obj_list.append(obj)
    return obj_list


def _create_tensor_list(rank, world_size, device='cpu'):
    obj_list = []
    rank += 1
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        obj['data'] = [
            flow.tensor([rank * 1.0, rank * 2.0, rank * 3.0], device=device)
        ]
        obj_list.append(obj)
    return obj_list


def _oneflow_dist_all_gather_fn(rank, world_size, comm_port, device):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank
    assert device in ['cpu', 'cuda'], 'only cpu & gpu is supported'

    # cpu
    obj_list = _create_obj_list(world_size)

    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')
    assert equal(gather_obj_list, obj_list)

    # cuda
    obj_list = _create_tensor_list(0, world_size, device)

    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')

    assert equal(gather_obj_list, obj_list)


def _oneflow_dist_broadcast_fn(rank, world_size, comm_port, device):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    assert device in ['cpu', 'cuda'], 'only cpu & gpu is supported'
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
    rank_0_obj = flow.randn(3, 4).to(device)

    print(f'rank {rank}, obj {rank_0_obj}')
    broadcast_obj = dist_comm.broadcast_object(rank_0_obj, src=0)
    print(f'rank {rank}, broadcast_obj {broadcast_obj}')

    assert equal(broadcast_obj, rank_0_obj)


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port', 'device'],
    argvalues=[
        pytest.param(1, 2350, 'cpu'),
        pytest.param(2, 2350, 'cpu'),
        pytest.param(
            1,
            2350,
            'cuda',
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 1,
                reason='CUDA device count must greater than 0.')),
        pytest.param(
            2,
            2350,
            'cuda',
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 2,
                reason='CUDA device count must greater than 1.'))
    ])
def test_all_gather_object(process_num, comm_port, device):
    if process_num < 2:
        _oneflow_dist_all_gather_fn(0, process_num, comm_port, device)
    else:
        file = os.path.join(
            os.path.dirname(__file__),
            'test_oneflow_dist_single_node_multi_processes.py')
        cmd = f'{sys.executable} -m oneflow.distributed.launch \
                --nproc_per_node {process_num}                 \
                --master_port {comm_port}                      \
                {file}                                         \
                --fn all_gather                                \
                --device {device}                              \
                '

        assert os.system(cmd) == 0


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port', 'device'],
    argvalues=[
        pytest.param(1, 2350, 'cpu'),
        pytest.param(2, 2350, 'cpu'),
        pytest.param(
            1,
            2350,
            'cuda',
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 1,
                reason='CUDA device count must greater than 0.')),
        pytest.param(
            2,
            2350,
            'cuda',
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 2,
                reason='CUDA device count must greater than 1.'))
    ])
def test_broadcast_object(process_num, comm_port, device):
    if process_num < 2:
        _oneflow_dist_broadcast_fn(0, 1, comm_port, device)
    else:
        file = os.path.join(
            os.path.dirname(__file__),
            'test_oneflow_dist_single_node_multi_processes.py')
        cmd = f'{sys.executable} -m oneflow.distributed.launch \
                --nproc_per_node {process_num}                 \
                --master_port {comm_port}                      \
                {file}                                         \
                --fn broadcast                                 \
                --device {device}                              \
                '

        assert os.system(cmd) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
