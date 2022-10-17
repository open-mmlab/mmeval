# Copyright (c) OpenMMLab. All rights reserved.

import os
import pytest

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

from mmeval.core.dist_backends.paddle_dist import PaddleDist

paddle = pytest.importorskip('paddle')
paddle_dist = pytest.importorskip('paddle.distributed')


def _create_obj_list(world_size):
    obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        obj['data'] = [i for i in range(idx)]
        obj_list.append(obj)
    return obj_list


def _init_dist_env(comm_backend):
    if comm_backend == 'gloo':
        paddle.set_device('cpu')
    try:
        paddle_dist.init_parallel_env()
    except Exception as e:
        if comm_backend == 'nccl':
            # NCCL maybe not be installed successfully so we skip this test.
            print(e)
            return
        else:
            raise e


def _paddle_dist_all_gather_fn(world_size, comm_backend):
    _init_dist_env(comm_backend)
    dist_comm = PaddleDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    obj_list = _create_obj_list(world_size)
    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')

    assert gather_obj_list == obj_list


def _paddle_dist_broadcast_fn(world_size, comm_backend):
    _init_dist_env(comm_backend)
    dist_comm = PaddleDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

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


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_backend'],
    argvalues=[
        (1, 'gloo'), (2, 'gloo'),
        pytest.param(
            2,
            'nccl',
            marks=pytest.mark.skipif(
                (not paddle.fluid.core.is_compiled_with_cuda()
                 or paddle.fluid.core.get_cuda_device_count() < 2),
                reason='Multi ranks in one GPU is not allowed since NCCL 2.5'))
    ])
def test_all_gather_object(process_num, comm_backend):
    paddle_dist.spawn(
        _paddle_dist_all_gather_fn,
        nprocs=process_num,
        backend=comm_backend,
        args=(process_num, comm_backend))


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_backend'],
    argvalues=[
        (1, 'gloo'), (2, 'gloo'),
        pytest.param(
            2,
            'nccl',
            marks=pytest.mark.skipif(
                (not paddle.fluid.core.is_compiled_with_cuda()
                 or paddle.fluid.core.get_cuda_device_count() < 2),
                reason='Multi ranks in one GPU is not allowed since NCCL 2.5'))
    ])
def test_broadcast_object(process_num, comm_backend):
    paddle_dist.spawn(
        _paddle_dist_broadcast_fn,
        nprocs=process_num,
        backend=comm_backend,
        args=(process_num, comm_backend))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
