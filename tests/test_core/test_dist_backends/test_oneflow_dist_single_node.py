# Copyright (c) OpenMMLab. All rights reserved.

import multiprocessing
import os
import pytest

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

from mmeval.core.dist_backends.oneflow_dist import OneFlowDist

flow = pytest.importorskip('oneflow')


def _create_obj_list(world_size):
    obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        obj['data'] = [i for i in range(idx)]
        obj_list.append(obj)
    return obj_list


def _oneflow_dist_all_gather_fn(world_size):
    dist_comm = OneFlowDist()

    assert dist_comm.is_initialized
    assert dist_comm.world_size == world_size

    rank = dist_comm.rank

    obj_list = _create_obj_list(world_size)
    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')

    assert gather_obj_list == obj_list


def _oneflow_dist_broadcast_fn(world_size):
    dist_comm = OneFlowDist()

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


def _subprocess(fn, local_rank, world_size, port=12345):
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_PORT'] = str(port)
    fn(local_rank)


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        pytest.param(
            1,
            2350,
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 0,
                reason='CUDA device count must greater than 0.')),
        pytest.param(
            2,
            2350,
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 2,
                reason='CUDA device count must greater than 1.'))
    ])
def test_all_gather_object(process_num, comm_port):
    p = multiprocessing.Pool(process_num)
    for rank in range(process_num):
        p.apply_async(
            _subprocess,
            args=(_oneflow_dist_all_gather_fn, rank, process_num, comm_port))
    p.close()
    p.join()


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        pytest.param(
            1,
            2350,
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 0,
                reason='CUDA device count must greater than 0.')),
        pytest.param(
            2,
            2350,
            marks=pytest.mark.skipif(
                flow.cuda.device_count() < 2,
                reason='CUDA device count must greater than 1.'))
    ])
def test_broadcast_object(process_num, comm_port):
    p = multiprocessing.Pool(process_num)
    for rank in range(process_num):
        p.apply_async(
            _subprocess,
            args=(_oneflow_dist_broadcast_fn, rank, process_num, comm_port))
    p.close()
    p.join()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
