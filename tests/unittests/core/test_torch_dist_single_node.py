# Copyright (c) OpenMMLab. All rights reserved.

import os

import pytest

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp

from mmeval.core.dist_backends.torch_cpu import TorchCPUDistributed
from mmeval.core.dist_backends.torch_cuda import TorchCUDADistributed

DIST_COMM_BACKENDS = {
    'gloo': TorchCPUDistributed,
    'mpi': TorchCPUDistributed,
    'nccl': TorchCUDADistributed,
}


def _init_torch_dist(rank_id, world_size, comm_backend, port):
    torch_dist.init_process_group(
        backend=comm_backend,
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank_id)

    if comm_backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank_id % num_gpus)


def _create_global_obj_list(world_size):
    global_obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank_id'] = idx
        obj['world_size'] = world_size
        obj['data'] = [i for i in range(idx)]
        global_obj_list.append(obj)
    return global_obj_list


def _torch_dist_all_gather_fn(rank_id, world_size, comm_backend, port):
    _init_torch_dist(rank_id, world_size, comm_backend, port)
    dist_comm_cls = DIST_COMM_BACKENDS[comm_backend]
    dist_comm = dist_comm_cls()

    assert dist_comm.rank_id == rank_id
    assert dist_comm.world_size == world_size

    global_obj_list = _create_global_obj_list(world_size)
    local_obj = global_obj_list[rank_id]
    print(f'rank {rank_id}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank_id}, gather_obj_list {gather_obj_list}')

    assert gather_obj_list == global_obj_list


def _torch_dist_broadcast_fn(rank_id, world_size, comm_backend, port):
    _init_torch_dist(rank_id, world_size, comm_backend, port)
    dist_comm_cls = DIST_COMM_BACKENDS[comm_backend]
    dist_comm = dist_comm_cls()

    assert dist_comm.rank_id == rank_id
    assert dist_comm.world_size == world_size

    rank_0_obj = {'rank_id': 0}

    if rank_id == 0:
        obj = rank_0_obj
    else:
        obj = None

    print(f'rank {rank_id}, obj {obj}')
    broadcast_obj = dist_comm.broadcast_object(obj, src=0)
    print(f'rank {rank_id}, broadcast_obj {broadcast_obj}')

    assert broadcast_obj == rank_0_obj


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[(1, 2345), [4, 2345]])
def test_gloo_all_gather_object(process_num, comm_port):
    comm_backend = 'gloo'
    mp.spawn(
        _torch_dist_all_gather_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_mpi_available(), 
    reason="MPI backend is not available."
)
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[(1, 2345), [4, 2345]])
def test_mpi_all_gather_object(process_num, comm_port):
    comm_backend = 'mpi'
    mp.spawn(
        _torch_dist_all_gather_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_nccl_available(), 
    reason="NCCL backend is not available."
)
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, 
    reason="CUDA device count must greater than 2."
)
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[(1, 2345), [2, 2347]])
def test_nccl_all_gather_object(process_num, comm_port):
    comm_backend = 'nccl'
    mp.spawn(
        _torch_dist_all_gather_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[(1, 2345), [4, 2348]])
def test_gloo_broadcast_object(process_num, comm_port):
    comm_backend = 'gloo'
    mp.spawn(
        _torch_dist_broadcast_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_mpi_available(), 
    reason="MPI backend is not available."
)
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[(1, 2345), [4, 2349]])
def test_mpi_broadcast_object(process_num, comm_port):
    comm_backend = 'mpi'
    mp.spawn(
        _torch_dist_broadcast_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_nccl_available(), 
    reason="NCCL backend is not available."
)
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, 
    reason="CUDA device count must greater than 2."
)
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[(1, 2345), [2, 2350]])
def test_nccl_broadcast_object(process_num, comm_port):
    comm_backend = 'nccl'
    mp.spawn(
        _torch_dist_broadcast_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


if __name__ == '__main__':
    pytest.main(['--capture=no', '--continue-on-collection-errors'])
