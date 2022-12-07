# Copyright (c) OpenMMLab. All rights reserved.

import os
import pytest

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

# skip torch.distributed test in Windnows
if os.name == 'nt':
    pytest.skip(allow_module_level=True)

from mmeval.core.dist_backends.torch_cpu import TorchCPUDist
from mmeval.core.dist_backends.torch_cuda import TorchCUDADist

torch = pytest.importorskip('torch')
torch_dist = pytest.importorskip('torch.distributed')
mp = pytest.importorskip('torch.multiprocessing')

DIST_COMM_BACKENDS = {
    'gloo': TorchCPUDist,
    'mpi': TorchCPUDist,
    'nccl': TorchCUDADist,
}


def _init_torch_dist(rank, world_size, comm_backend, port):
    torch_dist.init_process_group(
        backend=comm_backend,
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank)

    if comm_backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)


def _create_obj_list(world_size):
    obj_list = []
    for idx in range(world_size):
        obj = dict()
        obj['rank'] = idx
        obj['world_size'] = world_size
        obj['data'] = [i for i in range(idx)]
        obj_list.append(obj)
    return obj_list


def _torch_dist_all_gather_fn(rank, world_size, comm_backend, port):
    _init_torch_dist(rank, world_size, comm_backend, port)
    dist_comm_cls = DIST_COMM_BACKENDS[comm_backend]
    dist_comm = dist_comm_cls()

    assert dist_comm.is_initialized
    assert dist_comm.rank == rank
    assert dist_comm.world_size == world_size

    obj_list = _create_obj_list(world_size)
    local_obj = obj_list[rank]
    print(f'rank {rank}, local_obj {local_obj}')

    gather_obj_list = dist_comm.all_gather_object(local_obj)
    print(f'rank {rank}, gather_obj_list {gather_obj_list}')

    assert gather_obj_list == obj_list


def _torch_dist_broadcast_fn(rank, world_size, comm_backend, port):
    _init_torch_dist(rank, world_size, comm_backend, port)
    dist_comm_cls = DIST_COMM_BACKENDS[comm_backend]
    dist_comm = dist_comm_cls()

    assert dist_comm.is_initialized
    assert dist_comm.rank == rank
    assert dist_comm.world_size == world_size

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
    argnames=['process_num', 'comm_port'], argvalues=[(1, 2345), (4, 2345)])
def test_gloo_all_gather_object(process_num, comm_port):
    comm_backend = 'gloo'
    mp.spawn(
        _torch_dist_all_gather_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_mpi_available(), reason='MPI backend is not available.')
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'], argvalues=[(1, 2346), (4, 2346)])
def test_mpi_all_gather_object(process_num, comm_port):
    comm_backend = 'mpi'
    mp.spawn(
        _torch_dist_all_gather_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


try:
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        MAJOR, MINOR, PATCH = nccl_version
        nccl_version = MAJOR * 1000 + MINOR * 100 + PATCH
except Exception:
    nccl_version = 0


@pytest.mark.skipif(
    not torch_dist.is_nccl_available(),
    reason='NCCL backend is not available.')
@pytest.mark.skipif(
    torch.cuda.device_count() < 1,
    reason='CUDA device count must greater than 0.')
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        (1, 2347),
        pytest.param(
            2,
            2347,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2 and nccl_version >= 2500,
                reason='Multi ranks in one GPU is not allowed since NCCL 2.5'))
    ])
def test_nccl_all_gather_object(process_num, comm_port):
    comm_backend = 'nccl'
    mp.spawn(
        _torch_dist_all_gather_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'], argvalues=[(1, 2348), (4, 2348)])
def test_gloo_broadcast_object(process_num, comm_port):
    comm_backend = 'gloo'
    mp.spawn(
        _torch_dist_broadcast_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_mpi_available(), reason='MPI backend is not available.')
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'], argvalues=[(1, 2349), (4, 2349)])
def test_mpi_broadcast_object(process_num, comm_port):
    comm_backend = 'mpi'
    mp.spawn(
        _torch_dist_broadcast_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_nccl_available(),
    reason='NCCL backend is not available.')
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        pytest.param(
            1,
            2350,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 1,
                reason='CUDA device count must greater than 0.')),
        pytest.param(
            2,
            2350,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2 and nccl_version >= 2500,
                reason='Multi ranks in one GPU is not allowed since NCCL 2.5'))
    ])
def test_nccl_broadcast_object(process_num, comm_port):
    comm_backend = 'nccl'
    mp.spawn(
        _torch_dist_broadcast_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
