# Copyright (c) OpenMMLab. All rights reserved.

import os
import pytest

from mmeval.core.dist_backends.npu_dist import NPUDist

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

torch_npu = pytest.importorskip('torch_npu')

torch = pytest.importorskip('torch')
torch_dist = pytest.importorskip('torch.distributed')
mp = pytest.importorskip('torch.multiprocessing')


def _init_torch_dist(rank, world_size, comm_backend, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    torch_dist.init_process_group(
        backend=comm_backend,
        init_method='env://',
        world_size=world_size,
        rank=rank)

    if comm_backend == 'hccl':
        num_gpus = torch.npu.device_count()
        torch.npu.set_device(rank % num_gpus)


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
    dist_comm = NPUDist()

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
    dist_comm = NPUDist()

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


@pytest.mark.skipif(
    not torch_dist.is_hccl_available(),
    reason='HCCL backend is not available.')
@pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason='NPU device count must greater than 0.')
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        pytest.param(
            1,
            2347,
            marks=pytest.mark.skipif(
                torch.npu.device_count() < 1,
                reason='npu device count must greater than 0.')),
        pytest.param(
            2,
            2347,
            marks=pytest.mark.skipif(
                torch.npu.device_count() < 2,
                reason='NPU device count must greater than 2.'))
    ])
def test_hccl_all_gather_object(process_num, comm_port):
    comm_backend = 'hccl'
    mp.spawn(
        _torch_dist_all_gather_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


@pytest.mark.skipif(
    not torch_dist.is_hccl_available(),
    reason='HCCL backend is not available.')
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port'],
    argvalues=[
        pytest.param(
            1,
            2350,
            marks=pytest.mark.skipif(
                torch.npu.device_count() < 1,
                reason='npu device count must greater than 0.')),
        pytest.param(
            2,
            2350,
            marks=pytest.mark.skipif(
                torch.npu.device_count() < 2,
                reason='NPU device count must greater than 2.'))
    ])
def test_hccl_broadcast_object(process_num, comm_port):
    comm_backend = 'hccl'
    mp.spawn(
        _torch_dist_broadcast_fn,
        nprocs=process_num,
        args=(process_num, comm_backend, comm_port))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
