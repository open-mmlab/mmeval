# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from mmeval.core.base_metric import BaseMetric

try:
    import torch
except ImportError:
    torch = None


class Mertic(BaseMetric):

    def add(self, num):
        self._results.append(num)

    def compute_metric(self, results):
        return {'results': results}


def test_dataset_meta():
    datset_meta = {'CLASSES': ['test1', 'test2']}

    metric = Mertic(dataset_meta=None)
    assert metric.dataset_meta is None

    metric.dataset_meta = datset_meta
    assert metric.dataset_meta == datset_meta


def test_metric_reset():
    metric = Mertic()
    metric.add(1)
    assert len(metric._results) == 1

    metric.reset()
    assert len(metric._results) == 0


def test_metric_call():
    metric = Mertic()
    results = metric(1)
    assert results == {'results': [1]}

    metric.add(2)

    # stateless call
    results = metric(1)
    assert results == {'results': [1]}


def test_metric_compute():
    metric = Mertic()

    for i in range(10):
        metric.add(i)

    results = metric.compute()
    assert results == {'results': [i for i in range(10)]}


def _init_torch_dist(rank, world_size, comm_backend, port):
    torch.distributed.init_process_group(
        backend=comm_backend,
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank)

    if comm_backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)


def _test_metric_compute(rank, world_size, port, dist_merge_method):
    _init_torch_dist(rank, world_size, comm_backend='gloo', port=port)

    metric = Mertic(
        dist_backend='torch_cpu', dist_merge_method=dist_merge_method)

    if dist_merge_method == 'unzip':
        data_slice = range(rank, 10 * world_size, world_size)
    else:
        data_slice = range(rank * 10, (rank + 1) * 10)

    for i in data_slice:
        metric.add(i)

    results = metric.compute()
    assert results == {'results': [i for i in range(10 * world_size)]}

    if world_size == 1:
        return

    if dist_merge_method == 'unzip':
        results = metric.compute(size=(10 * world_size - 1))
        assert results == {'results': [i for i in range(10 * world_size - 1)]}
    else:
        results = metric.compute(size=(10 * world_size - 1))
        assert results == {'results': [i for i in range(10 * world_size - 1)]}


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason='torch.distributed is not available!')
@pytest.mark.parametrize(
    argnames=['process_num', 'comm_port', 'dist_merge_method'],
    argvalues=[(1, 2346, 'unzip'), (4, 2346, 'unzip'), (4, 2346, 'cat')])
def test_metric_compute_dist(process_num, comm_port, dist_merge_method):
    torch.multiprocessing.spawn(
        _test_metric_compute,
        nprocs=process_num,
        args=(process_num, comm_port, dist_merge_method))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
