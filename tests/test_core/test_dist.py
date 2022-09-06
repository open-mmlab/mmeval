# Copyright (c) OpenMMLab. All rights reserved.

import importlib
import pytest

import mmeval.core.dist as dist
from mmeval.core.dist_backends.non_dist import NonDist


@pytest.mark.parametrize(
    argnames=('dist_backend_name', 'is_valid'),
    argvalues=[('tf_horovod', True), ('mpi4py', True), ('torch_cpu', True),
               ('torch_cuda', True), ('non_dist', True), ('xxxx', False)])
def test_set_default_dist_backend(dist_backend_name, is_valid):
    if is_valid:
        dist.set_default_dist_backend(dist_backend_name)
        assert dist._DEFAULT_BACKEND == dist_backend_name
    else:
        with pytest.raises(AssertionError):
            dist.set_default_dist_backend(dist_backend_name)


def test_get_dist_backend():
    importlib.reload(dist)
    assert type(dist.get_dist_backend()) is NonDist
    assert type(dist.get_dist_backend('non_dist')) is NonDist

    with pytest.raises(AssertionError):
        dist.get_dist_backend('xxx')


def test_default_dist_backend():
    importlib.reload(dist)
    assert dist._DEFAULT_BACKEND == 'non_dist'


def test_list_all_backend():
    assert dist.list_all_backends() == list(dist._DIST_BACKENDS.keys())


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
