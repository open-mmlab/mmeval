# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Optional, no_type_check

from mmeval.core.dist_backends.base_dist import BaseDistributed
from mmeval.core.dist_backends.mpi4py import MPI4PyDistributed
from mmeval.core.dist_backends.non_dist import NonDistributed
from mmeval.core.dist_backends.tf_horovod import TFHorovodDistributed
from mmeval.core.dist_backends.torch_cpu import TorchCPUDistributed
from mmeval.core.dist_backends.torch_cuda import TorchCUDADistributed

DIST_BACKENDS = {
    'non_dist': NonDistributed,
    'mpi4py': MPI4PyDistributed,
    'tf_horovod': TFHorovodDistributed,
    'torch_cpu': TorchCPUDistributed,
    'torch_cuda': TorchCUDADistributed,
}

DEFAULT_BACKEND = 'non_dist'


def list_all_backends() -> List[str]:
    """Returns a list of all distributed backend names.

    Returns:
        List[str]: A list of all distributed backend names.
    """
    return list(DIST_BACKENDS.keys())


def set_default_dist_backend(dist_backend: str) -> None:
    """Set the given distributed backend as the default distributed backend.

    Args:
        dist_backend (str): The distribute backend name to set.
    """
    assert dist_backend in DIST_BACKENDS
    global DEFAULT_BACKEND
    DEFAULT_BACKEND = dist_backend


@no_type_check
def get_dist_backend(dist_backend: Optional[str] = None) -> BaseDistributed:
    """Returns distributed backend by the given distributed backend name.

    Args:
        dist_backend (str, optional): The distributed backend name want to get.
            if None, return the default distributed backend.

    Returns:
        :obj:`BaseDistributed`: The distributed backend instance.
    """
    if dist_backend is None:
        dist_backend = DEFAULT_BACKEND
    assert dist_backend in DIST_BACKENDS
    dist_backend_cls = DIST_BACKENDS[dist_backend]
    return dist_backend_cls()
