# Copyright (c) OpenMMLab. All rights reserved.

import os
import pytest

# check if current process is launch via mpirun
if os.environ.get('OMPI_COMM_WORLD_SIZE', '0') != '0':
    pytest.skip(allow_module_level=True)

from mmeval.core.dist_backends.non_dist import NonDist


def test_non_distributed():
    dist_comm = NonDist()
    assert not dist_comm.is_initialized

    assert dist_comm.rank == 0
    assert dist_comm.world_size == 1

    test_obj = {'test': 1}

    assert dist_comm.all_gather_object(test_obj) == [test_obj]
    assert dist_comm.broadcast_object(test_obj, src=0) == test_obj
    assert dist_comm.broadcast_object(test_obj) == test_obj


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
