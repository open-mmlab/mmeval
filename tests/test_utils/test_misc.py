# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from mmeval.utils import try_import


def test_try_import():
    import numpy as np
    assert try_import('numpy') is np
    assert try_import('numpy111') is None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
