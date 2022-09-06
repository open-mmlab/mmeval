# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import re

import mmeval


def test_version():
    assert hasattr(mmeval, '__version__')

    # Check if version identifier is in the canonical format.
    # Ref: https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions  # noqa: E501
    assert re.match(
        r'''^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)
        (0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$''',
        mmeval.__version__) is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
