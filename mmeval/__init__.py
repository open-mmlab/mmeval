# Copyright (c) OpenMMLab. All rights reserved.

# flake8: noqa

import warnings

from .core import *
from .fileio import *
from .metrics import *
from .utils import *
from .version import __version__

from .metrics import __deprecated_metric_names__, _deprecated_msg  # isort:skip


def __getattr__(attr: str):
    """Customization of module attribute access.

    Thanks to pep-0562, we can customize moudel's attribute access
    via ``__getattr__`` to implement deprecation warnings.

    With this function, we can implement the following features:

        >>> from mmeval import COCODetectionMetric
        <stdin>:1: DeprecationWarning: `COCODetectionMetric` is a deprecated
        metric alias for `COCODetection`. To silence this warning, use `COCODetection`
        by itself. The deprecated metric alias would be removed in mmeval 1.0.0!
    """
    if attr in __deprecated_metric_names__:
        message = _deprecated_msg.format(
            n1=attr, n2=__deprecated_metric_names__[attr])
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return getattr(metrics, attr)  # type: ignore
    raise AttributeError(f'module {__name__!r} has no attribute {attr!r}')
