# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path


def is_filepath(x):
    """Check if the given object is Path-like.

    Args:
        x (object): Any object.

    Returns:
        bool: Returns True if the given is a str or :class:`pathlib.Path`.
        Otherwise, returns False.
    """
    return isinstance(x, str) or isinstance(x, Path)
