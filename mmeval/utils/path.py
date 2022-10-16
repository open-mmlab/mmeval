# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path


def is_filepath(x):
    return isinstance(x, str) or isinstance(x, Path)
