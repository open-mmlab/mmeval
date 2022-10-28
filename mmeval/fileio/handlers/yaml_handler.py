# Copyright (c) OpenMMLab. All rights reserved.
import yaml

try:
    from yaml import CLoader as Loader  # type: ignore
except ImportError:
    from yaml import Loader  # type: ignore

from .base import BaseFileHandler  # isort:skip


class YamlHandler(BaseFileHandler):
    """A Yaml handler that parse yaml data from file object."""

    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault('Loader', Loader)
        return yaml.load(file, **kwargs)
