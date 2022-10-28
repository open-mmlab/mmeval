# Copyright (c) OpenMMLab. All rights reserved.
import json

from .base import BaseFileHandler


class JsonHandler(BaseFileHandler):
    """A Json handler that parse json data from file object."""

    def load_from_fileobj(self, file):
        return json.load(file)
