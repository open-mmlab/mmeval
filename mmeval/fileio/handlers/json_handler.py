# Copyright (c) OpenMMLab. All rights reserved.
import json

from .base import BaseFileHandler


class JsonHandler(BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)
