# Copyright (c) OpenMMLab. All rights reserved.
import pickle

from .base import BaseFileHandler


class PickleHandler(BaseFileHandler):

    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super().load_from_path(filepath, mode='rb', **kwargs)
