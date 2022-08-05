# Copyright (c) OpenMMLab. All rights reserved.

import horovod.tensorflow as hvd

from mmeval.core.dist_backends.base_dist import BaseDistributed


class TFHorovodDistributed(BaseDistributed):

    @property
    def rank_id(self) -> int:
        return hvd.rank()

    @property
    def world_size(self) -> int:
        return hvd.size()

    def all_gather_object(self, obj):
        return hvd.allgather_object(obj)

    def broadcast_object(self, obj, src):
        return hvd.broadcast_object(obj, root_rank=src)
