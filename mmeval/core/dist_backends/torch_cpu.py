# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pickle
import torch
import torch.distributed as torch_dist

from mmeval.core.dist_backends.base_dist import TensorBaseDistributed


class TorchCPUDistributed(TensorBaseDistributed):

    def rank_id(self) -> int:
        return torch_dist.get_rank()

    def world_size(self) -> int:
        return torch_dist.get_world_size()

    def _object_to_tensor(self, obj):
        buffer = pickle.dumps(obj)
        obj_tensor = torch.tensor(np.frombuffer(buffer, dtype=np.int8))
        obj_size_tensor = torch.tensor(len(buffer), dtype=torch.long)
        return obj_tensor, obj_size_tensor

    def _tensor_to_object(self, tensor, tensor_size):
        buffer = tensor.numpy().tobytes()[:tensor_size]
        obj = pickle.loads(buffer)
        return obj

    def _pad_tensor(self, tensor, max_size):
        padding = torch.ones(max_size - tensor.size()[0], dtype=tensor.dtype)
        padded_tensor = torch.cat([tensor, padding], axis=0)
        return padded_tensor

    def _all_gather(self, tensor):
        global_tensor_list = [
            torch.empty_like(tensor) for _ in range(self.world_size())
        ]
        torch_dist.all_gather(global_tensor_list, tensor, group=None)
        return global_tensor_list

    def _broadcast(self, tensor, src):
        torch_dist.broadcast(tensor, src=src, group=None)
        return tensor
