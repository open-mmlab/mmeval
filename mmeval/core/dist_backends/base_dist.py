# Copyright (c) OpenMMLab. All rights reserved.

from abc import ABCMeta, abstractmethod, abstractproperty


class BaseDistributed(metaclass=ABCMeta):

    @abstractproperty
    def rank_id(self) -> int:
        ...

    @abstractproperty
    def world_size(self) -> int:
        ...

    @abstractmethod
    def all_gather_object(self, obj):
        ...

    @abstractmethod
    def broadcast_object(self, obj, src):
        ...


class TensorBaseDistributed(BaseDistributed):

    @abstractmethod
    def _object_to_tensor(self, obj):
        ...

    @abstractmethod
    def _tensor_to_object(self, tensor, tensor_size):
        ...

    @abstractmethod
    def _pad_tensor(self, tensor, max_size):
        ...

    @abstractmethod
    def _all_gather(self, tensor):
        ...

    @abstractmethod
    def _broadcast(self, tensor, src):
        ...

    def all_gather_object(self, obj):
        obj_tensor, obj_size_tensor = self._object_to_tensor(obj)

        global_obj_size_tensor = self._all_gather(obj_size_tensor)
        max_obj_size = max(global_obj_size_tensor)

        padded_obj_tensor = self._pad_tensor(obj_tensor, max_obj_size)
        global_padded_obj_tensor = self._all_gather(padded_obj_tensor)

        global_obj_list = []
        for padded_obj_tensor, obj_size_tensor in zip(global_padded_obj_tensor,
                                                      global_obj_size_tensor):
            obj = self._tensor_to_object(padded_obj_tensor, obj_size_tensor)
            global_obj_list.append(obj)
        return global_obj_list

    def broadcast_object(self, obj, src):
        obj_tensor, obj_size_tensor = self._object_to_tensor(obj)

        broadcast_obj_size_tensor = self._broadcast(obj_size_tensor, src)

        if self.rank_id != src:
            obj_tensor = self._pad_tensor(obj_tensor,
                                          broadcast_obj_size_tensor)

        broadcast_obj_tensor = self._broadcast(obj_tensor, src)
        broadcast_obj = self._tensor_to_object(broadcast_obj_tensor,
                                               obj_size_tensor)

        return broadcast_obj
