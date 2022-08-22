# Copyright (c) OpenMMLab. All rights reserved.

from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Tuple, TypeVar, Union

Tensor = TypeVar('Tensor')


class BaseDistBackend(metaclass=ABCMeta):
    """The base backend of distributed communication used by mmeval Metric."""

    @abstractproperty
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
            initialized, otherwise returns False.
        """

    @abstractproperty
    def rank(self) -> int:
        """Returns the rank index of the current process group.

        Returns:
            int: The rank index of the current process group.
        """

    @abstractproperty
    def world_size(self) -> int:
        """Returns the world size of the current process group.

        The `world size` is the size of the communication process group.

        Returns:
           int: The size of the current process group.
        """

    @abstractmethod
    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process..

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """

    @abstractmethod
    def broadcast_object(self, obj: Any, src: int) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """


class TensorBaseDistBackend(BaseDistBackend):
    """A base backend of Tensor base distributed communication like PyTorch."""

    @abstractmethod
    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            Tuple: A tuple of the tensor converted from the given object and
            the tensor size.
        """

    @abstractmethod
    def _tensor_to_object(self, tensor: Tensor,
                          tensor_size: Union[int, Tensor]) -> Any:
        """Convert the given Tensor to a object via `pickle.loads`.

        Args:
            tenosr (Tensor): A tensor-like data.
            tensor_size (int or Tensor): The tensor size of the given Tensor to
                be convert object.

        Returns:
            Any: The object converted from the given tensor.
        """

    @abstractmethod
    def _pad_tensor(self, tensor: Tensor,
                    max_size: Union[int, Tensor]) -> Tensor:  # yapf: disable
        """Padding the given tensor to the given size with 0.

        Args:
            tensor (Tensor): A tensor-like data to be padded.
            max_size (int or Tensor): The max tensor size that for tensor
                padding.

        Returns:
            Tensor: The padded tensor.
        """

    @abstractmethod
    def _all_gather(self, tensor: Tensor) -> List[Tensor]:
        """All gather the given tensor.

        Args:
            tensor (Tensor): The tensor for all gather.

        Returns:
            list: A list of the gathered tensor.
        """

    @abstractmethod
    def _broadcast(self, tensor: Tensor, src: int) -> Tensor:
        """Broadcast the given object from the source rank.

        Args:
            tensor (Tensor): The tensor for broadcast.
            src (int): The source rank index.

        Returns:
            Tensor: The broadcast tensor.
        """

    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process..

        There are 3 steps to all gather a python object using Tensor
        distributed communication:

        1. Serialize picklable python object to tensor.
        2. All gather the tensor size and padding the tensor with
           the same size.
        3. All gather the padded tensor and deserialize tensor to picklable
           python object.

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """
        obj_tensor, obj_size_tensor = self._object_to_tensor(obj)  # type: ignore # noqa: E501 # yapf: disable

        obj_size_list = self._all_gather(obj_size_tensor)
        max_obj_size = max(obj_size_list)

        padded_obj_tensor = self._pad_tensor(obj_tensor, max_obj_size)
        padded_obj_tensor_list = self._all_gather(padded_obj_tensor)

        obj_list = []
        for padded_obj_tensor, obj_size_tensor in zip(padded_obj_tensor_list,
                                                      obj_size_list):
            obj = self._tensor_to_object(padded_obj_tensor, obj_size_tensor)
            obj_list.append(obj)
        return obj_list

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        There are 3 steps to broadcast a python object use Tensor distributed
        communication:

        1. Serialize picklable python object to tensor.
        2. Broadcast the tensor size and padding the tensor with the same size.
        3. Broadcast the padded tensor and deserialize tensor to picklable
        python object.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """
        obj_tensor, obj_size_tensor = self._object_to_tensor(obj)  # type: ignore # noqa: E501 # yapf: disable

        broadcast_obj_size_tensor = self._broadcast(obj_size_tensor, src)

        if self.rank != src:
            obj_tensor = self._pad_tensor(obj_tensor,
                                          broadcast_obj_size_tensor)

        broadcast_obj_tensor = self._broadcast(obj_tensor, src)
        broadcast_obj = self._tensor_to_object(broadcast_obj_tensor,
                                               obj_size_tensor)

        return broadcast_obj
