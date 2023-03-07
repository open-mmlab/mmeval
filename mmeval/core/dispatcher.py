# Copyright (c) OpenMMLab. All rights reserved.
"""This module introduces a multiple dispatch mechanism into mmeval.

Some mmeval metrics may have different calculation methods depending on the
deep learning framework or numeric computing libraries used, such as PyTorch
and NumPy.

In order to deal with the dispatch issue of different calculation methods, we
adopt a dynamic multiple dispatch mechanism based on type hints.

A simple example of multiple dispatch based on type hints is as below:

Example:

    >>> from mmeval.core import dispatch

    >>> @dispatch
    >>> def compute(x: int, y: int):
    ...     print('this is int')

    >>> @dispatch
    >>> def compute(x: str, y: str):
    ...     print('this is str')

    >>> compute(1, 1)
    this is int
    >>> compute('1', '1')
    this is str

Currently, we use plum (a multiple dispatch library) to implement multiple
dispatch mechanism in mmeval.

In this module, we optimized the execution speed of plum through the following
two tricks:

- Caching plum Type instances
- Caching plum Type hash value

Benefit from the tricks above, plum dispatch got twice faster as before.
More detail can be found at: https://github.com/wesselb/plum/issues/53

Besides, we implement `_MMEvalDispatcher` to extend plum dispatch for better
support of ``typing.ForwardRef``.
"""

import importlib
import inspect
import plum
import sys
from typing import Any, Callable, Dict, Hashable, Optional, Type

from mmeval.utils import DEFAULT_LOGGER

# Compatible with python 3.6
if sys.version_info.major >= 3 and sys.version_info.minor <= 6:
    from typing import _ForwardRef as ForwardRef  # type: ignore
else:
    from typing import ForwardRef

logger = DEFAULT_LOGGER


def _singleton_patch() -> None:
    """A monkey patch that makes `plum.type.TypeMeta` become singleton."""
    origin_call = plum.type.TypeMeta.__call__
    plum.type.TypeMeta._instances = {}

    def __call__(cls, *args, **kwargs):
        assert not kwargs
        key = (cls, args)
        if key not in cls._instances:
            cls._instances[key] = origin_call(cls, *args, **kwargs)
        return cls._instances[key]

    plum.type.TypeMeta.__call__ = __call__


try:
    # Since a lot of instance creation in `plum.type_of` can be expensive,
    # using the singleton Type can speed up a lot.
    _singleton_patch()
except Exception as e:
    logger.warning(
        f'Patch `plum.type.TypeMeta` with singleton failed, raise error: {e}. '
        'The multiple dispatch speed may be slow.')


def _hash_cache_patch(hashable_type: Type[Hashable]) -> None:
    """A monkey patch that makes class cache hash value.

    This is a very useful trick to optimize runtime speed for classes that
    frequently call hash methods.

    Args:
        hashable_type (Type[Hashable]): Hashable type that wants to cache hash
            value.
    """
    hash_core = hashable_type.__hash__
    hashable_type._hash = None  # type: ignore

    def __hash__(self):
        if self._hash is None:
            self._hash = hash_core(self)
        return self._hash

    hashable_type.__hash__ = __hash__  # type: ignore


try:
    # The hash methods of plum Type and Parametric would be called frequently.
    # Caching hash value can speed up a lot.
    from plum.parametric import Dict as plum_Dict
    from plum.parametric import Iterable as plum_Iterable
    from plum.parametric import List as plum_List
    from plum.parametric import Sequence as plum_Sequence
    from plum.parametric import Tuple as plum_Tuple

    _hash_cache_patch(plum.type.Type)
    _hash_cache_patch(plum.type.VarArgs)
    _hash_cache_patch(plum.type.Union)
    _hash_cache_patch(plum_Tuple)
    _hash_cache_patch(plum_List)
    _hash_cache_patch(plum_Dict)
    _hash_cache_patch(plum_Sequence)
    _hash_cache_patch(plum_Iterable)
except Exception as e:
    logger.warning(
        f'Patch plum Type with hash value cache failed, raise error: {e}. '
        'The multiple dispatch speed may be slow.')


class _MMEvalDispatcher(plum.Dispatcher):
    """A Dispatcher inherited from ``plum.Dispatcher`` that resolve
    ``typing.ForwardRef``.

    This dispatcher tries to use ``importlib.import_moudle`` to import
    ForwardRerf type and convert unimportable type as a placeholder.

    With the ``_MMEvalDispatcher``, we can run the following code example
    without PyTorch installed, which is ``plum.dispatch`` can't do.

    Example:
        >>> from mmeval.core import dispatch

        >>> @dispatch
        >>> def compute(x: 'torch.Tensor'):
        ...     print('The input is a `torch.Tensor`')

        >>> @dispatch
        >>> def compute(x: 'numpy.ndarray'):
        ...     print('The input is a `numpy.ndarray`')
    """

    # Caching the unimportable types to avoid repeated unimportable warning.
    # Importable case: `tensorflow.Tensor`.
    # Unimportable case: `tf.Tensor` (ModuleNotFoundError: No module named 'tf').  # noqa: E501
    _unimportable_types: Dict[str, Type] = {}

    def _resolve_importable_type(self, importable_name: str) -> Type:
        """Resolve the given importable name and returns a type.

        The given importable name should be a string contains at least one dot,
        so that we can split it as module path and type attribute name.
        e.g. 'torch.Tensor' and 'numpy.ndarray'.

        Args:
            importable_name (str): An importable string that wants to resolve.

        Returns:
            Type: The resolved type or an placeholder for unimportable type.
        """
        assert '.' in importable_name, 'The importable name should contain `.`'
        module_name, _, module_attr_basename = importable_name.rpartition('.')
        try:
            module = importlib.import_module(module_name)
            resolved_type = getattr(module, module_attr_basename)
        except Exception as e:
            if importable_name not in self._unimportable_types:
                logger.debug(
                    f"Unimportable: '{importable_name}', raise error: {e}.")
                resolved_type = type(importable_name, (), {})
                self._unimportable_types[importable_name] = resolved_type
            else:
                resolved_type = self._unimportable_types[importable_name]
        return resolved_type

    def _traverse_type_hints(self, annotation: Any) -> Any:
        """Traverse nested type hints, and resolve importable ForwardRef.

        Note:
            In general, we want a type hint, but be aware that function
            annotations could be anything. See PEP 3107 and PEP 484 for more.

        Args:
            annotation (Annotated): The function annotation that wants to
                resolve ForwardRef.

        Returns:
            Annotated: The traversed function annotation.
        """
        if isinstance(annotation, (ForwardRef, str)):
            # NOTE: ForwardRef could be a string directly.
            # https://docs.python.org/3/library/typing.html#typing.ForwardRef
            if isinstance(annotation, ForwardRef):
                forward_ref_name = annotation.__forward_arg__
            else:
                forward_ref_name = annotation
            # Currently, we only hold ForwardRef that contain `.`
            # In the case of self type, plum has considered that.
            if '.' in forward_ref_name:
                return self._resolve_importable_type(forward_ref_name)
            else:
                return annotation

        # Recursively traverse nested type hints.
        if getattr(annotation, '__module__', None) == 'typing' \
                and getattr(annotation, '__args__', None) is not None:
            new_tp_args = []
            for tp_arg in annotation.__args__:
                new_tp_arg = self._traverse_type_hints(tp_arg)
                new_tp_args.append(new_tp_arg)
            annotation.__args__ = tuple(new_tp_args)

        return annotation

    def __call__(self,
                 method: Optional[Callable] = None,
                 **kwargs) -> Callable:
        """Process the function annotations and resolve type hints that in
        ForwardRef form."""
        if method is not None:
            signature = inspect.signature(method)
            for param in signature.parameters.values():
                param._annotation = self._traverse_type_hints(  # type: ignore
                    param._annotation)  # type: ignore
            signature._return_annotation = self._traverse_type_hints(  # type: ignore # noqa: E501
                signature._return_annotation)  # type: ignore
            method.__signature__ = signature  # type: ignore
        return super().__call__(method=method, **kwargs)


dispatch = _MMEvalDispatcher()
