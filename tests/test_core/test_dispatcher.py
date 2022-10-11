# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pytest
from typing import Dict, List, overload

from mmeval.core.dispatcher import _MMEvalDispatcher, dispatch
from mmeval.utils import try_import

torch = try_import('torch')


class TestMMEvalDispatcher:

    dispatcher = _MMEvalDispatcher()

    def test_resolve_importable_type(self):
        ty = self.dispatcher._resolve_importable_type('numpy.ndarray')
        assert ty is np.ndarray

        # Got warning: No module named 'np'.
        ty = self.dispatcher._resolve_importable_type('np.ndarray')
        assert ty is not np.ndarray
        # The placeholder should be a type
        assert type(ty) is type

        # The importable name should contain `.`
        with pytest.raises(AssertionError):
            self.dispatcher._resolve_importable_type('numpy')

    def test_traverse_type_hints(self):
        annotation = 'xxx'
        assert self.dispatcher._traverse_type_hints(annotation) is annotation

        annotation = '111'
        assert self.dispatcher._traverse_type_hints(annotation) is annotation

        annotation = 'numpy.ndarray'  # noqa: F821
        # ForwardRef could be a string directly, so we can resolve it.
        assert self.dispatcher._traverse_type_hints(annotation) is np.ndarray

        annotation = List['numpy.ndarray']  # noqa: F821
        assert self.dispatcher._traverse_type_hints(annotation) == List[
            np.ndarray]  # noqa: E501

        annotation = Dict[str, 'numpy.ndarray']  # noqa: F821
        assert self.dispatcher._traverse_type_hints(annotation) == Dict[
            str, np.ndarray]  # noqa: E501

        annotation = Dict[int, List['numpy.ndarray']]  # noqa: F821
        assert self.dispatcher._traverse_type_hints(annotation) == Dict[
            int, List[np.ndarray]]  # noqa: E501

    def test__call__(self):

        def fn(x: 'numpy.ndarray'):  # noqa: F821
            pass

        self.dispatcher(fn)
        assert hasattr(fn, '__signature__')


def test_multiple_dispatch_function():

    @overload
    @dispatch
    def fn(x: int, y: int):
        """In the case of int, return 1."""
        return 1

    @overload
    @dispatch
    def fn(x: float, y: float):
        """In the case of float, return 2."""
        return 2

    @overload
    @dispatch
    def fn(x: 'numpy.int8', y: 'numpy.int8'):  # noqa: F821
        """In the case of 'numpy.int8', return 3."""
        return 3

    @overload
    @dispatch
    def fn(x: 'xxxx.int8', y: 'xxxx.int8'):  # noqa: F821
        """This function test dispatch resolve unimportable type."""

    @overload
    @dispatch
    def fn(x: List[int], y: List[int]):
        """In the case of List[int], return 4."""
        return 4

    @dispatch
    def fn(x: Dict[str, int], y: Dict[str, int]):
        """In the case of Dict[str, int], return 5."""
        return 5

    assert fn(1, 2) == 1
    assert fn(1.0, 2.0) == 2
    assert fn(np.int8(1), np.int8(2)) == 3
    assert fn([1, 2], [3, 4]) == 4
    assert fn({'1': 2}, {'3': 4}) == 5

    with pytest.raises(LookupError):
        fn('1', '2')

    # Test if the type inference is accurate
    with pytest.raises(LookupError):
        fn({'1': 2}, {'3': 4, 5: '6'})


def test_multiple_dispatch_class_method():

    class Tester:

        @overload
        @dispatch
        def __call__(self, x: int, y: int):
            """In the case of int, this is a minimum method."""
            if x < y:
                return x
            else:
                return y

        @dispatch
        def __call__(self, x: float, y: float):
            """In the case of int, this is a maximum method."""
            if x > y:
                return x
            else:
                return y

    tester = Tester()
    assert tester(1, 2) == 1
    assert tester(1.0, 2.0) == 2.0

    with pytest.raises(LookupError):
        tester('1', '2')


@pytest.mark.skipif(torch is None, reason='PyTorch is not available!')
def test_multiple_dispatch_tensor():

    @overload
    @dispatch
    def fn(x: 'torch.Tensor', y: 'torch.Tensor'):
        """In the case of 'torch.Tensor', return 1."""
        return 1

    @dispatch
    def fn(x: 'numpy.int8', y: 'numpy.int8'):  # noqa: F821
        """In the case of 'numpy.int8', return 2."""
        return 2

    assert fn(torch.Tensor([1]), torch.Tensor([2])) == 1
    assert fn(np.int8(1), np.int8(2)) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--capture=no'])
