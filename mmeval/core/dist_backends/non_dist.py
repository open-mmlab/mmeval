# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, List

from .base_backend import BaseDistBackend


class NonDist(BaseDistBackend):
    """A dummy distributed communication for non-distributed environment."""

    @property
    def is_initialized(self) -> bool:
        """Returns False directly in a non-distributed environment."""
        return False

    @property
    def rank(self) -> int:
        """Returns 0 as the rank index in a non-distributed environment."""
        return 0

    @property
    def world_size(self) -> int:
        """Returns 1 as the world_size in a non-distributed environment."""
        return 1

    def all_gather_object(self, obj: Any) -> List[Any]:
        """Returns the list with given obj in a non-distributed environment."""
        return [obj]

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Returns the given obj directly in a non-distributed environment."""
        return obj
