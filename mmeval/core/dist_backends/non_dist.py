# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, List

from mmeval.core.dist_backends.base_dist import BaseDistributed


class NonDistributed(BaseDistributed):
    """A dummy distributed communication for non-distributed environment."""

    @property
    def is_dist_initialized(self) -> bool:
        """Returns False directly in a non-distributed environment."""
        return False

    @property
    def rank_id(self) -> int:
        """Returns 0 as the rank_id in a non-distributed environment."""
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
