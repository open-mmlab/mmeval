# Distributed Communication Backend

The distributed communication requirements required by `MMEval` in the distributed evaluation mainly include the following:

- All-gather the intermediate results of the metric saved in each process.
- Broadcast the metric result calculated by the rank 0 process to all processes

In order to flexibly support multiple distributed communication libraries, MMEval abstracts the above distributed communication requirements and defines a distributed communication interface [BaseDistBackend](mmeval.core.dist_backends.BaseDistBackend):

```{mermaid}
classDiagram
    class BaseDistBackend
    BaseDistBackend : +bool is_initialized
    BaseDistBackend : +int rank
    BaseDistBackend : +int world_size
    BaseDistBackend : +all_gather_object()
    BaseDistBackend : +broadcast_object()
```

To implement a distributed communication backend, you need to inherit [BaseDistBackend](mmeval.core.dist_backends.BaseDistBackend) and implement the above interfaces, where:

- is_initialized: identifies whether the initialization of the distributed communication environment has been completed.
- rank: the rank index of the current process group.
- world_size: the world size of the current process group.
- all_gather_object: perform the all_tather operation on any Python object that can be serialized by `Pickle`.
- broadcast_object: broadcasts any Python object that can be serialized by `Pickle`.

Take the implementation of [MPI4PyDist](mmeval.core.dist_backends.MPI4PyDist) as an example:

```python
from mpi4py import MPI


class MPI4PyDist(BaseDistBackend):
    """A distributed communication backend for mpi4py."""

    @property
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized."""
        return 'OMPI_COMM_WORLD_SIZE' in os.environ

    @property
    def rank(self) -> int:
        """Returns the rank index of the current process group."""
        comm = MPI.COMM_WORLD
        return comm.Get_rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        comm = MPI.COMM_WORLD
        return comm.Get_size()

    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process."""
        comm = MPI.COMM_WORLD
        return comm.allgather(obj)

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast the given object from source process to the current
        process group."""
        comm = MPI.COMM_WORLD
        return comm.bcast(obj, root=src)
```

Some distributed communication backends have been implemented in `MMEval`, which can be viewed in the [support matrix](../get_started/support_matrix.md).
