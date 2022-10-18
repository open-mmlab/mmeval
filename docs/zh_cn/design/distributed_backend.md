# 分布式通信后端

MMEval 在分布式评测过程中所需的分布式通信需求，主要有以下两个：

- 将各个进程中保存的评测指标计算中间结果 all-gather
- 将 rank 0 进程计算得到的指标结果 broadcast 给所有进程

为了能够灵活的支持多种分布式通信库，MMEval 将上述分布式通信需求抽象定义了一个分布式通信接口 BaseDistBackend：

```{mermaid}
classDiagram
    class BaseDistBackend
    BaseDistBackend : +bool is_initialized
    BaseDistBackend : +int rank
    BaseDistBackend : +int world_size
    BaseDistBackend : +all_gather_object()
    BaseDistBackend : +broadcast_object()
```

实现一个分布式通信后端，需要继承 BaseDistBackend 并且实现上述接口，其中：

- is_initialized，标识当前是否已经完成分布式通信环境的初始化。
- rank，当前进程所在进程组的序号。
- world_size，进程数量。
- all_gather_object，对任意可以被 Pickle 序列化的 Python 对象进行 all_tather 操作。
- broadcast_object，对任意可以被 Pickle 序列化的 Python 对象进行广播操作。

以实现 MPI4PyDist 为例：

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

MMEval 中已经预置实现了一些分布式通信后端，具体可以在[支持矩阵](../get_started/support_matrix.md)中查看。
