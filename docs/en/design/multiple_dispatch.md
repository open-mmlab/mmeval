# Multiple Dispatch

MMEval wants to support multiple machine learning frameworks. One of the simplest solutions is to have NumPy support for the computation of all metrics.

Since all machine learning frameworks have Tensor data types that can be converted to numpy.ndarray, this can satisfy most of the evaluation requirements.

However, there may be some problems in some cases:

- NumPy has some common operators that have not been implemented yet, such as topk, which can affect the computational speed of the evaluation metrics.
- It is time-consuming to move a large number of Tensors from CUDA devices to CPU memory.

Alternatively, if it is desired that the computation of the metrics for the rubric be differentiable, then the Tensor data type of the respective machine learning framework needs to be used for the computation.

To deal with the above, MMEval's evaluation metrics provide some implementations of metrics computed with specific machine learning frameworks, which can be found in \[support_matrix\](.. /get_started/support_matrix.md).

Meanwhile, in order to deal with the dispatch problem of different metrics calculation methods, MMEval adopts a dynamic multi-distribution mechanism based on type hints, which can dynamically select corresponding calculation methods according to the input data types.

A simple example of multiple dispatch based on type hints is as below:

```python
from mmeval.core import dispatch

@dispatch
def compute(x: int, y: int):
    print('this is int')

@dispatch
def compute(x: str, y: str):
    print('this is str')

compute(1, 1)
# this is int

compute('1', '1')
# this is str
```

Currently, we use [plum-dispatch](https://github.com/wesselb/plum) to implement multiple
dispatch mechanism in `MMEval`. Based on plum-dispatch, some speed optimizations have been made and extended to support `typing.ForwardRef`.

```{warning}
Due to the dynamically typed feature of Python, determining the exact type of a variable at runtime can be time-consuming, especially when you encounter large nested structures of data. Therefore, the dynamic multi-dispatch mechanism based on type hints may have some performance problems, for more information see at：[wesselb/plum/issues/53](https://github.com/wesselb/plum/issues/53)
```
