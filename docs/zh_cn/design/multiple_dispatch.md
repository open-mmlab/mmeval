# 基于参数类型注释的多分派

MMEval 希望能够支持多种机器学习框架，一个最为简单的方案是让所有评测指标的计算都支持 NumPy 即可。

这样做可以实现大部分评测需求，因为所有机器学习框架的 Tensor 数据类型都可以转为 numpy.ndarray。

但是在某些情况下可能会存在一些问题：

- NumPy 有一些常用算子尚未实现，如 topk，会影响评测指标的计算速度。
- 大量的 Tensor 从 CUDA 设备搬运到 CPU 内存会比较耗时。

另外，如果希望评测指标的计算过程是可导的，那么就需要用各自机器学习框架的 Tensor 数据类型进行计算。

为了应对上述问题，MMEval 的评测指标提供了一些特定机器学习框架的指标计算实现，具体可以在 [支持矩阵](../get_started/support_matrix.md) 中查看。

同时，为了应对不同指标计算方式的分发问题，MMEval 采用了基于类型注释的动态多分派机制，可以根据输入的数据类型，动态的选择不同的计算方式。

一个基于类型注释的多分派简单示例如下：

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

目前，我们使用 [plum-dispatch](https://github.com/wesselb/plum) 来实现 MMEval 中的分发机制，在 plum-dispatch 基础上，做了一些速度上的优化，并且扩展支持了 `typing.ForwardRef`。

```{warning}
受限于 Python 动态类型的特性，在运行时确定一个变量的具体类型可能会比较耗时，尤其是碰到一些大的嵌套结构数据。因此基于类型注释的动态多分派机制可能会存在一些性能问题，更多信息可以参考：[wesselb/plum/issues/53](https://github.com/wesselb/plum/issues/53)
```
