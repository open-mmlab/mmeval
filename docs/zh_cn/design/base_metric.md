# BaseMetric 设计

在评测过程中，通常会以数据并行的形式，在每张卡上推理部分数据集的结果，以加快评测速度。

而在每个数据子集上计算得到的评测结果，通常不能通过简单的求平均来与整个数据集的评测结果进行等价。因此通常的做法是在分布式评测过程中，将每张卡得到的推理结果或者指标计算中间结果保存下来，在所有进程中进行 all-gather 操作，最后再计算整个评测数据集的指标结果。

上述操作在 MMEval 中由 [BaseMetric](mmeval.core.BaseMetric) 来完成，其接口设计如下图所示：

```{mermaid}
classDiagram
    class BaseMetric
    BaseMetric : +{BaseDistBackend} dist_comm
    BaseMetric : +str dist_collect_mode
    BaseMetric : +dict dataset_meta
    BaseMetric : #list _results
    BaseMetric : +reset()
    BaseMetric : +compute()
    BaseMetric : +{abstractmethod} add()
    BaseMetric : +{abstractmethod} compute_metric()
```

其中 `add` 与 `compute_metric` 方法为需要用户继承实现的接口，具体可以参考[自定义评测指标](../tutorials/custom_metric.md)。

通过 [BaseMetric](mmeval.core.BaseMetric) 接口可以看出，[BaseMetric](mmeval.core.BaseMetric) 主要功能是提供分布式评测，其基本流程为：

1. 用户调用 `add` 方法，将推理结果或者指标计算中间结果保存在 `BaseMetric._results` 列表中。
2. 用户调用 `compute` 方法，`BaseMetric` 将 `_results` 列表中的数据进行进程间同步并调用用户定义的 `compute_metric` 方法进行指标的计算。

除此之外，[BaseMetric](mmeval.core.BaseMetric) 还考虑到数据并行过程中，为了保证所有进程中的数据样本数量一致，部分进程会有补齐重复数据样本的情况，比如 PyTorch 中的 `DistributedSampler`，这会影响到指标计算的正确性。

为了应对这个问题，[BaseMetric.compute](mmeval.core.BaseMetric.compute) 可以接收一个 `size` 参数，表示整个评测数据集的真实样本数量，在 `_results` 进程同步之后，调用 `compute_metric` 方法之前，根据 `dist_collect_mode` 去除用来补齐的重复样本，以实现正确的指标计算。

```{note}
为了能够在分布式评测时候将补齐的重复样本删除掉，存储在 `_results` 列表的中间值需要和评测数据集样本是一一对应的关系。
```
