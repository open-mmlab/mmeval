# BaseMetric Design

During the evaluation process, the results of partial datasets are usually inferred on each GPU in data parallel to speed up the evaluation.

Most of the time, we can't just reduce the metric results from each subset of the dataset as the metric result of the dataset.

Therefore, the usual practice is to save the inference results obtained by each process or the intermediate results of the metric calculation. Then perform an all-gather operation across all processes, and finally calculate the metric results of the entire evaluation dataset.

The above operations are completed by [BaseMetric](mmeval.core.BaseMetric) in `MMEval`, and its interface design is shown in the following:

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

The `add` and `compute_metric` methods are interfaces that need to be implemented by users. For more details, please refer to [Custom Evaluation Metrics](../tutorials/custom_metric.md).

It can be seen from the `[BaseMetric](mmeval.core.BaseMetric) interface that the main function of `[BaseMetric](mmeval.core.BaseMetric) is to provide distributed evaluation. The basic process is as follows:

1. The user calls the `add` method to save the inference result or the intermediate result of the metric calculation in the `BaseMetric._results` list.
2. The user calls the `compute` method, and `BaseMetric` synchronizes the data in the `_results` list across processes and calls the user-defined `compute_metric` method to calculate the metrics.

In addition, [BaseMetric](mmeval.core.BaseMetric) also considers that in distributed evaluation, some processes may pad repeated data samples, in order to ensure the same number of data samples in all processes. Such behavior will affect the indicators correctness of the calculation. E.g. `DistributedSampler` in PyTorch.

To deal with this problem, [BaseMetric.compute](mmeval.core.BaseMetric.compute) can receive a `size` parameter, which represents the actual number of samples in the evaluation dataset. After `_results` completes process synchronization, the padded samples will be removed according to `dist_collect_mode` to achieve correct metric calculation.

```{note}
Be aware that the intermediate results stored in `_results` should correspond one-to-one with the samples, in that we need to remove the padded samples for the most accurate result.
```
