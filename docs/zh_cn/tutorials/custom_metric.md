# 自定义评测指标

在 MMEval 中实现一个自定义评测指标，需要继承 [BaseMetric](mmeval.core.BaseMetric) 并且实现 `add` 和 `compute_metric` 方法。

在评测过程中，评测指标需要在调用 `add` 后更新 `_results` 以存储中间结果。在最后进行指标计算的时候，将会对 `_results` 进行进程同步后调用 `compute_metric` 进行指标的计算。

以实现 `Accuracy` 指标为例：

```python
import numpy as np
from mmeval.core import BaseMetric

class Accuracy(BaseMetric):

    def add(self, predictions, labels):
        self._results.append((predictions, labels))

    def compute_metric(self, results):
        predictions = np.concatenate(
            [res[0] for res in results])
        labels = np.concatenate(
            [res[1] for res in results])
        correct = (predictions == labels)
        accuracy = sum(correct) / len(predictions)
        return {'accuracy': accuracy}
```

使用 `Accuracy`：

```python
# stateless call
accuracy = Accuracy()
metric_results = accuracy(predictions=[1, 2, 3, 4], labels=[1, 2, 3, 1])
print(metric_results)
# {'accuracy': 0.75}

# Accumulate batch
for i in range(10):
    predicts = np.random.randint(0, 4, size=(10,))
    labels = predicts = np.random.randint(0, 4, size=(10,))
    accuracy.add(predicts, labels)

metric_results = accuracy.compute()
accuracy.reset()  # clear the intermediate results
```
