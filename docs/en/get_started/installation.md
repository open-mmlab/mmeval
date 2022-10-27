# Installation and Usage

## Installation

`MMEval` requires Python 3.6+ and can be installed via pip.

```bash
pip install mmeval
```

To install the dependencies required for all the metrics provided in `MMEval`, you can install them with the following command.

```bash
pip install 'mmeval[all]'
```

## How to use

There are two ways to use `MMEval`'s metrics, using [Accuracy](mmeval.metrics.Accuracy) as an example:

```python
from mmeval import Accuracy
import numpy as np

accuracy = Accuracy()
```

The first way is to directly call the instantiated `Accuracy` object to calculate the metric.

```python
labels = np.asarray([0, 1, 2, 3])
preds = np.asarray([0, 2, 1, 3])
accuracy(preds, labels)
# {'top1': 0.5}
```

The second way is to calculate the metric after accumulating data from multiple batches.

```python
for i in range(10):
    labels = np.random.randint(0, 4, size=(100, ))
    predicts = np.random.randint(0, 4, size=(100, ))
    accuracy.add(predicts, labels)

accuracy.compute()
# {'top1': ...}
```
