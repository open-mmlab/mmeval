# 安装与使用

### 安装

MMEval 依赖 Python 3.6+，可以通过 pip 来安装 MMEval。安装 MMEval 的过程中会安装一些 MMEval 运行时的依赖库：

```bash
pip install mmeval
```

如果要安装 MMEval 中所有评测指标都需要的依赖，可以通过以下命令安装：

```bash
pip install 'mmeval[all]'
```

### 使用

MMEval 中的评测指标提供两种使用方式，以 `Accuracy` 为例：

```python
from mmeval import Accuracy
import numpy as np

accuracy = Accuracy()
```

第一种是直接调用实例化的 Accuracy 对象，计算评测指标：

```python
labels = np.asarray([0, 1, 2, 3])
preds = np.asarray([0, 2, 1, 3])
accuracy(preds, labels)
# {'top1': 0.5}
```

第二种是累积多个批次的数据后，计算评测指标：

```python
for i in range(10):
    labels = np.random.randint(0, 4, size=(100, ))
    predicts = np.random.randint(0, 4, size=(100, ))
    accuracy.add(predicts, labels)

accuracy.compute()
# {'top1': ...}
```
