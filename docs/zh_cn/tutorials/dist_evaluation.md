# 使用分布式评测

分布式评测一般采用数据并行的策略，每个进程执行相同的程序来处理不同的数据。

MMEval 中已支持的分布式通信后端可以通过 [list_all_backends](mmeval.core.list_all_backends) 查看：

```python
import mmeval

print(mmeval.core.dist.list_all_backends())
# ['non_dist', 'mpi4py', 'tf_horovod', 'torch_cpu', 'torch_cuda', ...]
```

本节将以 CIFAR-10 数据集的评测为例，分别介绍如何使用 MMEval 结合 `torch.distributed` 和 `MPI4Py` 进行分布式评测，相关代码可以在 [mmeval/examples/cifar10_dist_eval](https://github.com/open-mmlab/mmeval/tree/main/examples/cifar10_dist_eval) 中找到。

## 评测数据与模型准备

首先我们需要加载 CIFAR-10 测试数据，我们可以使用 `Torchvison` 提供的数据集类。

另外，为了能够在分布式评测中将数据集根据进程数量进行切分，我们需要引入 `DistributedSampler`。

```python
import torchvision as tv
from torch.utils.data import DataLoader, DistributedSampler

def get_eval_dataloader(rank=0, num_replicas=1):
    dataset = tv.datasets.CIFAR10(
        root='./', train=False, download=True,
        transform=tv.transforms.ToTensor())
    dist_sampler = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank)
    data_loader = DataLoader(dataset, batch_size=1, sampler=dist_sampler)
    return data_loader, len(dataset)
```

其次，我们需要准备待评测的模型，这里我们使用 `Torchvision` 中的 `resnet18`。

```python
import torch
import torchvision as tv

def get_model(pretrained_model_fpath=None):
    model = tv.models.resnet18(num_classes=10)
    if pretrained_model_fpath is not None:
        model.load_state_dict(torch.load(pretrained_model_fpath))
    return model.eval()
```

## 单进程评测

有了待评测的数据集与模型，就可以使用 [mmeval.Accuracy](mmeval.metrics.Accuracy) 指标对模型预测结果进行评测，下面是一个单进程评测的示例：

```python
import tqdm
import torch
from mmeval import Accuracy

eval_dataloader, total_num_samples = get_eval_dataloader()
model = get_model()
# 实例化 `Accuracy`，计算 top1 与 top3 准确率
accuracy = Accuracy(topk=(1, 3))

with torch.no_grad():
    for images, labels in tqdm.tqdm(eval_dataloader):
        predicted_score = model(images)
        # 累计批次数据，中间结果将保存在 `accuracy._results` 中
        accuracy.add(predictions=predicted_score, labels=labels)

# 调用 `accuracy.compute` 进行指标计算
print(accuracy.compute())
# 调用 `accuracy.reset` 清除保存在 `accuracy._results` 中的中间结果
accuracy.reset()
```

## 使用 torch.distributed 进行分布式评测

在 `MMEval` 中为 `torch.distributed` 实现了两个分布式通信后端，分别是 [TorchCPUDist](mmeval.core.dist_backends.TorchCPUDist) 和 [TorchCUDADist](mmeval.core.dist_backends.TorchCUDADist)。

为 `MMEval` 设置分布式通信后端的方式有两种：

```python
from mmeval.core import set_default_dist_backend
from mmeval import Accuracy

# 1. 设置全局默认分布式通信后端
set_default_dist_backend('torch_cpu')

# 2. 初始化评测指标时候通过 `dist_backend` 传参
accuracy = Accuracy(dist_backend='torch_cpu')
```

结合上述单进程评测的代码，再加入分布式环境启动以及初始化即可实现分布式评测。

```python
import tqdm
import torch
from mmeval import Accuracy


def eval_fn(rank, process_num):
    # 分布式环境初始化
    torch.distributed.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:2345',
        world_size=process_num,
        rank=rank)

    eval_dataloader, total_num_samples = get_eval_dataloader(rank, process_num)
    model = get_model()
    # 实例化 Accuracy 并设置分布式通信后端
    accuracy = Accuracy(topk=(1, 3), dist_backend='torch_cpu')

    with torch.no_grad():
        for images, labels in tqdm.tqdm(eval_dataloader, disable=(rank!=0)):
            predicted_score = model(images)
            accuracy.add(predictions=predicted_score, labels=labels)

    # 通过 size 指定数据集样本数量，以便去除 DistributedSampler 补齐的重复样本。
    print(accuracy.compute(size=total_num_samples))
    accuracy.reset()


if __name__ == "__main__":
    # 分布式进程数量
    process_num = 3
    # 采用 spawn 的方式启动分布式
    torch.multiprocessing.spawn(
        eval_fn, nprocs=process_num, args=(process_num, ))
```

## 使用 MPI4Py 进行分布式评测

`MMEval` 将分布式通信功能抽象解耦了，因此虽然上述例子使用的是 PyTorch 模型和数据加载，我们仍然可以使用除 `torch.distributed` 以外的分布式通信后端来实现分布式评测，下面将展示如何使用 `MPI4Py` 作为分布式通信后端来进行分布式评测。

首先需要安装 `MPI4Py` 以及 `openmpi`，建议使用 `conda` 进行安装：

```bash
conda install openmpi
conda install mpi4py
```

然后将上述代码修改为使用 `MPI4Py` 做为分布式通信后端：

```python
# cifar10_eval_mpi4py.py

import tqdm
from mpi4py import MPI
import torch
from mmeval import Accuracy


def eval_fn(rank, process_num):
    eval_dataloader, total_num_samples = get_eval_dataloader(rank, process_num)
    model = get_model()
    accuracy = Accuracy(topk=(1, 3), dist_backend='mpi4py')

    with torch.no_grad():
        for images, labels in tqdm.tqdm(eval_dataloader, disable=(rank!=0)):
            predicted_score = model(images)
            accuracy.add(predictions=predicted_score, labels=labels)

    print(accuracy.compute(size=total_num_samples))
    accuracy.reset()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    eval_fn(comm.Get_rank(), comm.Get_size())
```

使用 `mpirun` 作为分布式评测启动方式：

```bash
# 使用 mpirun 启动 3 个进程
mpirun -np 3 python3 cifar10_eval_mpi4py.py
```
