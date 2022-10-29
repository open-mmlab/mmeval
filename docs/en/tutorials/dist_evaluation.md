# Using Distributed Evaluation

Distributed evaluation generally uses a strategy of data parallelism, where each process executes the same program to process different data.

The supported distributed communication backends in MMEval can be viewed via [list_all_backends](mmeval.core.list_all_backends).

```python
import mmeval

print(mmeval.core.dist.list_all_backends())
# ['non_dist', 'mpi4py', 'tf_horovod', 'torch_cpu', 'torch_cuda', ...]
```

This section shows how to use MMEval in the combination of `torch.distributed` and `MPI4Py` for distributed evaluation, using the CIFAR-10 dataset as an example. The related code can be found at [mmeval/examples/cifar10_dist_eval](https://github.com/open-mmlab/mmeval/tree/main/examples/cifar10_dist_eval).

## Prepare the evaluation dataset and model

First of all, we need to load the CIFAR-10 test data, we can use the dataset classes provided by `Torchvison`.

In addition, to be able to slice the dataset according to the number of processes in a distributed evaluation, we need to introduce the `DistributedSampler`.

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

Secondly, we need to prepare the model to be evaluated, here we use `resnet18` from `Torchvision`.

```python
import torch
import torchvision as tv

def get_model(pretrained_model_fpath=None):
    model = tv.models.resnet18(num_classes=10)
    if pretrained_model_fpath is not None:
        model.load_state_dict(torch.load(pretrained_model_fpath))
    return model.eval()
```

## Single process evaluation

After preparing the test data and the model, the model predictions can be evaluated using the [mmeval.Accuracy](mmeval.metrics.Accuracy) metric. The following is an example of a single process evaluation.

```python
import tqdm
import torch
from mmeval import Accuracy

eval_dataloader, total_num_samples = get_eval_dataloader()
model = get_model()
# Instantiate `Accuracy` and calculate the top1 and top3 accuracy
accuracy = Accuracy(topk=(1, 3))

with torch.no_grad():
    for images, labels in tqdm.tqdm(eval_dataloader):
        predicted_score = model(images)
        # Accumulate batch data, intermediate results will be saved in
        # `accuracy._results`.
        accuracy.add(predictions=predicted_score, labels=labels)

# Invoke `accuracy.compute` for metric calculation
print(accuracy.compute())
# Invoke `accuracy.reset` to clear the intermediate results saved in
# `accuracy._results`
accuracy.reset()
```

## Distributed evaluation with torch.distributed

There are two distributed communication backends implemented in `MMEval` for `torch.distributed`, [TorchCPUDist](mmeval.core.dist_backends.TorchCPUDist) and [TorchCUDADist](mmeval.core.dist_backends.TorchCUDADist).

There are 2 ways to set up a distributed communication backend for `MMEval`:

```python
from mmeval.core import set_default_dist_backend
from mmeval import Accuracy

# 1. Set the global default distributed communication backend.
set_default_dist_backend('torch_cpu')

# 2. Initialize the evaluation metrics by passing `dist_backend`.
accuracy = Accuracy(dist_backend='torch_cpu')
```

Together with the above code for single process evaluation, the distributed evaluation can be implemented by adding the distributed environment startup and initialization.

```python
import tqdm
import torch
from mmeval import Accuracy


def eval_fn(rank, process_num):
    # Distributed environment initialization
    torch.distributed.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:2345',
        world_size=process_num,
        rank=rank)

    eval_dataloader, total_num_samples = get_eval_dataloader(rank, process_num)
    model = get_model()
    # Instantiate `Accuracy` and set up a distributed communication backend
    accuracy = Accuracy(topk=(1, 3), dist_backend='torch_cpu')

    with torch.no_grad():
        for images, labels in tqdm.tqdm(eval_dataloader, disable=(rank!=0)):
            predicted_score = model(images)
            accuracy.add(predictions=predicted_score, labels=labels)

    # Specify the number of dataset samples by size in order to remove
    # duplicate samples padded by the `DistributedSampler`.
    print(accuracy.compute(size=total_num_samples))
    accuracy.reset()


if __name__ == "__main__":
    # Number of distributed processes
    process_num = 3
    # Launching distributed with spawn
    torch.multiprocessing.spawn(
        eval_fn, nprocs=process_num, args=(process_num, ))
```

## Distributed evaluation with MPI4Py

`MMEval` has decoupled the distributed communication capability. While the above example uses the `PyTorch` model and data loading, we can still use distributed communication backends other than `torch.distributed` to implement distributed evaluation.

The following will show how to use `MPI4Py` as a distributed communication backend for distributed evaluation.

First, you need to install `MPI4Py` and `openmpi`, it is recommended to use `conda` to install.

```bash
conda install openmpi
conda install mpi4py
```

Then modify the above code to use `MPI4Py` as the distributed communication backend:

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

Using `mpirun` as the distributed launch method.

```bash
# Launch 3 processes with mpirun
mpirun -np 3 python3 cifar10_eval_mpi4py.py
```
