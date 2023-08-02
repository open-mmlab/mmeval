import os
import torch
import torchvision as tv
import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from mmeval import Accuracy


def get_eval_dataloader(rank=0, num_replicas=1):
    dataset = tv.datasets.CIFAR10(
        root='./',
        train=False,
        download=True,
        transform=tv.transforms.ToTensor())
    dist_sampler = DistributedSampler(
        dataset, num_replicas=num_replicas, rank=rank)
    data_loader = DataLoader(dataset, batch_size=1, sampler=dist_sampler)
    return data_loader, len(dataset)


def get_model(pretrained_model_fpath=None):
    model = tv.models.resnet18(num_classes=10)
    if pretrained_model_fpath is not None:
        model.load_state_dict(torch.load(pretrained_model_fpath))
    return model.eval()


def eval_fn(rank, process_num):
    master_addr = 'localhost'
    master_port = 12345

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    torch.distributed.init_process_group(
        backend='hccl',
        init_method='env://',
        world_size=process_num,
        rank=rank)

    num_npus = torch.npu.device_count()
    torch.npu.set_device(rank % num_npus)

    eval_dataloader, total_num_samples = get_eval_dataloader(rank, process_num)
    model = get_model().npu()
    accuracy = Accuracy(topk=(1, 3), dist_backend='npu_dist')

    with torch.no_grad():
        for images, labels in tqdm.tqdm(eval_dataloader, disable=(rank != 0)):
            images = images.npu()
            labels = labels.npu()
            predicted_score = model(images)
            accuracy.add(predictions=predicted_score, labels=labels)

    print(accuracy.compute(size=total_num_samples))
    accuracy.reset()


if __name__ == '__main__':
    process_num = 8
    torch.multiprocessing.spawn(
        eval_fn, nprocs=process_num, args=(process_num, ))
