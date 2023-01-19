# PaddleSeg

[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 是一个基于 Paddle 的语义分割算法库，支持许多和语义分割相关的下游任务。

本小节展示如何在 PaddleSeg 中使用 [mmeval.MeanIoU](mmeval.metrics.MeanIoU) 进行评测，相关代码可以在 [mmeval/examples/paddleseg](https://github.com/open-mmlab/mmeval/tree/main/examples/paddleseg) 中找到。

首先需要安装 Paddle 与 PaddleSeg，可以参照 PaddleSeg 中的[安装文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/install_cn.md)进行。另外需要下载待评测的预训练模型，以及根据配置文件准备数据集。

PaddleSeg 算法库中提供了进行模型评测的脚本，可以通过以下命令对模型进行评测：

```bash
python val.py --config <config_path> --model_path <model_path>
```

需要注意，PaddleSeg 算法仓库中的 `val.py` 评测脚本只支持单卡评测，尚未支持多卡评测。

MMEval 为 PaddleSeg 提供了适配 [mmeval.MeanIoU](mmeval.metrics.MeanIoU) 的[评测脚本](https://github.com/open-mmlab/mmeval/blob/main/examples/paddleseg/ppseg_mmeval.py)，可以通过以下命令执行评测：

```bash
# 单卡评测
python ppseg_mmeval.py --config <config_path> --model_path <model_path>

# 单机多卡评测
python ppseg_mmeval.py --config <config_path> --model_path <model_path> --launcher paddle --num_process <num_gpus>
```

我们在 [fastfcn_resnet50_os8_ade20k_480x480_120k](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/configs/fastfcn) 配置上测试了该评测脚本，与 PaddleSeg 中的 `val.py` 得到的评测结果一致。

|                                                                               Config                                                                                |                                                         Weights                                                          |  mIoU  |  aAcc  | Kappa  | mDice  |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: | :----: | :----: | :----: | :----: |
| [fastfcn_resnet50_os8_ade20k_480x480_120k](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml) | [model.pdparams](https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/model.pdparams) | 0.4373 | 0.8074 | 0.7928 | 0.5772 |
