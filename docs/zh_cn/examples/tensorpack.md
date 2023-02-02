# TensorPack

[TensorPack](https://github.com/tensorpack/tensorpack) 是一个基于 TensorFlow 的深度学习训练库，具有高效与灵活的特点。

在 [TensorPack](https://github.com/tensorpack/tensorpack) 代码仓库中，提供了许多经典模型与任务的[示例](https://github.com/tensorpack/tensorpack/tree/master/examples)，本小节展示如何在 [TensorPack-FasterRCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) 中使用 [mmeval.COCODetection](mmeval.metrics.COCODetection) 进行评测，相关代码可以在 [mmeval/examples/tensorpack](https://github.com/open-mmlab/mmeval/tree/main/examples/tensorpack) 中找到。

首先需要安装 TensorFlow 与 TensorPack，然后按照 TensorPack-FasterRCNN 示例中的准备步骤，安装依赖和准备 COCO 数据集，以及下载需要评测的预训练模型权重。

TensorPack-FasterRCNN 自带了评测功能，可以通过以下命令执行评测：

```bash
./predict.py --evaluate output.json --load /path/to Trained-Model-Checkpoint --config SAME-AS-TRAINING
```

MMEval 为 TensorPack-FasterRCNN 提供了适配 [mmeval.COCODetection](mmeval.metrics.COCODetection) 的[评测脚本](https://github.com/open-mmlab/mmeval/tree/main/examples/tensorpack/tensorpack_mmeval.py)，需要将该脚本放至 TensorPack-FasterRCNN 示例目录下，然后通过以下命令执行评测：

```bash
# 单卡评测
python tensorpack_mmeval.py --load <model_path> --config SAME-AS-TRAINING

# 支持基于 MPI4Py 的分布式评测，通过 mpirun 启动多卡评测
mpirun -np 8 python tensorpack_mmeval.py --load <model_path> --config SAME-AS-TRAINING
```

我们在 [COCO-MaskRCNN-R50C41x](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz) 配置上测试了该评测脚本，与 TensorPack-FasterRCNN 报告的评测结果一致。

|                                           Model                                            | mAP (box) | mAP (mask) |  Configurations  |
| :----------------------------------------------------------------------------------------: | :-------: | :--------: | :--------------: |
| [COCO-MaskRCNN-R50C41x](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz) |   36.2    |    31.8    | `MODE_FPN=False` |
