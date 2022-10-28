<div align="center">
  <img src="https://github.com/open-mmlab/mmeval/raw/main/docs/zh_cn/_static/image/mmeval-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmeval)](https://pypi.org/project/mmeval/)
[![PyPI](https://img.shields.io/pypi/v/mmeval)](https://pypi.org/project/mmeval)
[![license](https://img.shields.io/github/license/open-mmlab/mmeval.svg)](https://github.com/open-mmlab/mmeval/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmeval.svg)](https://github.com/open-mmlab/mmeval/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmeval.svg)](https://github.com/open-mmlab/mmeval/issues)

[📘使用文档](https://mmeval.readthedocs.io/zh_CN/latest/) |
[🛠️安装教程](https://mmeval.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[🤔报告问题](https://github.com/open-mmlab/mmeval/issues/new/choose)

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## 简介

MMEval 是一个机器学习算法评测库，提供高效准确的分布式评测以及多种机器学习框架后端支持，具有以下特点：

- 提供丰富的计算机视觉各细分方向评测指标（自然语言处理方向的评测指标正在支持中）
- 支持多种分布式通信库，实现高效准确的分布式评测。
- 支持多种机器学习框架，根据输入自动分发对应实现。

<div  align="center">
  <img src="docs/zh_cn/_static/image/mmeval-arch.png" width="600"/>
</div>

<details>
<summary> 支持的分布式通信后端 </summary>

|                                                                        MPI4Py                                                                         |                                                                                                                                                     torch.distributed                                                                                                                                                      |                                                                           Horovod                                                                           |                                                                  paddle.distributed                                                                   |
| :---------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: |
| [MPI4PyDist](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.MPI4PyDist) | [TorchCPUDist](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TorchCPUDist) <br> [TorchCUDADist](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TorchCUDADist) | [TFHorovodDist](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TFHorovodDist) | [PaddleDist](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.PaddleDist) |

</details>

<details>
<summary> 支持的评测指标及机器学习框架 </summary>

|                                                                          评测指标                                                                          | NumPy | PyTorch | TensorFlow | Paddle |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :---: | :-----: | :--------: | :----: |
|                 [Accuracy](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.Accuracy.html#mmeval.metrics.Accuracy)                  |   ✔   |    ✔    |     ✔      |   ✔    |
|    [SingleLabelMetric](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.SingleLabelMetric.html#mmeval.metrics.SingleLabelMetric)    |   ✔   |    ✔    |            |        |
|     [MultiLabelMetric](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.MultiLabelMetric.html#mmeval.metrics.MultiLabelMetric)      |   ✔   |    ✔    |            |        |
|     [AveragePrecision](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.AveragePrecision.html#mmeval.metrics.AveragePrecision)      |   ✔   |    ✔    |            |        |
|                   [MeanIoU](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.MeanIoU.html#mmeval.metrics.MeanIoU)                   |   ✔   |    ✔    |     ✔      |   ✔    |
|                [VOCMeanAP](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.VOCMeanAP.html#mmeval.metrics.VOCMeanAP)                |   ✔   |         |            |        |
|                [OIDMeanAP](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.OIDMeanAP.html#mmeval.metrics.OIDMeanAP)                |   ✔   |         |            |        |
| [CocoDetectionMetric](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.COCODetectionMetric.html#mmeval.metrics.COCODetectionMetric) |   ✔   |         |            |        |
|        [ProposalRecall](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.ProposalRecall.html#mmeval.metrics.ProposalRecall)         |   ✔   |         |            |        |
|                 [F1Metric](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.F1Metric.html#mmeval.metrics.F1Metric)                  |   ✔   |    ✔    |            |        |
|                 [HmeanIoU](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.HmeanIoU.html#mmeval.metrics.HmeanIoU)                  |   ✔   |         |            |        |
|             [PCKAccuracy](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.PCKAccuracy.html#mmeval.metrics.PCKAccuracy)             |   ✔   |         |            |        |
|       [MpiiPCKAccuracy](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.MpiiPCKAccuracy.html#mmeval.metrics.MpiiPCKAccuracy)       |   ✔   |         |            |        |
|     [JhmdbPCKAccuracy](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.JhmdbPCKAccuracy.html#mmeval.metrics.JhmdbPCKAccuracy)      |   ✔   |         |            |        |
|          [EndPointError](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.EndPointError.html#mmeval.metrics.EndPointError)          |   ✔   |    ✔    |            |        |
|                [AVAMeanAP](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.AVAMeanAP.html#mmeval.metrics.AVAMeanAP)                |   ✔   |         |            |        |
|                       [SSIM](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.SSIM.html#mmeval.metrics.SSIM)                        |   ✔   |         |            |        |
|                         [SNR](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.SNR.html#mmeval.metrics.SNR)                         |   ✔   |         |            |        |
|                       [PSNR](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.PSNR.html#mmeval.metrics.PSNR)                        |   ✔   |         |            |        |
|                         [MAE](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.MAE.html#mmeval.metrics.MAE)                         |   ✔   |         |            |        |
|                         [MSE](https://mmeval.readthedocs.io/zh_CN/latest/api/generated/mmeval.metrics.MSE.html#mmeval.metrics.MSE)                         |   ✔   |         |            |        |

</details>

## 安装

MMEval 依赖 Python 3.6+，可以通过 pip 来安装 MMEval。安装 MMEval 的过程中会安装一些 MMEval 运行时的依赖库：

```bash
pip install mmeval
```

如果要安装 MMEval 中所有评测指标都需要的依赖，可以通过以下命令安装：

```bash
pip install 'mmeval[all]'
```

## 快速上手

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

## 了解更多

<details>
<summary>入门教程</summary>

- [自定义评测指标](https://mmeval.readthedocs.io/zh_CN/latest/tutorials/custom_metric.html)

</details>

<details>
<summary>示例</summary>

- [MMCls](https://mmeval.readthedocs.io/zh_CN/latest/examples/mmclassification.html)
- [TensorPack](https://mmeval.readthedocs.io/zh_CN/latest/examples/tensorpack.html)
- [PaddleSeg](https://mmeval.readthedocs.io/zh_CN/latest/examples/paddleseg.html)

</details>

<details>
<summary>设计</summary>

- [BaseMetric 设计](https://mmeval.readthedocs.io/zh_CN/latest/design/base_metric.html)
- [分布式通信后端](https://mmeval.readthedocs.io/zh_CN/latest/design/distributed_backend.html)
- [基于参数类型注释的多分派](https://mmeval.readthedocs.io/zh_CN/latest/design/multiple_dispatch.html)

</details>

## 未来工作规划

- 持续添加更多评测指标，扩展更多任务领域（例如：NLP，语音等）。
- 支持更多机器学习框架，探索多机器学习框架支持模式。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMEngine 所作出的努力。请参考[贡献指南](CONTRIBUTING_zh-CN.md)来了解参与项目贡献的相关指引。

## 开源许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMLab 项目、算法、模型的统一入口
- [MMCV](https://github.com/open-mmlab/mmcv/tree/dev-2.x): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification/tree/dev-1.x): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/dev-3.x): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/dev-1.x): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate/tree/dev-1.x): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/dev-1.x): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr/tree/dev-1.x): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor/tree/dev-1.x): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2/tree/dev-1.x): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking/tree/dev-1.x): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow/tree/dev-1.x): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting/tree/dev-1.x): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration/tree/dev-1.x): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)，或通过添加微信“Open小喵Lab”加入官方交流微信群。

<div align="center">
<img src="https://user-images.githubusercontent.com/58739961/187154320-f3312cdf-31f2-4316-9dbb-8d7b0e1b7e08.jpg" height="400" />  <img src="https://user-images.githubusercontent.com/58739961/187151554-1a0748f0-a1bb-4565-84a6-ab3040247ef1.jpg" height="400" />  <img src="https://user-images.githubusercontent.com/58739961/187151778-d17c1368-125f-4fde-adbe-38cc6eb3be98.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
