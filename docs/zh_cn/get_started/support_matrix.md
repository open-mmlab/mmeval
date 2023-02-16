# 支持矩阵

### 支持的分布式通信后端

|                                                    MPI4Py                                                     |                                                                                                                torch.distributed                                                                                                                |                                                        Horovod                                                         |                                              paddle.distributed                                               |                                                   oneflow.comm                                                   |
| :-----------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| [MPI4PyDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.MPI4PyDist) | [TorchCPUDist](../api/generated/mmeval.core.dist_backends.TorchCPUDist.html#mmeval.core.dist_backends.TorchCPUDist) <br> [TorchCUDADist](../api/generated/mmeval.core.dist_backends.TorchCUDADist.html#mmeval.core.dist_backends.TorchCUDADist) | [TFHorovodDist](../api/generated/mmeval.core.dist_backends.TFHorovodDist.html#mmeval.core.dist_backends.TFHorovodDist) | [PaddleDist](../api/generated/mmeval.core.dist_backends.PaddleDist.html#mmeval.core.dist_backends.PaddleDist) | [OneFlowDist](../api/generated/mmeval.core.dist_backends.OneFlowDist.html#mmeval.core.dist_backends.OneFlowDist) |

### 支持的评测指标及机器学习框架

```{note}
下表列出 MMEval 已实现的评测指标与对应的机器学习框架支持情况，打勾表示能够直接接收对应框架的数据类型（如 Tensor）进行计算。
```

```{note}
MMEval 在 PyTorch 1.6+，TensorFlow 2.4+, Paddle 2.2+ 和 OneFlow 0.8+ 测试通过。
```

|                                                         评测指标                                                         | numpy.ndarray | torch.Tensor | tensorflow.Tensor | paddle.Tensor | oneflow.Tensor |
| :----------------------------------------------------------------------------------------------------------------------: | :-----------: | :----------: | :---------------: | :-----------: | :------------: |
|                    [Accuracy](../api/generated/mmeval.metrics.Accuracy.html#mmeval.metrics.Accuracy)                     |       ✔       |      ✔       |         ✔         |       ✔       |       ✔        |
|       [SingleLabelMetric](../api/generated/mmeval.metrics.SingleLabelMetric.html#mmeval.metrics.SingleLabelMetric)       |       ✔       |      ✔       |                   |               |       ✔        |
|        [MultiLabelMetric](../api/generated/mmeval.metrics.MultiLabelMetric.html#mmeval.metrics.MultiLabelMetric)         |       ✔       |      ✔       |                   |               |       ✔        |
|        [AveragePrecision](../api/generated/mmeval.metrics.AveragePrecision.html#mmeval.metrics.AveragePrecision)         |       ✔       |      ✔       |                   |               |       ✔        |
|                      [MeanIoU](../api/generated/mmeval.metrics.MeanIoU.html#mmeval.metrics.MeanIoU)                      |       ✔       |      ✔       |         ✔         |       ✔       |       ✔        |
|                   [VOCMeanAP](../api/generated/mmeval.metrics.VOCMeanAP.html#mmeval.metrics.VOCMeanAP)                   |       ✔       |              |                   |               |                |
|                   [OIDMeanAP](../api/generated/mmeval.metrics.OIDMeanAP.html#mmeval.metrics.OIDMeanAP)                   |       ✔       |              |                   |               |                |
|             [COCODetection](../api/generated/mmeval.metrics.COCODetection.html#mmeval.metrics.COCODetection)             |       ✔       |              |                   |               |                |
|           [ProposalRecall](../api/generated/mmeval.metrics.ProposalRecall.html#mmeval.metrics.ProposalRecall)            |       ✔       |              |                   |               |                |
|                      [F1Score](../api/generated/mmeval.metrics.F1Score.html#mmeval.metrics.F1Score)                      |       ✔       |      ✔       |                   |               |       ✔        |
|                    [HmeanIoU](../api/generated/mmeval.metrics.HmeanIoU.html#mmeval.metrics.HmeanIoU)                     |       ✔       |              |                   |               |                |
|                [PCKAccuracy](../api/generated/mmeval.metrics.PCKAccuracy.html#mmeval.metrics.PCKAccuracy)                |       ✔       |              |                   |               |                |
|          [MpiiPCKAccuracy](../api/generated/mmeval.metrics.MpiiPCKAccuracy.html#mmeval.metrics.MpiiPCKAccuracy)          |       ✔       |              |                   |               |                |
|        [JhmdbPCKAccuracy](../api/generated/mmeval.metrics.JhmdbPCKAccuracy.html#mmeval.metrics.JhmdbPCKAccuracy)         |       ✔       |              |                   |               |                |
|             [EndPointError](../api/generated/mmeval.metrics.EndPointError.html#mmeval.metrics.EndPointError)             |       ✔       |      ✔       |                   |               |       ✔        |
|                   [AVAMeanAP](../api/generated/mmeval.metrics.AVAMeanAP.html#mmeval.metrics.AVAMeanAP)                   |       ✔       |              |                   |               |                |
|  [StructuralSimilarity](../api/generated/mmeval.metrics.StructuralSimilarity.html#mmeval.metrics.StructuralSimilarity)   |       ✔       |              |                   |               |                |
|        [SignalNoiseRatio](../api/generated/mmeval.metrics.SignalNoiseRatio.html#mmeval.metrics.SignalNoiseRatio)         |       ✔       |              |                   |               |                |
|  [PeakSignalNoiseRatio](../api/generated/mmeval.metrics.PeakSignalNoiseRatio.html#mmeval.metrics.PeakSignalNoiseRatio)   |       ✔       |              |                   |               |                |
|       [MeanAbsoluteError](../api/generated/mmeval.metrics.MeanAbsoluteError.html#mmeval.metrics.MeanAbsoluteError)       |       ✔       |              |                   |               |                |
|        [MeanSquaredError](../api/generated/mmeval.metrics.MeanSquaredError.html#mmeval.metrics.MeanSquaredError)         |       ✔       |              |                   |               |                |
| [KeypointEndPointError](../api/generated/mmeval.metrics.KeypointEndPointError.html#mmeval.metrics.KeypointEndPointError) |       ✔       |              |                   |               |                |
