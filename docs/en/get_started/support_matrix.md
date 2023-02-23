# Support Matrix

## Supported distributed communication backends

|                                                    MPI4Py                                                     |                                                                                                                torch.distributed                                                                                                                |                                                        Horovod                                                         |                                              paddle.distributed                                               |                                                   oneflow.comm                                                   |
| :-----------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| [MPI4PyDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.MPI4PyDist) | [TorchCPUDist](../api/generated/mmeval.core.dist_backends.TorchCPUDist.html#mmeval.core.dist_backends.TorchCPUDist) <br> [TorchCUDADist](../api/generated/mmeval.core.dist_backends.TorchCUDADist.html#mmeval.core.dist_backends.TorchCUDADist) | [TFHorovodDist](../api/generated/mmeval.core.dist_backends.TFHorovodDist.html#mmeval.core.dist_backends.TFHorovodDist) | [PaddleDist](../api/generated/mmeval.core.dist_backends.PaddleDist.html#mmeval.core.dist_backends.PaddleDist) | [OneFlowDist](../api/generated/mmeval.core.dist_backends.OneFlowDist.html#mmeval.core.dist_backends.OneFlowDist) |

## Supported metrics and ML frameworks

```{note}
The following table lists the metrics implemented by MMEval and the corresponding machine learning framework support. A check mark indicates that the data type of the corresponding framework (e.g. Tensor) can be directly passed for computation.
```

```{note}
MMEval tested with PyTorch 1.6+, TensorFlow 2.4+, Paddle 2.2+ and OneFlow 0.8+.
```

|                                                                    Metric                                                                     | numpy.ndarray | torch.Tensor | tensorflow.Tensor | paddle.Tensor | oneflow.Tensor |
| :-------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :----------: | :---------------: | :-----------: | :------------: |
|                               [Accuracy](../api/generated/mmeval.metrics.Accuracy.html#mmeval.metrics.Accuracy)                               |       ✔       |      ✔       |         ✔         |       ✔       |       ✔        |
|                 [SingleLabelMetric](../api/generated/mmeval.metrics.SingleLabelMetric.html#mmeval.metrics.SingleLabelMetric)                  |       ✔       |      ✔       |                   |               |       ✔        |
|                   [MultiLabelMetric](../api/generated/mmeval.metrics.MultiLabelMetric.html#mmeval.metrics.MultiLabelMetric)                   |       ✔       |      ✔       |                   |               |       ✔        |
|                   [AveragePrecision](../api/generated/mmeval.metrics.AveragePrecision.html#mmeval.metrics.AveragePrecision)                   |       ✔       |      ✔       |                   |               |       ✔        |
|                                [MeanIoU](../api/generated/mmeval.metrics.MeanIoU.html#mmeval.metrics.MeanIoU)                                 |       ✔       |      ✔       |         ✔         |       ✔       |       ✔        |
|                             [VOCMeanAP](../api/generated/mmeval.metrics.VOCMeanAP.html#mmeval.metrics.VOCMeanAP)                              |       ✔       |              |                   |               |                |
|                             [OIDMeanAP](../api/generated/mmeval.metrics.OIDMeanAP.html#mmeval.metrics.OIDMeanAP)                              |       ✔       |              |                   |               |                |
|                       [COCODetection](../api/generated/mmeval.metrics.COCODetection.html#mmeval.metrics.COCODetection)                        |       ✔       |              |                   |               |                |
|                      [ProposalRecall](../api/generated/mmeval.metrics.ProposalRecall.html#mmeval.metrics.ProposalRecall)                      |       ✔       |              |                   |               |                |
|                                [F1Score](../api/generated/mmeval.metrics.F1Score.html#mmeval.metrics.F1Score)                                 |       ✔       |      ✔       |                   |               |       ✔        |
|                               [HmeanIoU](../api/generated/mmeval.metrics.HmeanIoU.html#mmeval.metrics.HmeanIoU)                               |       ✔       |              |                   |               |                |
|                          [PCKAccuracy](../api/generated/mmeval.metrics.PCKAccuracy.html#mmeval.metrics.PCKAccuracy)                           |       ✔       |              |                   |               |                |
|                    [MpiiPCKAccuracy](../api/generated/mmeval.metrics.MpiiPCKAccuracy.html#mmeval.metrics.MpiiPCKAccuracy)                     |       ✔       |              |                   |               |                |
|                   [JhmdbPCKAccuracy](../api/generated/mmeval.metrics.JhmdbPCKAccuracy.html#mmeval.metrics.JhmdbPCKAccuracy)                   |       ✔       |              |                   |               |                |
|                       [EndPointError](../api/generated/mmeval.metrics.EndPointError.html#mmeval.metrics.EndPointError)                        |       ✔       |      ✔       |                   |               |       ✔        |
|                             [AVAMeanAP](../api/generated/mmeval.metrics.AVAMeanAP.html#mmeval.metrics.AVAMeanAP)                              |       ✔       |              |                   |               |                |
|             [StructuralSimilarity](../api/generated/mmeval.metrics.StructuralSimilarity.html#mmeval.metrics.StructuralSimilarity)             |       ✔       |              |                   |               |                |
|                   [SignalNoiseRatio](../api/generated/mmeval.metrics.SignalNoiseRatio.html#mmeval.metrics.SignalNoiseRatio)                   |       ✔       |              |                   |               |                |
|             [PeakSignalNoiseRatio](../api/generated/mmeval.metrics.PeakSignalNoiseRatio.html#mmeval.metrics.PeakSignalNoiseRatio)             |       ✔       |              |                   |               |                |
|                 [MeanAbsoluteError](../api/generated/mmeval.metrics.MeanAbsoluteError.html#mmeval.metrics.MeanAbsoluteError)                  |       ✔       |              |                   |               |                |
|                   [MeanSquaredError](../api/generated/mmeval.metrics.MeanSquaredError.html#mmeval.metrics.MeanSquaredError)                   |       ✔       |              |                   |               |                |
| [NaturalImageQualityEvaluator](../api/generated/mmeval.metrics.NaturalImageQualityEvaluator.html#mmeval.metrics.NaturalImageQualityEvaluator) |       ✔       |              |                   |               |                |
