# Support Matrix

## Supported distributed communication backends

|                                                    MPI4Py                                                     |                                                                                                             torch.distributed                                                                                                              |                                                       Horovod                                                       |                                              paddle.distributed                                               |
| :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| [MPI4PyDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.MPI4PyDist) | [TorchCPUDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TorchCPUDist) <br> [TorchCUDADist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TorchCUDADist) | [TFHorovodDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TFHorovodDist) | [PaddleDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.PaddleDist) |

## Supported metrics and ML frameworks

```{note}
The following table lists the metrics implemented by MMEval and the corresponding machine learning framework support. A check mark indicates that the data type of the corresponding framework (e.g. Tensor) can be directly passed for computation.
```

|                                                       Metric                                                       | NumPy | PyTorch | TensorFlow | Paddle |
| :----------------------------------------------------------------------------------------------------------------: | :---: | :-----: | :--------: | :----: |
|                 [Accuracy](../api/generated/mmeval.metrics.Accuracy.html#mmeval.metrics.Accuracy)                  |   ✔   |    ✔    |            |        |
|    [SingleLabelMetric](../api/generated/mmeval.metrics.SingleLabelMetric.html#mmeval.metrics.SingleLabelMetric)    |   ✔   |    ✔    |            |        |
|     [MultiLabelMetric](../api/generated/mmeval.metrics.MultiLabelMetric.html#mmeval.metrics.MultiLabelMetric)      |   ✔   |    ✔    |            |        |
|     [AveragePrecision](../api/generated/mmeval.metrics.AveragePrecision.html#mmeval.metrics.AveragePrecision)      |   ✔   |    ✔    |            |        |
|                   [MeanIoU](../api/generated/mmeval.metrics.MeanIoU.html#mmeval.metrics.MeanIoU)                   |   ✔   |    ✔    |            |   ✔    |
|                [VOCMeanAP](../api/generated/mmeval.metrics.VOCMeanAP.html#mmeval.metrics.VOCMeanAP)                |   ✔   |         |            |        |
|                [OIDMeanAP](../api/generated/mmeval.metrics.OIDMeanAP.html#mmeval.metrics.OIDMeanAP)                |   ✔   |         |            |        |
| [CocoDetectionMetric](../api/generated/mmeval.metrics.COCODetectionMetric.html#mmeval.metrics.COCODetectionMetric) |   ✔   |         |            |        |
|        [ProposalRecall](../api/generated/mmeval.metrics.ProposalRecall.html#mmeval.metrics.ProposalRecall)         |   ✔   |         |            |        |
|                 [F1Metric](../api/generated/mmeval.metrics.F1Metric.html#mmeval.metrics.F1Metric)                  |   ✔   |    ✔    |            |        |
|                 [HmeanIoU](../api/generated/mmeval.metrics.HmeanIoU.html#mmeval.metrics.HmeanIoU)                  |   ✔   |         |            |        |
|             [PCKAccuracy](../api/generated/mmeval.metrics.PCKAccuracy.html#mmeval.metrics.PCKAccuracy)             |   ✔   |         |            |        |
|       [MpiiPCKAccuracy](../api/generated/mmeval.metrics.MpiiPCKAccuracy.html#mmeval.metrics.MpiiPCKAccuracy)       |   ✔   |         |            |        |
|     [JhmdbPCKAccuracy](../api/generated/mmeval.metrics.JhmdbPCKAccuracy.html#mmeval.metrics.JhmdbPCKAccuracy)      |   ✔   |         |            |        |
|          [EndPointError](../api/generated/mmeval.metrics.EndPointError.html#mmeval.metrics.EndPointError)          |   ✔   |    ✔    |            |        |
|                [AVAMeanAP](../api/generated/mmeval.metrics.AVAMeanAP.html#mmeval.metrics.AVAMeanAP)                |   ✔   |         |            |        |
|                       [SSIM](../api/generated/mmeval.metrics.SSIM.html#mmeval.metrics.SSIM)                        |   ✔   |         |            |        |
|                         [SNR](../api/generated/mmeval.metrics.SNR.html#mmeval.metrics.SNR)                         |   ✔   |         |            |        |
|                       [PSNR](../api/generated/mmeval.metrics.PSNR.html#mmeval.metrics.PSNR)                        |   ✔   |         |            |        |
|                         [MAE](../api/generated/mmeval.metrics.MAE.html#mmeval.metrics.MAE)                         |   ✔   |         |            |        |
|                         [MSE](../api/generated/mmeval.metrics.MSE.html#mmeval.metrics.MSE)                         |   ✔   |         |            |        |
