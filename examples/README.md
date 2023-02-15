# MMEval Examples

Evaluation examples that use MMEval.

|                  Name                  | DL Framework |         Task          |                        Metric                        | Distributed Evaluation |
| :------------------------------------: | :----------: | :-------------------: | :--------------------------------------------------: | :--------------------: |
|    [CIFAR-10](./cifar10_dist_eval/)    | Torchvision  |    Classification     |      [Accuracy](../mmeval/metrics/accuracy.py)       |           ✔            |
| [TensorPack-FasterRCNN](./tensorpack/) |  TensorFlow  |   Object detection    | [COCODetection](../mmeval/metrics/coco_detection.py) |           ✔            |
|       [PaddelSeg](./paddleseg/)        |    Paddle    | Semantic segmentation |       [MeanIoU](../mmeval/metrics/mean_iou.py)       |           ✔            |
