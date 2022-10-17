# MMEval Examples

Evaluation examples that use MMEval.

|                  Name                  | DL Framework |         Task          |                     Metric                     | Distributed Evaluation |
| :------------------------------------: | :----------: | :-------------------: | :--------------------------------------------: | :--------------------: |
| [TensorPack-FasterRCNN](./tensorpack/) |  TensorFlow  |   Object detection    | [CocoMetric](../mmeval/metrics/coco_metric.py) |           ✔            |
|       [PaddelSeg](./paddleseg/)        |    Paddle    | Semantic segmentation |    [MeanIoU](../mmeval/metrics/mean_iou.py)    |           ✔            |
