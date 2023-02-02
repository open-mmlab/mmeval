# TensorPack

[TensorPack](https://github.com/tensorpack/tensorpack) is a neural net training interface on TensorFlow, with focus on speed + flexibility

There are many [examples](https://github.com/tensorpack/tensorpack/tree/master/examples) of classic models and tasks provided in the [TensorPack](https://github.com/tensorpack/tensorpack) repository. This section shows how to use [mmeval.COCODetection](mmeval.metrics.COCODetection) for evaluation in [TensorPack-FasterRCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN), and the related code can be found at [mmeval/examples/tensorpack](https://github.com/open-mmlab/mmeval/tree/main/examples/tensorpack).

First you need to install `TensorFlow` and `TensorPack`, then follow the preparation steps in the TensorPack-FasterRCNN example to install the dependencies and prepare the COCO dataset, as well as download the pre-trained model weights to be evaluated.

Scripts for model evaluation are provided in [predict.py](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN/predict.py), and the model can be evaluated with the following commands:

```bash
./predict.py --evaluate output.json --load /path/to Trained-Model-Checkpoint --config SAME-AS-TRAINING
```

`MMEval` provides a [evaluation tools](https://github.com/open-mmlab/mmeval/tree/main/examples/tensorpack/tensorpack_mmeval.py) for `TensorPack-FasterRCNN` that use [mmeval.COCODetection](mmeval.metrics.COCODetection). This evaluation script needs to be placed in the `TensorPack-FasterRCNN` example directory, and then the evaluation can be executed with the following command.

```bash
# run evaluation
python tensorpack_mmeval.py --load <model_path>

# launch multi-gpus evaluation by mpirun
mpirun -np 8 python tensorpack_mmeval.py --load <model_path>
```

We tested this evaluation script on [COCO-MaskRCNN-R50C41x](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz) and got the same evaluation results as the [TensorPack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN#results) report.

|                                           Model                                            | mAP (box) | mAP (mask) |  Configurations  |
| :----------------------------------------------------------------------------------------: | :-------: | :--------: | :--------------: |
| [COCO-MaskRCNN-R50C41x](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz) |   36.2    |    31.8    | `MODE_FPN=False` |
