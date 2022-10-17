# TensorPack Evaluation

This example provides an evaluation script that can be used to evaluate the model using `MMEval` in [TensorPack FasterRCNN Examples](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

This evaluation script is modified from

- https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/eval.py

with the following changes:

- Using `MMEval.CocoMetric`.
- Support multi-gpus evaluation by MPI4Py.

## Usage

1. Following the installation steps in [TensorPack](https://github.com/tensorpack/tensorpack) to install `TensorPack`.

2. Following the dependencies install steps in [tensorpack/examples/FasterRCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and preapare coco dataset.

3. Install mpi4py and mmeval.

4. Move the `tensorpack_mmeval.py` to [tensorpack/examples/FasterRCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and run evaluation.

```bash
# run evaluation
python tensorpack_mmeval.py --load <model_path>

# launch multi-gpus evaluation by mpirun
mpirun -np 8 python tensorpack_mmeval.py --load <model_path>
```

## Results

We tested this evaluation script on [COCO-MaskRCNN-R50C41x](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz) and got the same evaluation results as the [TensorPack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN#results) report.

Evaluation commands:

```bash
mpirun -np 8 python tensorpack_mmeval.py --load <model_path> --config MODE_FPN=False
```

|                                           Model                                            | mAP (box) | mAP (mask) |  Configurations  |
| :----------------------------------------------------------------------------------------: | :-------: | :--------: | :--------------: |
| [COCO-MaskRCNN-R50C41x](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz) |   36.2    |    31.8    | `MODE_FPN=False` |
