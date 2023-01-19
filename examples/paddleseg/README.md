# PaddleSeg Evaluation

This example provides an evaluation script that can be used to evaluate the model using `MMEval` in `PaddleSeg`.

This evaluation script is modified from

- https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/val.py
- https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/paddleseg/core/val.py

with the following changes:

- Using `MMEval.MeanIoU`.
- Support multi-gpus evaluation.

## Usage

1. Following the installation steps in [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/install.md).

2. Run evaluation

```bash
# run evaluation
python ppseg_mmeval.py --config <config_path> --model_path <model_path>

# run evaluation with multi-gpus
python ppseg_mmeval.py --config <config_path> --model_path <model_path> --launcher paddle --num_process <num_gpus>
```

## Results

We tested this evaluation script on [fastfcn_resnet50_os8_ade20k_480x480_120k](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/configs/fastfcn) and got the same evaluation results as the [val.py](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/val.py) in PaddleSeg.

|                                                                               Config                                                                                |                                                         Weights                                                          |  mIoU  |  aAcc  | Kappa  | mDice  |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: | :----: | :----: | :----: | :----: |
| [fastfcn_resnet50_os8_ade20k_480x480_120k](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml) | [model.pdparams](https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/model.pdparams) | 0.4373 | 0.8074 | 0.7928 | 0.5772 |
