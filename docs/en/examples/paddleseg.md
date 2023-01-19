# PaddleSeg

[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) is a semantic segmentation algorithm library based on Paddle that supports many downstream tasks related to semantic segmentation.

This section shows how to use [mmeval.MeanIoU](mmeval.metrics.MeanIoU) for evaluation in PaddleSeg, and the related code can be found at [mmeval/examples/paddleseg](https://github.com/open-mmlab/mmeval/tree/main/examples/paddleseg).

First you need to install `Paddle` and `PaddleSeg`, you can refer to the [installation documentation](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/install_cn.md) in `PaddleSeg`. In addition, you need to download the pre-trained model to be evaluated, and prepare the evaluation data according to the configuration.

Scripts for model evaluation are provided in the `PaddleSeg` repo, and the model can be evaluated with the following commands:

```bash
python val.py --config <config_path> --model_path <model_path>
```

Note that the `val.py` script in the `PaddleSeg` only supports single-GPU evaluation, not multi-GPU evaluation yet.

`MMEval` provides a [evaluation tools](https://github.com/open-mmlab/mmeval/blob/main/examples/paddleseg/ppseg_mmeval.py) for `PaddleSeg` that use [mmeval.MeanIoU](mmeval.metrics.MeanIoU), which can be executed with the following command:

```bash
# run evaluation
python ppseg_mmeval.py --config <config_path> --model_path <model_path>

# run evaluation with multi-gpus
python ppseg_mmeval.py --config <config_path> --model_path <model_path> --launcher paddle --num_process <num_gpus>
```

We tested this evaluation script on [fastfcn_resnet50_os8_ade20k_480x480_120k](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/configs/fastfcn) and got the same evaluation results as the [val.py](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/val.py) in PaddleSeg.

|                                                                               Config                                                                                |                                                         Weights                                                          |  mIoU  |  aAcc  | Kappa  | mDice  |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: | :----: | :----: | :----: | :----: |
| [fastfcn_resnet50_os8_ade20k_480x480_120k](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml) | [model.pdparams](https://bj.bcebos.com/paddleseg/dygraph/ade20k/fastfcn_resnet50_os8_ade20k_480x480_120k/model.pdparams) | 0.4373 | 0.8074 | 0.7928 | 0.5772 |
