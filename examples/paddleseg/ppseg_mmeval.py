# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A evlaution script for PaddleSeg that use MMEval.

Modified from:
- https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/val.py
- https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/paddleseg/core/val.py
"""

# flake8: noqa

import argparse
import numpy as np
import paddle
import paddle.nn.functional as F
import time
from paddleseg.core import infer
from paddleseg.cvlibs import Config
from paddleseg.utils import (TimeAverager, config_check, get_sys_env, logger,
                             metrics, progbar, utils)

from mmeval import MeanIoU

np.set_printoptions(suppress=True)


def evaluate(model,
             eval_dataset,
             aug_eval=False,
             scales=1.0,
             flip_horizontal=False,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             precision='fp32',
             amp_level='O1',
             num_workers=0,
             print_detail=True,
             auc_roc=False):
    """Launch evaluation.

    Args:
        model（nn.Layer): A semantic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    local_rank = paddle.distributed.ParallelEnv().local_rank

    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True)

    total_iters = len(loader)
    logits_all = None
    label_all = None

    if print_detail:
        logger.info(
            'Start evaluating (total_samples: {}, total_iters: {})...'.format(
                len(eval_dataset), total_iters))
    #TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    mean_iou = MeanIoU(
        num_classes=eval_dataset.num_classes,
        ignore_index=eval_dataset.ignore_index,
        classwise_results=True,
        dist_backend='paddle_dist')

    with paddle.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = data['label'].astype('int64')

            if aug_eval:
                if precision == 'fp16':
                    with paddle.amp.auto_cast(
                            level=amp_level,
                            enable=True,
                            custom_white_list={
                                'elementwise_add', 'batch_norm',
                                'sync_batch_norm'
                            },
                            custom_black_list={'bilinear_interp_v2'}):
                        pred, logits = infer.aug_inference(
                            model,
                            data['img'],
                            trans_info=data['trans_info'],
                            scales=scales,
                            flip_horizontal=flip_horizontal,
                            flip_vertical=flip_vertical,
                            is_slide=is_slide,
                            stride=stride,
                            crop_size=crop_size)
                else:
                    pred, logits = infer.aug_inference(
                        model,
                        data['img'],
                        trans_info=data['trans_info'],
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
            else:
                if precision == 'fp16':
                    with paddle.amp.auto_cast(
                            level=amp_level,
                            enable=True,
                            custom_white_list={
                                'elementwise_add', 'batch_norm',
                                'sync_batch_norm'
                            },
                            custom_black_list={'bilinear_interp_v2'}):
                        pred, logits = infer.inference(
                            model,
                            data['img'],
                            trans_info=data['trans_info'],
                            is_slide=is_slide,
                            stride=stride,
                            crop_size=crop_size)
                else:
                    pred, logits = infer.inference(
                        model,
                        data['img'],
                        trans_info=data['trans_info'],
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)

            mean_iou.add(
                paddle.squeeze(pred, axis=1), paddle.squeeze(label, axis=1))

            if auc_roc:
                logits = F.softmax(logits, axis=1)
                if logits_all is None:
                    logits_all = logits.numpy()
                    label_all = label.numpy()
                else:
                    logits_all = np.concatenate([logits_all,
                                                 logits.numpy()
                                                 ])  # (KN, C, H, W)
                    label_all = np.concatenate([label_all, label.numpy()])

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    metric_results = mean_iou.compute(len(eval_dataset))
    class_iou = metric_results['classwise_results']['IoU']
    miou = metric_results['mIoU']
    acc = metric_results['aAcc']
    class_precision = metric_results['classwise_results']['Precision']
    class_recall = metric_results['classwise_results']['Recall']
    mdice = metric_results['mDice']
    kappa = metric_results['kappa']

    if auc_roc:
        auc_roc = metrics.auc_roc(
            logits_all, label_all, num_classes=eval_dataset.num_classes)
        auc_info = f' Auc_roc: {auc_roc:.4f}'

    if print_detail:
        info = '[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}'.format(
            len(eval_dataset), miou, acc, kappa, mdice)
        info = info + auc_info if auc_roc else info
        logger.info(info)
        logger.info('[EVAL] Class IoU: \n' + str(np.round(class_iou, 4)))
        logger.info('[EVAL] Class Precision: \n' +
                    str(np.round(class_precision, 4)))
        logger.info('[EVAL] Class Recall: \n' + str(np.round(class_recall, 4)))
    return miou, acc, class_iou, class_precision, kappa


def get_test_config(cfg, args):

    test_config = cfg.test_config
    if args.aug_eval:
        test_config['aug_eval'] = args.aug_eval
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    return test_config


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        '--config',
        dest='cfg',
        help='The config file.',
        default=None,
        type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)

    # augment for evaluation
    parser.add_argument(
        '--aug_eval',
        dest='aug_eval',
        help='Whether to use mulit-scales and flip augment for evaluation',
        action='store_true')
    parser.add_argument(
        '--scales',
        dest='scales',
        nargs='+',
        help='Scales for augment',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        dest='flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        dest='flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # sliding window evaluation
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to evaluate by sliding window',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help=
        'The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help=
        'The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)

    parser.add_argument(
        '--data_format',
        dest='data_format',
        help=
        'Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')

    parser.add_argument(
        '--auc_roc',
        dest='add auc_roc metric',
        help='Whether to use auc_roc metric',
        type=bool,
        default=False)

    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be GPU, XPU, NPU, CPU',
        default='gpu',
        type=str)

    parser.add_argument(
        '--launcher',
        choices=['none', 'paddle'],
        default='none',
        help='job launcher')

    parser.add_argument(
        '--num_process',
        dest='num_process',
        help='Num process for evaluation',
        type=int,
        default=0)

    return parser.parse_args()


def main(args):
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        paddle.distributed.init_parallel_env()
        paddle.set_device(f'gpu:{local_rank}')

    env_info = get_sys_env()

    if args.device == 'gpu' and env_info[
            'Paddle compiled with cuda'] and env_info['GPUs used']:
        place = 'gpu'
    elif args.device == 'xpu' and paddle.is_compiled_with_xpu():
        place = 'xpu'
    elif args.device == 'npu' and paddle.is_compiled_with_npu():
        place = 'npu'
    else:
        place = 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    # Only support for the DeepLabv3+ model
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format

    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    test_config = get_test_config(cfg, args)
    config_check(cfg, val_dataset=val_dataset)

    evaluate(model, val_dataset, num_workers=args.num_workers, **test_config)


if __name__ == '__main__':
    args = parse_args()
    if args.launcher == 'paddle':
        paddle.distributed.spawn(
            main, nprocs=args.num_process, args=(args, ), backend='nccl')
    else:
        main(args)
