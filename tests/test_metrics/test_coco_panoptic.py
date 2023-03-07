# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import os
import os.path as osp
import pytest
import tempfile
from json import dump

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import CocoPanoptic
from mmeval.metrics.coco_panoptic import INSTANCE_OFFSET
from mmeval.utils import try_import

try:
    from mmcv import imwrite
except ImportError:
    from mmeval.utils import imwrite

panopticapi = try_import('panopticapi')


def _create_panoptic_gt_annotations(ann_file, seg_map_dir):
    categories = [{
        'id': 0,
        'name': 'person',
        'supercategory': 'person',
        'isthing': 1
    }, {
        'id': 1,
        'name': 'cat',
        'supercategory': 'cat',
        'isthing': 1
    }, {
        'id': 2,
        'name': 'dog',
        'supercategory': 'dog',
        'isthing': 1
    }, {
        'id': 3,
        'name': 'wall',
        'supercategory': 'wall',
        'isthing': 0
    }]

    images = [{
        'id': 0,
        'width': 80,
        'height': 60,
        'file_name': 'fake_name1.jpg',
    }]

    annotations = [{
        'segments_info': [{
            'id': 1,
            'category_id': 0,
            'area': 400,
            'bbox': [10, 10, 10, 40],
            'iscrowd': 0
        }, {
            'id': 2,
            'category_id': 0,
            'area': 400,
            'bbox': [30, 10, 10, 40],
            'iscrowd': 0
        }, {
            'id': 3,
            'category_id': 2,
            'iscrowd': 0,
            'bbox': [50, 10, 10, 5],
            'area': 50
        }, {
            'id': 4,
            'category_id': 3,
            'iscrowd': 0,
            'bbox': [0, 0, 80, 60],
            'area': 3950
        }],
        'file_name':
        'fake_name1.png',
        'image_id':
        0
    }]

    gt_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # 4 is the id of the background class annotation.
    gt = np.zeros((60, 80), dtype=np.int64) + 4
    gt_bboxes = np.array([[10, 10, 10, 40], [30, 10, 10, 40], [50, 10, 10, 5]],
                         dtype=np.int64)
    for i in range(3):
        x, y, w, h = gt_bboxes[i]
        gt[y:y + h, x:x + w] = i + 1  # id starts from 1

    rgb_gt_seg_map = np.zeros(gt.shape + (3, ), dtype=np.uint8)
    rgb_gt_seg_map[:, :, 2] = gt // (256 * 256)
    rgb_gt_seg_map[:, :, 1] = gt % (256 * 256) // 256
    rgb_gt_seg_map[:, :, 0] = gt % 256
    img_path = osp.join(seg_map_dir, 'fake_name1.png')

    imwrite(rgb_gt_seg_map[:, :, ::-1], img_path)
    with open(ann_file, 'w') as f:
        dump(gt_json, f)

    return gt_json


def _create_panoptic_data_samples(gt_seg_dir):
    # predictions
    # TP for background class, IoU=3576/4324=0.827
    # 2 the category id of the background class
    pred = np.zeros((60, 80), dtype=np.int64) + 2
    pred_bboxes = np.array(
        [
            [11, 11, 10, 40],  # TP IoU=351/449=0.78
            [38, 10, 10, 40],  # FP
            [51, 10, 10, 5]  # TP IoU=45/55=0.818
        ],
        dtype=np.int64)
    pred_labels = np.array([0, 0, 1], dtype=np.int64)
    for i in range(3):
        x, y, w, h = pred_bboxes[i]
        pred[y:y + h, x:x + w] = (i + 1) * INSTANCE_OFFSET + pred_labels[i]

    segm_file = osp.basename('fake_name1.png')

    predictions = {'image_id': 0, 'sem_seg': pred, 'segm_file': segm_file}
    groundtruths = {
        'image_id':
        0,
        'width':
        80,
        'height':
        60,
        'segm_file':
        segm_file,
        'segments_info': [{
            'id': 1,
            'category': 0,
            'is_thing': 1
        }, {
            'id': 2,
            'category': 0,
            'is_thing': 1
        }, {
            'id': 3,
            'category': 1,
            'is_thing': 1
        }, {
            'id': 4,
            'category': 2,
            'is_thing': 0
        }],
    }

    return {'predictions': predictions, 'groundtruths': groundtruths}


@pytest.mark.skipif(
    panopticapi is None, reason='panopticapi is not available!')
def test_panoptic_metric_interface():

    fake_dataset_metas = {
        'classes': ('person', 'dog', 'wall'),
        'thing_classes': ('person', 'dog'),
        'stuff_classes': ('wall', )
    }

    tmp_dir = tempfile.TemporaryDirectory()
    gt_json_path = osp.join(tmp_dir.name, 'gt.json')
    gt_seg_dir = osp.join(tmp_dir.name, 'gt_seg')
    os.mkdir(gt_seg_dir)

    # create fake gt json.
    _create_panoptic_gt_annotations(gt_json_path, gt_seg_dir)
    # create fake predictions and groundtruths.
    data_samples = _create_panoptic_data_samples(gt_seg_dir)

    cp_data_samples = copy.deepcopy(data_samples)
    predictions = cp_data_samples['predictions']
    groundtruths = cp_data_samples['groundtruths']

    targets = {
        'PQ': 0.6786874803219071,
        'SQ': 0.8089770126158936,
        'RQ': 0.8333333333333334,
        'PQ_th': 0.6045252075318891,
        'SQ_th': 0.799959505972869,
        'RQ_th': 0.75,
        'PQ_st': 0.8270120259019427,
        'SQ_st': 0.8270120259019427,
        'RQ_st': 1.0
    }

    with pytest.raises(AssertionError):
        CocoPanoptic(format_only=True)

    # compute coco panoptic metrics with ann_file.
    panoptic_metric = CocoPanoptic(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=4,
        outfile_prefix=None,
        dataset_meta=fake_dataset_metas)
    assert isinstance(panoptic_metric, BaseMetric)

    eval_results = panoptic_metric([predictions], [groundtruths])

    assert eval_results == targets

    # compute coco panoptic metrics without ann_file.
    cp_data_samples = copy.deepcopy(data_samples)
    predictions = cp_data_samples['predictions']
    groundtruths = cp_data_samples['groundtruths']

    panoptic_metric = CocoPanoptic(
        ann_file=None,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=4,
        outfile_prefix=None,
        dataset_meta=fake_dataset_metas)

    eval_results = panoptic_metric([predictions], [groundtruths])
    assert eval_results == targets

    # compute coco panoptic metrics with outfile_prefix.
    cp_data_samples = copy.deepcopy(data_samples)
    predictions = cp_data_samples['predictions']
    groundtruths = cp_data_samples['groundtruths']

    panoptic_metric = CocoPanoptic(
        ann_file=None,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=4,
        outfile_prefix=osp.join(tmp_dir.name, 'outfile'),
        dataset_meta=fake_dataset_metas)
    eval_results = panoptic_metric([predictions], [groundtruths])
    assert eval_results == targets

    # compute coco panoptic metrics with outfile_prefix and nproc is 1.
    cp_data_samples = copy.deepcopy(data_samples)
    predictions = cp_data_samples['predictions']
    groundtruths = cp_data_samples['groundtruths']
    panoptic_metric = CocoPanoptic(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=1,
        outfile_prefix=osp.join(tmp_dir.name, 'outfile'),
        dataset_meta=fake_dataset_metas)
    eval_results = panoptic_metric([predictions], [groundtruths])
    assert eval_results == targets

    # compute coco panoptic metrics.
    cp_data_samples = copy.deepcopy(data_samples)
    predictions = cp_data_samples['predictions']
    groundtruths = cp_data_samples['groundtruths']
    panoptic_metric = CocoPanoptic(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=1,
        outfile_prefix=osp.join(tmp_dir.name, 'outfile'),
        dataset_meta=fake_dataset_metas)
    eval_results = panoptic_metric([predictions], [groundtruths])
    assert eval_results == targets

    # compute coco panoptic metrics with 2 times
    cp_data_samples = copy.deepcopy(data_samples)
    predictions = cp_data_samples['predictions']
    groundtruths = cp_data_samples['groundtruths']
    panoptic_metric = CocoPanoptic(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        outfile_prefix=None,
        dataset_meta=fake_dataset_metas)
    eval_results = panoptic_metric(
        [copy.deepcopy(predictions) for _ in range(2)],
        [copy.deepcopy(groundtruths) for _ in range(2)])
    assert eval_results == targets

    # compute coco panoptic metrics with classiwse
    cp_data_samples = copy.deepcopy(data_samples)
    predictions = cp_data_samples['predictions']
    groundtruths = cp_data_samples['groundtruths']
    panoptic_metric = CocoPanoptic(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=True,
        outfile_prefix=None,
        dataset_meta=fake_dataset_metas)
    eval_results = panoptic_metric(
        [copy.deepcopy(predictions) for _ in range(2)],
        [copy.deepcopy(groundtruths) for _ in range(2)])
    classwise_targets = {
        'person_PQ': 0.3908685968819599,
        'dog_PQ': 0.8181818181818182,
        'wall_PQ': 0.8270120259019427
    }
    classwise_targets.update(targets)
    assert eval_results == classwise_targets

    tmp_dir.cleanup()
