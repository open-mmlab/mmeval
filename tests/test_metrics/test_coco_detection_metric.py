# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
import pytest
import tempfile
from json import dump

from mmeval.core.base_metric import BaseMetric
from mmeval.metrics import COCODetection
from mmeval.utils import try_import

coco_wrapper = try_import('mmeval.metrics.utils.coco_wrapper')


def _create_dummy_coco_json(json_name):
    # create and dump fake json file
    dummy_mask = np.zeros((10, 10), order='F', dtype=np.uint8)
    dummy_mask[:5, :5] = 1
    rle_mask = coco_wrapper.mask_util.encode(dummy_mask)
    rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 400,
        'bbox': [50, 60, 20, 20],
        'iscrowd': 0,
        'segmentation': rle_mask,
    }

    annotation_2 = {
        'id': 2,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
        'segmentation': rle_mask,
    }

    annotation_3 = {
        'id': 3,
        'image_id': 0,
        'category_id': 1,
        'area': 1600,
        'bbox': [150, 160, 40, 40],
        'iscrowd': 0,
        'segmentation': rle_mask,
    }

    annotation_4 = {
        'id': 4,
        'image_id': 0,
        'category_id': 0,
        'area': 10000,
        'bbox': [250, 260, 100, 100],
        'iscrowd': 0,
        'segmentation': rle_mask,
    }

    categories = [
        {
            'id': 0,
            'name': 'car',
            'supercategory': 'car',
        },
        {
            'id': 1,
            'name': 'bicycle',
            'supercategory': 'bicycle',
        },
    ]

    fake_json = {
        'images': [image],
        'annotations':
        [annotation_1, annotation_2, annotation_3, annotation_4],
        'categories': categories
    }
    with open(json_name, 'w') as f:
        dump(fake_json, f)


def _create_dummy_results():
    # create fake results
    bboxes = np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                       [150, 160, 190, 200], [250, 260, 350, 360]])
    scores = np.array([1.0, 0.98, 0.96, 0.95])
    labels = np.array([0, 0, 1, 0])

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :5] = 1

    dummy_mask = [
        coco_wrapper.mask_util.encode(
            np.array(mask[:, :, np.newaxis], order='F', dtype='uint8'))[0]
        for _ in range(4)
    ]
    return dict(
        img_id=0,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        masks=dummy_mask)


def _create_dummy_gts():
    # create fake gts
    bboxes = np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                       [150, 160, 190, 200], [250, 260, 350, 360]])
    labels = np.array([0, 0, 1, 0])
    ignore_flags = np.array([0, 0, 0, 0])
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :5] = 1

    dummy_mask = [
        coco_wrapper.mask_util.encode(
            np.array(mask[:, :, np.newaxis], order='F', dtype='uint8'))[0]
        for _ in range(4)
    ]
    return dict(
        img_id=0,
        width=640,
        height=640,
        bboxes=bboxes,
        labels=labels,
        ignore_flags=ignore_flags,
        masks=dummy_mask)


# TODO: move necessary function to somewhere
def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
    # random generate bounding boxes in 'xyxy' formart.
    x = np.random.rand(num_bboxes, ) * img_w
    y = np.random.rand(num_bboxes, ) * img_h
    w = np.random.rand(num_bboxes, ) * (img_w - x)
    h = np.random.rand(num_bboxes, ) * (img_h - y)
    return np.stack([x, y, x + w, y + h], axis=1)


def _gen_masks(bboxes, img_w=256, img_h=256):
    # random generate masks
    if coco_wrapper is None:
        raise ImportError('Please try to install official pycocotools by '
                          '"pip install pycocotools"')
    masks = []
    for i, bbox in enumerate(bboxes):
        mask = np.zeros((img_h, img_w))
        bbox = bbox.astype(np.int32)
        box_mask = (np.random.rand(bbox[3] - bbox[1], bbox[2] - bbox[0]) >
                    0.3).astype(np.int32)
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = box_mask
        masks.append(
            coco_wrapper.mask_util.encode(
                np.array(mask[:, :, np.newaxis], order='F',
                         dtype='uint8'))[0])  # encoded with RLE
    return masks


def _gen_prediction(num_pred=10,
                    num_classes=10,
                    img_w=256,
                    img_h=256,
                    img_id=1,
                    with_mask=False):
    # random create prediction
    pred_boxes = _gen_bboxes(num_bboxes=num_pred, img_w=img_w, img_h=img_h)
    prediction = {
        'img_id': img_id,
        'bboxes': pred_boxes,
        'scores': np.random.rand(num_pred, ),
        'labels': np.random.randint(0, num_classes, size=(num_pred, ))
    }
    if with_mask:
        pred_masks = _gen_masks(bboxes=pred_boxes, img_w=img_w, img_h=img_h)
        prediction['masks'] = pred_masks
    return prediction


def _gen_groundtruth(num_gt=10,
                     num_classes=10,
                     img_w=256,
                     img_h=256,
                     img_id=1,
                     with_mask=False):
    # random create prediction
    gt_boxes = _gen_bboxes(num_bboxes=num_gt, img_w=img_w, img_h=img_h)
    groundtruth = {
        'img_id': img_id,
        'width': img_w,
        'height': img_h,
        'bboxes': gt_boxes,
        'labels': np.random.randint(0, num_classes, size=(num_gt, )),
        'ignore_flags': np.zeros(num_gt)
    }
    if with_mask:
        pred_masks = _gen_masks(bboxes=gt_boxes, img_w=img_w, img_h=img_h)
        groundtruth['masks'] = pred_masks
    return groundtruth


@pytest.mark.skipif(
    coco_wrapper is None, reason='coco_wrapper is not available!')
@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {
            'ann_file': 'tests/test_metrics/data/coco_sample.json'
        },
        {
            'iou_thrs': [0.5, 0.75]
        },
        {
            'classwise': True
        },
        {
            'metric_items': ['mAP', 'mAP_50']
        },
        {
            'proposal_nums': [10, 30, 100]
        },
    ])
def test_box_metric_interface(metric_kwargs):
    num_classes = 10
    metric = ['bbox']
    # Avoid some potential error
    fake_dataset_metas = {
        'classes': tuple([str(i) for i in range(num_classes)])
    }
    coco_det_metric = COCODetection(
        metric=metric, dataset_meta=fake_dataset_metas, **metric_kwargs)
    assert isinstance(coco_det_metric, BaseMetric)

    metric_results = coco_det_metric(
        predictions=[
            _gen_prediction(num_classes=num_classes) for _ in range(10)
        ],
        groundtruths=[
            _gen_groundtruth(num_classes=num_classes) for _ in range(10)
        ],
    )
    assert isinstance(metric_results, dict)
    assert 'bbox_mAP' in metric_results


@pytest.mark.skipif(
    coco_wrapper is None, reason='coco_wrapper is not available!')
@pytest.mark.parametrize(
    argnames='metric_kwargs',
    argvalues=[
        {},
        {
            'ann_file': 'tests/test_metrics/data/coco_sample.json'
        },
        {
            'iou_thrs': [0.5, 0.75]
        },
        {
            'classwise': True
        },
        {
            'metric_items': ['mAP', 'mAP_50']
        },
        {
            'proposal_nums': [10, 30, 100]
        },
    ])
def test_segm_metric_interface(metric_kwargs):
    num_classes = 10
    metric = ['segm']
    # Avoid some potential error
    fake_dataset_metas = {
        'classes': tuple([str(i) for i in range(num_classes)])
    }
    coco_det_metric = COCODetection(
        metric=metric, dataset_meta=fake_dataset_metas, **metric_kwargs)
    assert isinstance(coco_det_metric, BaseMetric)

    metric_results = coco_det_metric(
        predictions=[
            _gen_prediction(num_classes=num_classes, with_mask=True)
            for _ in range(10)
        ],
        groundtruths=[
            _gen_groundtruth(num_classes=num_classes, with_mask=True)
            for _ in range(10)
        ],
    )
    assert isinstance(metric_results, dict)
    assert 'segm_mAP' in metric_results


@pytest.mark.skipif(
    coco_wrapper is None, reason='coco_wrapper is not available!')
def test_metric_invalid_usage():
    with pytest.raises(KeyError):
        COCODetection(metric='xxx')

    with pytest.raises(TypeError):
        COCODetection(iou_thrs=1)

    with pytest.raises(AssertionError):
        COCODetection(format_only=True)

    num_classes = 10
    # Avoid some potential error
    fake_dataset_metas = {
        'classes': tuple([str(i) for i in range(num_classes)])
    }
    coco_det_metric = COCODetection(dataset_meta=fake_dataset_metas)

    with pytest.raises(KeyError):
        prediction = _gen_prediction(num_classes=num_classes)
        groundtruth = _gen_groundtruth(num_classes=num_classes)
        del prediction['bboxes']
        coco_det_metric([prediction], [groundtruth])

    with pytest.raises(AssertionError):
        prediction = _gen_prediction(num_classes=num_classes)
        groundtruth = _gen_groundtruth(num_classes=num_classes)
        coco_det_metric(prediction, groundtruth)


@pytest.mark.skipif(
    coco_wrapper is None, reason='coco_wrapper is not available!')
def test_compute_metric():
    tmp_dir = tempfile.TemporaryDirectory()

    # create dummy data
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_dummy_coco_json(fake_json_file)
    dummy_pred = _create_dummy_results()
    fake_dataset_metas = dict(classes=['car', 'bicycle'])

    # test single coco dataset evaluation
    coco_det_metric = COCODetection(
        ann_file=fake_json_file,
        classwise=False,
        outfile_prefix=f'{tmp_dir.name}/test',
        dataset_meta=fake_dataset_metas)
    eval_results = coco_det_metric([dummy_pred], [dict()])
    target = {
        'bbox_mAP': 1.0,
        'bbox_mAP_50': 1.0,
        'bbox_mAP_75': 1.0,
        'bbox_mAP_s': 1.0,
        'bbox_mAP_m': 1.0,
        'bbox_mAP_l': 1.0,
    }
    results = {k: round(v, 4) for k, v in eval_results.items()}
    assert results == target
    assert osp.isfile(osp.join(tmp_dir.name, 'test.bbox.json'))

    # test box and segm coco dataset evaluation
    coco_det_metric = COCODetection(
        ann_file=fake_json_file,
        classwise=False,
        metric=['bbox', 'segm'],
        outfile_prefix=f'{tmp_dir.name}/test',
        dataset_meta=fake_dataset_metas)
    eval_results = coco_det_metric([dummy_pred], [dict()])
    target = {
        'bbox_mAP': 1.0,
        'bbox_mAP_50': 1.0,
        'bbox_mAP_75': 1.0,
        'bbox_mAP_s': 1.0,
        'bbox_mAP_m': 1.0,
        'bbox_mAP_l': 1.0,
        'segm_mAP': 1.0,
        'segm_mAP_50': 1.0,
        'segm_mAP_75': 1.0,
        'segm_mAP_s': 1.0,
        'segm_mAP_m': 1.0,
        'segm_mAP_l': 1.0,
    }
    results = {k: round(v, 4) for k, v in eval_results.items()}
    assert results == target
    assert osp.isfile(osp.join(tmp_dir.name, 'test.bbox.json'))
    assert osp.isfile(osp.join(tmp_dir.name, 'test.segm.json'))

    # test classwise result evaluation
    coco_det_metric = COCODetection(
        ann_file=fake_json_file,
        classwise=True,
        outfile_prefix=f'{tmp_dir.name}/test',
        dataset_meta=fake_dataset_metas)
    eval_results = coco_det_metric([dummy_pred], [dict()])
    target = {
        'bbox_mAP': 1.0,
        'bbox_mAP_50': 1.0,
        'bbox_mAP_75': 1.0,
        'bbox_mAP_s': 1.0,
        'bbox_mAP_m': 1.0,
        'bbox_mAP_l': 1.0,
        'bbox_car_precision': 1.0,
        'bbox_bicycle_precision': 1.0,
    }
    results = {k: round(v, 4) for k, v in eval_results.items()}
    assert results == target

    # test proposal
    coco_det_metric = COCODetection(
        ann_file=fake_json_file,
        metric='proposal',
        classwise=False,
        outfile_prefix=f'{tmp_dir.name}/test',
        dataset_meta=fake_dataset_metas)
    eval_results = coco_det_metric([dummy_pred], [dict()])
    target = {
        'AR@1': 0.25,
        'AR@10': 1.0,
        'AR@100': 1.0,
        'AR_s@100': 1.0,
        'AR_m@100': 1.0,
        'AR_l@100': 1.0
    }
    results = {k: round(v, 4) for k, v in eval_results.items()}
    assert results == target

    # test empty results
    coco_det_metric = COCODetection(
        ann_file=fake_json_file,
        classwise=False,
        outfile_prefix=f'{tmp_dir.name}/test',
        dataset_meta=fake_dataset_metas)

    empty_pred = _gen_prediction(num_pred=0, num_classes=2, img_id=1)
    eval_results = coco_det_metric([empty_pred], [dict()])
    assert eval_results == dict()

    # test format only evaluation
    coco_det_metric = COCODetection(
        ann_file=fake_json_file,
        classwise=False,
        format_only=True,
        outfile_prefix=f'{tmp_dir.name}/test',
        dataset_meta=fake_dataset_metas)
    eval_results = coco_det_metric([dummy_pred], [dict()])
    assert osp.exists(f'{tmp_dir.name}/test.bbox.json')
    assert eval_results == dict()

    # test evaluate metric without loading ann_file
    # the gt instance area based on mask
    dummy_gt = _create_dummy_gts()
    coco_det_metric = COCODetection(
        outfile_prefix=f'{tmp_dir.name}/test',
        metric=['bbox', 'segm'],
        dataset_meta=fake_dataset_metas,
        gt_mask_area=True)
    target = {
        'bbox_mAP': 1.0,
        'bbox_mAP_50': 1.0,
        'bbox_mAP_75': 1.0,
        'bbox_mAP_s': 1.0,
        'bbox_mAP_m': -1.0,
        'bbox_mAP_l': -1.0,
        'segm_mAP': 1.0,
        'segm_mAP_50': 1.0,
        'segm_mAP_75': 1.0,
        'segm_mAP_s': 1.0,
        'segm_mAP_m': -1.0,
        'segm_mAP_l': -1.0,
    }

    eval_results = coco_det_metric([dummy_pred], [dummy_gt])
    results = {k: round(v, 4) for k, v in eval_results.items()}
    assert results == target

    # test evaluate metric without loading ann_file
    # the gt instance area based on bounding box
    coco_det_metric = COCODetection(
        outfile_prefix=f'{tmp_dir.name}/test',
        metric=['bbox', 'segm'],
        dataset_meta=fake_dataset_metas,
        gt_mask_area=False)
    target = {
        'bbox_mAP': 1.0,
        'bbox_mAP_50': 1.0,
        'bbox_mAP_75': 1.0,
        'bbox_mAP_s': 1.0,
        'bbox_mAP_m': 1.0,
        'bbox_mAP_l': 1.0,
        'segm_mAP': 1.0,
        'segm_mAP_50': 1.0,
        'segm_mAP_75': 1.0,
        'segm_mAP_s': 1.0,
        'segm_mAP_m': 1.0,
        'segm_mAP_l': 1.0,
    }

    eval_results = coco_det_metric([dummy_pred], [dummy_gt])
    results = {k: round(v, 4) for k, v in eval_results.items()}
    assert results == target
    tmp_dir.cleanup()
