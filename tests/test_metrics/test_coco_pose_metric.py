# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import os.path as osp
import pytest
import tempfile
from json import dump
from unittest import TestCase

from mmeval.fileio import load
from mmeval.metrics import COCOKeyPointDetection

try:
    from xtcocotools.coco import COCO
    HAS_XTCOCOTOOLS = True
except ImportError:
    HAS_XTCOCOTOOLS = False


@pytest.mark.skipif(
    HAS_XTCOCOTOOLS is False, reason='xtcocotools is not available!')
class TestCOCOPoseMetric(TestCase):

    def setUp(self):
        """Setup some variables which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ann_file = 'tests/test_metrics/data/coco_pose_sample.json'
        self.coco = COCO(self.ann_file)
        coco_sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]).astype(np.float32)

        self.coco_dataset_meta = {
            'CLASSES': self.coco.loadCats(self.coco.getCatIds()),
            'num_keypoints': 17,
            'sigmas': coco_sigmas,
        }

        self.predictions, self.groundtruths = self._convert_ann_to_pred_and_gt(
            self.ann_file)
        assert len(self.predictions) == len(self.groundtruths) == 14
        self.target = {
            'AP': 1.0,
            'AP .5': 1.0,
            'AP .75': 1.0,
            'AP (M)': 1.0,
            'AP (L)': 1.0,
            'AR': 1.0,
            'AR .5': 1.0,
            'AR .75': 1.0,
            'AR (M)': 1.0,
            'AR (L)': 1.0,
        }

        self.ann_file_crowdpose = 'tests/test_metrics/data/' \
            'crowdpose_sample.json'
        self.coco_crowdpose = COCO(self.ann_file_crowdpose)
        sigmas_crowdpose = np.array([
            0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087,
            0.087, 0.089, 0.089, 0.079, 0.079
        ]).astype(np.float32)
        self.dataset_meta_crowdpose = {
            'CLASSES':
            self.coco_crowdpose.loadCats(self.coco_crowdpose.getCatIds()),
            'num_keypoints':
            14,
            'sigmas':
            sigmas_crowdpose,
        }

        self.preds_crowdpose, self.gts_crowdpose = \
            self._convert_ann_to_pred_and_gt(self.ann_file_crowdpose)
        assert len(self.preds_crowdpose) == len(self.gts_crowdpose) == 5
        self.target_crowdpose = {
            'AP': 1.0,
            'AP .5': 1.0,
            'AP .75': 1.0,
            'AR': 1.0,
            'AR .5': 1.0,
            'AR .75': 1.0,
            'AP(E)': -1.0,
            'AP(M)': 1.0,
            'AP(H)': -1.0,
        }

        self.ann_file_ap10k = 'tests/test_metrics/data/ap10k_sample.json'
        self.coco_ap10k = COCO(self.ann_file_ap10k)
        sigmas_ap10k = np.array([
            0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079,
            0.072, 0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
        ]).astype(np.float32)
        self.dataset_meta_ap10k = {
            'CLASSES': self.coco_ap10k.loadCats(self.coco_ap10k.getCatIds()),
            'num_keypoints': 17,
            'sigmas': sigmas_ap10k,
        }

        self.preds_ap10k, self.gts_ap10k = \
            self._convert_ann_to_pred_and_gt(self.ann_file_ap10k)
        assert len(self.preds_ap10k) == len(self.gts_ap10k) == 2
        self.target_ap10k = {
            'AP': 1.0,
            'AP .5': 1.0,
            'AP .75': 1.0,
            'AP (M)': -1.0,
            'AP (L)': 1.0,
            'AR': 1.0,
            'AR .5': 1.0,
            'AR .75': 1.0,
            'AR (M)': -1.0,
            'AR (L)': 1.0,
        }

    def _convert_ann_to_pred_and_gt(self, ann_file):
        """Convert annotations to topdown-style batch data."""
        predictions = []
        groundtruths = []

        db = load(ann_file)
        imgid2info = dict()

        for img in db['images']:
            imgid2info[img['id']] = img

        for ann in db['annotations']:
            bboxes = np.array(ann['bbox'], dtype=np.float32).reshape(-1, 4)
            keypoints = np.array(ann['keypoints']).reshape((1, -1, 3))

            prediction = {
                'id': ann['id'],
                'img_id': ann['image_id'],
                'category_id': ann.get('category_id', 1),
                'bboxes': bboxes,
                'keypoints': keypoints[..., :2],
                'keypoint_scores': keypoints[..., -1],
                'bbox_scores': np.ones((1, ), dtype=np.float32),
            }

            groundtruth = {
                'img_id': ann['image_id'],
                # dummy image_shape for testing
                'width': 640,
                'height': 480,
                'num_keypoints': ann['num_keypoints'],
                'raw_ann_info': [copy.deepcopy(ann)],
            }
            if 'area' in ann:
                groundtruth['area'] = ann['area']

            # add crowdIndex to gt if it is present in the image_info
            if 'crowdIndex' in imgid2info[ann['image_id']]:
                groundtruth['crowd_index'] = imgid2info[
                    ann['image_id']]['crowdIndex']

            predictions.append(prediction)
            groundtruths.append(groundtruth)

        return predictions, groundtruths

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        """test metric init method."""
        # test score_mode option
        with self.assertRaisesRegex(ValueError,
                                    '`score_mode` should be one of'):
            _ = COCOKeyPointDetection(
                ann_file=self.ann_file, score_mode='keypoint')

        # test nms_mode option
        with self.assertRaisesRegex(ValueError, '`nms_mode` should be one of'):
            _ = COCOKeyPointDetection(
                ann_file=self.ann_file, nms_mode='invalid')

        # test format_only option
        with self.assertRaisesRegex(
                AssertionError,
                '`outfile_prefix` can not be None when `format_only` is True'):
            _ = COCOKeyPointDetection(
                ann_file=self.ann_file, format_only=True, outfile_prefix=None)

    def test_other_methods(self):
        """test other useful methods."""
        # test `_sort_and_unique_bboxes` method
        coco_pose_metric = COCOKeyPointDetection(
            ann_file=self.ann_file,
            score_mode='bbox',
            nms_mode='none',
            dataset_meta=self.coco_dataset_meta)
        # an extra sample
        predictions = self.predictions + [self.predictions[0]]
        groundtruths = self.groundtruths + [self.groundtruths[0]]

        eval_results = coco_pose_metric(predictions, groundtruths)
        self.assertDictEqual(eval_results, self.target)

    def test_format_only(self):
        """test `format_only` option."""
        coco_pose_metric = COCOKeyPointDetection(
            ann_file=self.ann_file,
            format_only=True,
            outfile_prefix=f'{self.tmp_dir.name}/test',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms',
            dataset_meta=self.coco_dataset_meta)
        # process one sample
        eval_results = coco_pose_metric(self.predictions, self.groundtruths)
        self.assertDictEqual(eval_results, {})
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.keypoints.json')))

        # test when gt annotations are absent
        db_ = load(self.ann_file)
        del db_['annotations']
        tmp_ann_file = osp.join(self.tmp_dir.name, 'temp_ann.json')
        with open(tmp_ann_file, 'w') as f:
            dump(db_, f, sort_keys=True, indent=4)
        with self.assertRaisesRegex(
                AssertionError,
                'Ground truth annotations are required for evaluation'):
            _ = COCOKeyPointDetection(ann_file=tmp_ann_file, format_only=False)

    def test_evaluate(self):
        """test COCO metric evaluation."""
        # case 1: score_mode='bbox', nms_mode='none'
        coco_pose_metric = COCOKeyPointDetection(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test1',
            score_mode='bbox',
            nms_mode='none',
            dataset_meta=self.coco_dataset_meta)

        eval_results = coco_pose_metric(self.predictions, self.groundtruths)
        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test1.keypoints.json')))

        # case 2: score_mode='bbox_keypoint', nms_mode='oks_nms'
        coco_pose_metric = COCOKeyPointDetection(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test2',
            score_mode='bbox_keypoint',
            nms_mode='oks_nms',
            dataset_meta=self.coco_dataset_meta)
        coco_pose_metric.dataset_meta = self.coco_dataset_meta

        eval_results = coco_pose_metric(self.predictions, self.groundtruths)
        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test2.keypoints.json')))

        # case 3: score_mode='bbox_rle', nms_mode='soft_oks_nms'
        coco_pose_metric = COCOKeyPointDetection(
            ann_file=self.ann_file,
            outfile_prefix=f'{self.tmp_dir.name}/test3',
            score_mode='bbox_rle',
            nms_mode='soft_oks_nms',
            dataset_meta=self.coco_dataset_meta)

        eval_results = coco_pose_metric(self.predictions, self.groundtruths)
        self.assertDictEqual(eval_results, self.target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test3.keypoints.json')))

        # case 4: test without providing ann_file
        coco_pose_metric = COCOKeyPointDetection(
            outfile_prefix=f'{self.tmp_dir.name}/test4',
            dataset_meta=self.coco_dataset_meta)
        eval_results = coco_pose_metric(self.predictions, self.groundtruths)

        self.assertDictEqual(eval_results, self.target)
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test4.gt.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test4.keypoints.json')))

        # case 5: test Crowdpose dataset
        metric_crowdpose = COCOKeyPointDetection(
            ann_file=self.ann_file_crowdpose,
            outfile_prefix=f'{self.tmp_dir.name}/test5',
            use_area=False,
            iou_type='keypoints_crowd',
            dataset_meta=self.dataset_meta_crowdpose)
        eval_results = metric_crowdpose(self.preds_crowdpose,
                                        self.gts_crowdpose)
        self.assertDictEqual(eval_results, self.target_crowdpose)
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test5.keypoints.json')))

        # case 5: test Crowdpose dataset + without ann_file
        metric_crowdpose = COCOKeyPointDetection(
            outfile_prefix=f'{self.tmp_dir.name}/test6',
            use_area=False,
            iou_type='keypoints_crowd',
            dataset_meta=self.dataset_meta_crowdpose)
        eval_results = metric_crowdpose(self.preds_crowdpose,
                                        self.gts_crowdpose)
        self.assertDictEqual(eval_results, self.target_crowdpose)
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test6.gt.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test6.keypoints.json')))

        # case 7: test AP10k dataset
        metric_ap10k = COCOKeyPointDetection(
            ann_file=self.ann_file_ap10k,
            outfile_prefix=f'{self.tmp_dir.name}/test7',
            use_area=False,
            dataset_meta=self.dataset_meta_ap10k)
        eval_results = metric_ap10k(self.preds_ap10k, self.gts_ap10k)
        for key in self.target_ap10k:
            self.assertTrue(
                abs(eval_results[key] - self.target_ap10k[key]) < 1e-7)
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test7.keypoints.json')))

        # case 8: test AP10k dataset + without ann_file
        metric_ap10k = COCOKeyPointDetection(
            outfile_prefix=f'{self.tmp_dir.name}/test8',
            dataset_meta=self.dataset_meta_ap10k)
        eval_results = metric_ap10k(self.preds_ap10k, self.gts_ap10k)
        for key in self.target_ap10k:
            self.assertTrue(
                abs(eval_results[key] - self.target_ap10k[key]) < 1e-7)
        # test whether convert the annotation to COCO format
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test8.gt.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test8.keypoints.json')))
