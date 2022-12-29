# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import unittest

from mmeval.metrics import InstanceSeg


class TestInstanceSegMetric(unittest.TestCase):

    def _demo_mm_model_output(self):
        """Create a superset of inputs needed to run test or train batches."""

        n_points_list = [3300, 3000]
        gt_labels_list = [[0, 0, 0, 0, 0, 0, 14, 14, 2, 1],
                          [13, 13, 2, 1, 3, 3, 0, 0, 0]]

        predictions = []
        groundtruths = []

        for n_points, gt_labels in zip(n_points_list, gt_labels_list):
            gt_instance_mask = np.ones(n_points, dtype=int) * -1
            gt_semantic_mask = np.ones(n_points, dtype=int) * -1
            for i, gt_label in enumerate(gt_labels):
                begin = i * 300
                end = begin + 300
                gt_instance_mask[begin:end] = i
                gt_semantic_mask[begin:end] = gt_label

            ann_info_data = dict()
            ann_info_data['pts_instance_mask'] = torch.tensor(gt_instance_mask)
            ann_info_data['pts_semantic_mask'] = torch.tensor(gt_semantic_mask)

            results_dict = dict()
            pred_instance_mask = np.ones(n_points, dtype=int) * -1
            labels = []
            scores = []
            for i, gt_label in enumerate(gt_labels):
                begin = i * 300
                end = begin + 300
                pred_instance_mask[begin:end] = i
                labels.append(gt_label)
                scores.append(.99)

            results_dict['pts_instance_mask'] = torch.tensor(
                pred_instance_mask)
            results_dict['instance_labels'] = torch.tensor(labels)
            results_dict['instance_scores'] = torch.tensor(scores)

            predictions.append(results_dict)
            groundtruths.append(ann_info_data)

        return predictions, groundtruths

    def test_evaluate(self):
        predictions, groundtruths = self._demo_mm_model_output()
        predictions = [{k: v.clone().numpy()
                        for k, v in i.items()} for i in predictions]
        groundtruths = [{k: v.clone().numpy()
                         for k, v in i.items()} for i in groundtruths]
        seg_valid_class_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                               33, 34, 36, 39)
        class_labels = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                        'window', 'bookshelf', 'picture', 'counter', 'desk',
                        'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                        'sink', 'bathtub', 'garbagebin')
        dataset_meta = dict(
            seg_valid_class_ids=seg_valid_class_ids, classes=class_labels)
        instance_seg_metric = InstanceSeg(dataset_meta=dataset_meta)
        res = instance_seg_metric(predictions, groundtruths)
        self.assertIsInstance(res, dict)
        for label in [
                'cabinet', 'bed', 'chair', 'sofa', 'showercurtrain', 'toilet'
        ]:
            metrics = res['classes'][label]
            assert metrics['ap'] == 1.0
            assert metrics['ap50%'] == 1.0
            assert metrics['ap25%'] == 1.0

        predictions[1]['pts_instance_mask'][2240:2700] = -1
        predictions[0]['pts_instance_mask'][2700:3000] = 8
        predictions[0]['instance_labels'][9] = 2

        res = instance_seg_metric(predictions, groundtruths)

        assert abs(res['classes']['cabinet']['ap50%'] - 0.72916) < 0.01
        assert abs(res['classes']['cabinet']['ap25%'] - 0.88888) < 0.01
        assert abs(res['classes']['bed']['ap50%'] - 0.5) < 0.01
        assert abs(res['classes']['bed']['ap25%'] - 0.5) < 0.01
        assert abs(res['classes']['chair']['ap50%'] - 0.375) < 0.01
        assert abs(res['classes']['chair']['ap25%'] - 1.0) < 0.01
