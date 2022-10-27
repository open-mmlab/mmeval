# Copyright (c) OpenMMLab. All rights reserved.
import csv
import logging
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Sequence

from mmeval.core.base_metric import BaseMetric
from ._vendor.ava_evaluation import object_detection_evaluation as det_eval
from ._vendor.ava_evaluation import standard_fields

logger = logging.getLogger(__name__)


class AVAMeanAP(BaseMetric):
    """AVA evaluation metric.

    AVA(Atomic Visual Action): https://research.google.com/ava.

    This metric computes mAP using the ava evaluation toolkit
    provided by the author.

    Args:
        ann_file (str): The annotation file path.
        label_file (str): The label file path.
        exclude_file (str, optional): The excluded timestamp file path.
            Defaults to None.
        num_classes (int): Number of classes. Defaults to 81.
        custom_classes (list(int), optional): A subset of class ids
            from origin dataset. Defaults to None.
        verbose (bool): Whether to print messages in the evaluation
            process. Defaults to True.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Examples:

        >>> from mmeval import AVAMeanAP
        >>> import numpy as np
        >>>
        >>> ann_file = 'tests/test_metrics/ava_detection_gt.csv'
        >>> label_file = 'tests/test_metrics/ava_action_list.txt'
        >>> num_classes = 4
        >>> ava_metric = AVAMeanAP(ann_file=ann_file, label_file=label_file,
        >>>                        num_classes=4)
        >>>
        >>> predictions = [
        >>> {
        >>>     'video_id': '3reY9zJKhqN',
        >>>     'timestamp': 1774,
        >>>     'outputs': [
        >>>         np.array([[0.362, 0.156, 0.969, 0.666, 0.106],
        >>>                   [0.442, 0.083, 0.721, 0.947, 0.162]]),
        >>>         np.array([[0.288, 0.365, 0.766, 0.551, 0.706],
        >>>                   [0.178, 0.296, 0.707, 0.995, 0.223]]),
        >>>         np.array([[0.417, 0.167, 0.843, 0.939, 0.015],
        >>>                   [0.35, 0.421, 0.57, 0.689, 0.427]]) ]
        >>> },
        >>> {
        >>>     'video_id': 'HmR8SmNIoxu',
        >>>     'timestamp': 1384,
        >>>     'outputs': [
        >>>         np.array([[0.256, 0.338, 0.726, 0.799, 0.563],
        >>>                   [0.071, 0.256, 0.64, 0.75, 0.297]]),
        >>>         np.array([[0.326, 0.036, 0.513, 0.991, 0.405],
        >>>                   [0.351, 0.035, 0.729, 0.936, 0.945]]),
        >>>         np.array([[0.051, 0.005, 0.975, 0.942, 0.424],
        >>>                   [0.347, 0.05, 0.97, 0.944, 0.396]])]
        >>> },
        >>> {
        >>>     'video_id': '5HNXoce1raG',
        >>>     'timestamp': 1097,
        >>>     'outputs': [
        >>>         np.array([[0.39, 0.087, 0.833, 0.616, 0.447],
        >>>                   [0.461, 0.212, 0.627, 0.527, 0.036]]),
        >>>         np.array([[0.022, 0.394, 0.93, 0.527, 0.109],
        >>>                   [0.208, 0.462, 0.874, 0.948, 0.954]]),
        >>>         np.array([[0.206, 0.456, 0.564, 0.725, 0.685],
        >>>                   [0.106, 0.445, 0.782, 0.673, 0.367]])]}
        >>> ]
        >>> ava_metric(predictions)
        {'mAP@0.5IOU': 0.027777778}
    """

    def __init__(self,
                 ann_file: str,
                 label_file: str,
                 exclude_file: Optional[str] = None,
                 num_classes: int = 81,
                 custom_classes: Optional[List[int]] = None,
                 verbose: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.ann_file = ann_file
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.custom_classes = custom_classes
        self.verbose = verbose
        if custom_classes is not None:
            self.custom_classes = [0] + custom_classes

    def add(self, predictions: Sequence[dict]) -> None:  # type: ignore
        """Add detection results to the results list.

        Args:
            predictions (Sequence[dict]): A list of prediction dict which
                contains the following keys:

                - `video_id`: The id of the video, e.g., `3reY9zJKhqN`.
                - `timestamp`: The timestamp of the video e.g., `1774`.
                - `outputs`: A list bbox results of each class with the format
                  of [x1, y1, x2, y2, score].
        """
        for prediction in predictions:
            self._results.append(prediction)

    def results2csv(self, results: List[dict], out_file: str) -> None:
        """Dump the results to a csv file.

        Args:
            results (list[dict]): A list of detection results.
            out_file (str): The output csv file path.
        """
        csv_results = []
        for res in results:
            video_id, timestamp = res['video_id'], res['timestamp']
            outputs = res['outputs']
            for label in range(len(outputs)):
                for bbox in outputs[label]:
                    bbox_ = [f'{i:.3f}' for i in bbox.tolist()]
                    if self.custom_classes is not None:
                        actual_label = self.custom_classes[label + 1]
                    else:
                        actual_label = label + 1

                    row_result = [video_id, str(timestamp)]
                    row_result += bbox_[:4]
                    row_result += [str(actual_label)]
                    row_result += bbox_[4:]
                    csv_results.append(row_result)

        with open(out_file, 'w') as f:
            for csv_result in csv_results:
                f.write(','.join(csv_result))
                f.write('\n')

    def read_label(self, label_file: str) -> tuple:
        """Reads a label mapping file.

        Args:
            label_file (str): The path of label file.

        Returns:
            tuple (labelmap, class_ids):

            - labelmap (list): The label map in the form used by the
              object_detection_evaluation module - a list of
              {"id": integer, "name": classname } dicts.
            - class_ids (set): A set containing all of the valid
              class id integers.
        """
        labelmap = []
        class_ids = set()
        name = ''
        with open(label_file) as f:
            for line in f:
                if line.startswith('  name:'):
                    name = line.split('"')[1]
                elif line.startswith('  id:') or line.startswith(
                        '  label_id:'):
                    class_id = int(line.strip().split(' ')[-1])
                    labelmap.append({'id': class_id, 'name': name})
                    class_ids.add(class_id)
        return labelmap, class_ids

    def read_csv(self,
                 csv_file: str,
                 class_whitelist: Optional[set] = None) -> tuple:
        """Loads boxes and class labels from a CSV file in the AVA format.

        CSV file format described at
        https://research.google.com/ava/download.html.

        Args:
            csv_file (str): A csv file path.
            class_whitelist (set, optional): If provided, boxes corresponding
                to (integer) class labels not in this set are skipped.

        Returns:
            tuple (boxes, labels, scores):

            - boxes (dict): A dictionary mapping each unique image key (string)
              to a list of boxes, given as coordinates [y1, x1, y2, x2].
            - labels (dict): A dictionary mapping each unique image key
              (string) to a list of integer class labels, matching the
              corresponding box in `boxes`.
            - scores (dict): A dictionary mapping each unique image
              key (string) to a list of score values labels, matching
              the corresponding label in `labels`. If scores are not
              provided in the csv, then they will default to 1.0.
        """
        entries = defaultdict(list)
        boxes = defaultdict(list)
        labels = defaultdict(list)
        scores = defaultdict(list)
        reader = csv.reader(open(csv_file))
        for row in reader:
            assert len(row) in [7, 8], f'Wrong number of columns: {row}'
            image_key = f'{row[0]}, {int(row[1]): 04d}'
            x1, y1, x2, y2 = (float(n) for n in row[2:6])
            action_id = int(row[6])
            if class_whitelist and action_id not in class_whitelist:
                continue

            score = 1.0
            if len(row) == 8:
                score = float(row[7])

            entries[image_key].append((score, action_id, y1, x1, y2, x2))

        for image_key in entries:
            # Evaluation API assumes boxes with descending scores
            entry = sorted(entries[image_key], key=lambda tup: -tup[0])
            boxes[image_key] = [x[2:] for x in entry]
            labels[image_key] = [x[1] for x in entry]
            scores[image_key] = [x[0] for x in entry]

        logger.info('read file ' + csv_file)
        return boxes, labels, scores

    def read_exclusions(self, exclude_file: str) -> set:
        """Reads a CSV file of excluded timestamps.

        Args:
            exclude_file (str): The path of exclude file.

        Returns:
            excluded (set): A set of strings containing excluded image keys,
            e.g. "aaaaaaaaaaa,0904" or an empty set if exclusions file is None.
        """
        excluded = set()
        reader = csv.reader(open(exclude_file))  # type: ignore
        for row in reader:
            assert len(row) == 2, f'Expected only 2 columns, got: {row}'
            excluded.add(f'{row[0]}, {int(row[1]): 04d}')
        return excluded

    def ava_eval(self, result_file: str) -> dict:
        """Perform ava evaluation.

        Args:
            result_file (str): The dumped results file path.

        Returns:
            dict: The evaluation results.
        """
        categories, class_whitelist = self.read_label(self.label_file)
        if self.custom_classes is not None:
            custom_classes = self.custom_classes[1:]
            assert set(custom_classes).issubset(set(class_whitelist))
            class_whitelist = custom_classes
            categories = [
                cat for cat in categories if cat['id'] in custom_classes
            ]

        # loading gt, do not need gt score
        gt_boxes, gt_labels, _ = self.read_csv(
            self.ann_file, class_whitelist=class_whitelist)
        if self.verbose:
            logger.info('Reading detection results')

        if self.exclude_file is not None:
            excluded_keys = self.read_exclusions(self.exclude_file)
        else:
            excluded_keys = set()

        boxes, labels, scores = self.read_csv(result_file, class_whitelist)
        if self.verbose:
            logger.info('Reading detection results')

        # Evaluation for mAP
        pascal_evaluator = det_eval.PascalDetectionEvaluator(categories)

        for image_key in gt_boxes:
            if image_key in excluded_keys:
                if self.verbose:
                    logging.info(
                        'Found excluded timestamp in detections: %s.'
                        'It will be ignored.', image_key)
                continue
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(gt_boxes[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                    np.array(gt_labels[image_key], dtype=int)
                })
        if self.verbose:
            logger.info('Convert groundtruth')

        for image_key in boxes:
            if self.verbose and image_key in excluded_keys:
                logging.info(
                    'Found excluded timestamp in detections: %s.'
                    'It will be ignored.', image_key)
                continue
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores[image_key], dtype=float)
                })
        if self.verbose:
            logger.info('convert detections')

        metrics = pascal_evaluator.evaluate()
        if self.verbose:
            logger.info('run_evaluator')
        for display_name in metrics:
            logger.info(f'{display_name}=\t{metrics[display_name]}')
        return {
            display_name: metrics[display_name]
            for display_name in metrics if 'ByCategory' not in display_name
        }

    def compute_metric(self, results: list) -> dict:
        """Compute the AVA MeanAP.

        Args:
            results (list): A list of detection results.

        Returns:
            dict: The computed ava metric.
        """
        time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = f'AVA_{time_now}_result.csv'
        self.results2csv(results, temp_file)
        eval_results = self.ava_eval(temp_file)
        os.remove(temp_file)
        return eval_results
