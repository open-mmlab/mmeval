# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

from mmeval.core.base_metric import BaseMetric


class ProposalRecall(BaseMetric):
    """Pascal VOC evaluation metric.

    Args:
        iou_thrs (float ｜ List[float]): IoU thresholds. Defaults to 0.5.
        proposal_nums
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        use_legacy_coordinate (bool): Whether to use coordinate
            system in mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.
        nproc (int): Processes used for computing TP and FP. If nproc
            is less than or equal to 1, multiprocessing will not be used.
            Defaults to 4.
        drop_class_ap (bool): Whether to drop the class without ground truth
            when calculating the average precision for each class.
        classwise_result (bool): Whether to return the computed
            results of each class.
            Defaults to False.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[Tuple]] = None,
                 num_classes: Optional[int] = None,
                 eval_mode: str = 'area',
                 use_legacy_coordinate: bool = False,
                 nproc: int = 4,
                 drop_class_ap: bool = True,
                 classwise_result: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(iou_thrs, float):
            iou_thrs = [iou_thrs]
        self.iou_thrs = iou_thrs

        if scale_ranges is None:
            scale_ranges = [(None, None)]
        elif (None, None) not in scale_ranges:
            # We allawys compute the mAP across all scale.
            scale_ranges.append((None, None))
        self.scale_ranges = scale_ranges

        area_ranges = []
        for min_scale, max_scale in self.scale_ranges:
            min_area = min_scale if min_scale is None else min_scale**2
            max_area = max_scale if max_scale is None else max_scale**2
            area_ranges.append((min_area, max_area))
        self._area_ranges = area_ranges

        self._num_classes = num_classes

        assert eval_mode in ['area', '11points'], \
            'Unrecognized mode, only "area" and "11points" are supported'
        self.eval_mode = eval_mode

        self.nproc = nproc
        self.use_legacy_coordinate = use_legacy_coordinate
        self.drop_class_ap = drop_class_ap
        self.classwise_result = classwise_result

        self.num_iou = len(self.iou_thrs)
        self.num_scale = len(self.scale_ranges)
