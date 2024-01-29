# Copyright (c) OpenMMLab. All rights reserved.
from faster_coco_eval import COCO as _COCO
from faster_coco_eval import COCOeval_faster as _faster_COCOeval
from mmeval.metrics.utils.coco_wrapper import COCO

class FasterCOCO(COCO, _COCO):
    """This class is almost the same as official pycocotools package and faster-coco-eval package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.

    Args:
        annotation_file (str, optional): Path of annotation file.
            Defaults to None.
    """

# just for the ease of import
FasterCOCOeval = _faster_COCOeval
