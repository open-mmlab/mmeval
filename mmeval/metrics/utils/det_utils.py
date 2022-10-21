# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Optional


def calculate_average_precision(recalls: np.ndarray,
                                precisions: np.ndarray,
                                mode: str = 'area') -> float:
    """Calculate average precision in the detection task.

    Args:
        recalls (ndarray): The recalls with shape (num_dets, ).
        precisions (ndarray): The precisions with shape (num_dets, ).
        mode (str): The average mode, should be 'area' or '11points'.
            'area' means calculating the area under precision-recall curve.
            '11points' means calculating the average precision of recalls at
            [0, 0.1, ..., 1.0]. Defaults to 'area'.

    Returns:
        float: Calculated average precision.
    """
    assert mode in ['area', '11points']
    if mode == 'area':
        mrec = np.hstack((0, recalls, 1))
        mpre = np.hstack((0, precisions, 0))
        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        ind = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[ind + 1] - mrec[ind]) * mpre[ind + 1])
    else:
        ap = 0.0
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[recalls >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11.0
    return ap


def calculate_bboxes_area(bboxes: np.ndarray,
                          use_legacy_coordinate: bool = False) -> np.ndarray:
    """Calculate area of bounding boxes.

    Args:
        bboxes (numpy.ndarray): The bboxes with shape (n, 4) or (4, ) in 'xyxy'
            format.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Defaults to False.

     Returns:
        numpy.ndarray: The area of bboxes.
    """
    if use_legacy_coordinate:
        extra_length = 1.
    else:
        extra_length = 0.
    bboxes_w = (bboxes[..., 2] - bboxes[..., 0] + extra_length)
    bboxes_h = (bboxes[..., 3] - bboxes[..., 1] + extra_length)
    areas = bboxes_w * bboxes_h
    return areas


def filter_by_bboxes_area(bboxes: np.ndarray,
                          min_area: Optional[float],
                          max_area: Optional[float],
                          use_legacy_coordinate=False) -> np.ndarray:
    """Filter the bboxes with an area range.

    Args:
        bboxes (numpy.ndarray): The bboxes with shape (n, 4) in 'xyxy' format.
        min_area (Optional[float]): The minimum area. If None, does not filter
            the minimum area.
        max_area (Optional[float]): The maximum area. If None, does not filter
            the maximum area.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        numpy.ndarray: A mask of ``bboxes`` identify which bbox are filtered.
    """
    bboxes_area = calculate_bboxes_area(bboxes, use_legacy_coordinate)
    area_mask = np.ones_like(bboxes_area, dtype=bool)
    if min_area is not None:
        area_mask &= (bboxes_area >= min_area)
    if max_area is not None:
        area_mask &= (bboxes_area < max_area)
    return area_mask


def calculate_overlaps(bboxes1: np.ndarray,
                       bboxes2: np.ndarray,
                       mode: str = 'iou',
                       eps: float = 1e-6,
                       use_legacy_coordinate: bool = False) -> np.ndarray:
    """Calculate the overlap between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (numpy.ndarray): The bboxes with shape (n, 4) in 'xyxy' format.
        bboxes2 (numpy.ndarray): The bboxes with shape (k, 4) in 'xyxy' format.
        mode (str): 'iou' (intersection over union) or 'iof'
            (intersection over foreground). Defaults to 'iou'.
        eps (float): The epsilon value. Defaults to 1e-6.
        use_legacy_coordinate (bool): Whether to use coordinate
            system in mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        numpy.ndarray: IoUs or IoFs with shape (n, k).
    """
    assert mode in ['iou', 'iof']
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    overlaps = np.zeros((rows, cols), dtype=np.float32)

    if rows * cols == 0:
        return overlaps

    if bboxes1.shape[0] > bboxes2.shape[0]:
        # Swap bboxes for faster calculation.
        bboxes1, bboxes2 = bboxes2, bboxes1
        overlaps = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    else:
        exchange = False

    # Calculate the bboxes area.
    area1 = calculate_bboxes_area(bboxes1, use_legacy_coordinate)
    area2 = calculate_bboxes_area(bboxes2, use_legacy_coordinate)

    if use_legacy_coordinate:
        extra_length = 1.
    else:
        extra_length = 0.

    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap_w = np.maximum(x_end - x_start + extra_length, 0)
        overlap_h = np.maximum(y_end - y_start + extra_length, 0)
        overlap = overlap_w * overlap_h

        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2

        union = np.maximum(union, eps)
        overlaps[i, :] = overlap / union
    return overlaps if not exchange else overlaps.T