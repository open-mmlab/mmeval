# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np


def qbox_to_rbox(boxes: np.ndarray) -> np.ndarray:
    """Convert quadrilateral boxes to rotated boxes.

    Args:
        boxes (np.ndarray): Quadrilateral box tensor with shape of (N, 8).

    Returns:
        np.ndarray: Rotated box array with shape of (N, 5).
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 8, \
        'The boxes shape should be (N, 8)'
    original_shape = boxes.shape[:-1]
    points = boxes.reshape(-1, 4, 2).astype(np.int64)
    rboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    rboxes = np.array(rboxes)
    return rboxes.reshape(*original_shape, 5)


def le90_to_oc(bboxes: np.ndarray):
    """convert bboxes with le90 angle version to OpenCV angle version.

    Args:
        bboxes (np.ndarray): The shape of bboxes should be (N, 5),
            the format is 'xywha'. And the angle of the bboxes
            should be in range of [-pi/2, pi/2).

    Returns:
        np.ndarray: An numpy.ndarray with the same shape of input.
    """
    assert bboxes.ndim == 2 and bboxes.shape[-1] == 5, \
        'The boxes shape should be (N, 5)'

    # a mask to indicate if input angles belong to [-pi/2, 0)
    mask = bboxes[:, 4] <= 0.0
    # convert the boxes whose angle is in range of [-pi/2, 0)
    ret_bboxes = bboxes.copy()
    ret_bboxes[:, 4] += np.pi / 2 * np.ones(bboxes.shape[0]) * mask
    # swap w and h
    temp = ret_bboxes[mask]
    temp[:, [2, 3]] = temp[:, [3, 2]]
    ret_bboxes[mask] = temp

    # a special case of angle = -pi/2
    mask = (np.round(bboxes[:, 4], 8) == -np.round(np.pi / 2, 8))
    ret_bboxes[:, 4] += np.pi / 2 * np.ones(bboxes.shape[0]) * mask
    temp = ret_bboxes[mask]
    temp[:, [2, 3]] = temp[:, [3, 2]]
    ret_bboxes[mask] = temp
    return ret_bboxes


def calculate_bboxes_area_rotated(bboxes: np.ndarray) -> np.ndarray:
    """Calculate area of rotated bounding boxes.

    Args:
        bboxes (np.ndarray): The bboxes with shape (n, 5) or (5, )
            in 'xywha' format.

    Returns:
        np.ndarray: The area of bboxes.
    """
    bboxes_w = bboxes[..., 2]
    bboxes_h = bboxes[..., 3]
    areas = bboxes_w * bboxes_h
    return areas


def calculate_overlaps_rotated(bboxes1: np.ndarray,
                               bboxes2: np.ndarray,
                               clockwise: bool = True) -> np.ndarray:
    """Calculate the overlap between each rotated bbox of bboxes1 and bboxes2.

    Modified from: https://github.com/yangxue0827/RotationDetection/blob/
    45c708a34348e38040786ba85b288bc147786452/alpharotate/libs/
    utils/iou_rotate.py#L39

    Args:
        bboxes1 (np.ndarray): The bboxes with shape (n, 5) in 'xywha' format.
        bboxes2 (np.ndarray): The bboxes with shape (k, 5) in 'xywha' format.
        clockwise (bool, optional): flag indicating whether the positive
            angular orientation is clockwise. Defaults to True.

    Returns:
        np.ndarray: IoUs with shape (n, k).
    """
    assert bboxes1.ndim == 2 and bboxes1.shape[-1] == 5, \
        'The shape of bboxes1 should be (N, 5)'
    assert bboxes2.ndim == 2 and bboxes2.shape[-1] == 5, \
        'The shape of bboxes2 should be (K, 5)'

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)

    if rows * cols == 0:
        return ious

    if not clockwise:
        flip_mat = np.ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat

    # convert the bboxes with le90 angle version to OpenCV angle version
    bboxes1 = le90_to_oc(bboxes1)
    bboxes2 = le90_to_oc(bboxes2)
    bboxes1[:, 4] = bboxes1[:, 4] * 180.0 / np.pi
    bboxes2[:, 4] = bboxes2[:, 4] * 180.0 / np.pi

    area1 = bboxes1[:, 2] * bboxes1[:, 3]
    area2 = bboxes2[:, 2] * bboxes2[:, 3]
    for i, box1 in enumerate(bboxes1):
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(bboxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                inter = int_area * 1.0 / (
                    area1[i] + area2[j] - int_area + 1e-5)
                ious[i][j] = inter
            else:
                ious[i][j] = 0.0
    return ious
