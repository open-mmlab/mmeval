# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Optional, Union


def _convert_input_type_range(img: np.ndarray) -> np.ndarray:
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1]. It is
    mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (np.ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        np.ndarray: The converted image with type of np.float32 and range of
        [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(
        img: np.ndarray, dst_type: Union[np.uint8, np.float32]) -> np.ndarray:
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (np.ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        np.ndarray: The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def rgb2ycbcr(img: np.ndarray, y_only: bool = False) -> np.ndarray:
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (np.ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Defaults to False.

    Returns:
        np.ndarray: The converted YCbCr image. The output image has the same
        type and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                  [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img: np.ndarray, y_only: bool = False) -> np.ndarray:
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (np.ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Defaults to False.

    Returns:
        np.ndarray: The converted YCbCr image. The output image has the same
        type and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def reorder_image(img: np.array, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (np.ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Defaults to 'HWC'.

    Returns:
        np.ndarray: Reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def reorder_and_crop(img: np.ndarray,
                     crop_border: int = 0,
                     input_order: str = 'HWC',
                     convert_to: Optional[str] = None,
                     channel_order: str = 'rgb'):
    """Image transformations contains Crop, Reorder to 'HWC' and Convert to
    'Y'.

    Args:
        img (np.ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Defaults to 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Defaults to 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Defaults to None.
        channel_order (str): The channel order of image. Defaults to 'rgb'

    Returns:
        np.array: The transformation results.
    """

    img = reorder_image(img, input_order=input_order)
    img = img.astype(np.float32)

    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        if channel_order == 'rgb':
            img = rgb2ycbcr(img / 255., y_only=True) * 255.
        elif channel_order == 'bgr':
            img = bgr2ycbcr(img / 255., y_only=True) * 255.
        else:
            raise ValueError(
                'Only support `rgb2y` and `bgr2y`, but the channel_order '
                f'is {channel_order}')
        img = np.expand_dims(img, axis=2)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None.')

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, :]

    return img.astype(np.float64)
