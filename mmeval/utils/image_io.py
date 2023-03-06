# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from mmeval.utils.misc import try_import
from mmeval.utils.path import is_filepath

if TYPE_CHECKING:
    import cv2
else:
    cv2 = try_import('cv2')


def imwrite(img: np.ndarray, file_path: str) -> bool:
    """Write image to local path.

    Args:
        img (np.ndarray): Image array to be written.
        file_path (str): Image file path.

    Returns:
        bool: Successful or not.
    """
    if cv2 is None:
        raise ImportError('To use `imwrite` function, '
                          'please install opencv-python first.')

    assert is_filepath(file_path)

    # auto make dir
    dir_name = osp.expanduser(osp.dirname(file_path))
    os.makedirs(dir_name, exist_ok=True)

    img_ext = osp.splitext(file_path)[-1]
    file_path = str(file_path)
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    flag, img_buff = cv2.imencode(img_ext, img)
    with open(file_path, 'wb') as f:
        f.write(img_buff)
    return flag


def imread(file_path: Union[str, Path],
           flag: str = 'color',
           channel_order: str = 'rgb',
           backend_args: Optional[dict] = None) -> np.ndarray:
    """Read an image from path.

    Args:
        file_path (str or Path): Either a str or pathlib.Path.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            Defaults to 'color'.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
            Defaults to 'rgb'.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.

    Returns:
        ndarray: Loaded image array.
    """
    if cv2 is None:
        raise ImportError('To use `imread` function, '
                          'please install opencv-python first.')

    imread_flags = {
        'color':
        cv2.IMREAD_COLOR,
        'grayscale':
        cv2.IMREAD_GRAYSCALE,
        'unchanged':
        cv2.IMREAD_UNCHANGED,
        'color_ignore_orientation':
        cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR,
        'grayscale_ignore_orientation':
        cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_GRAYSCALE
    }

    from mmeval.fileio import get

    assert is_filepath(file_path)

    img_bytes = get(file_path, backend_args=backend_args)
    img_np = np.frombuffer(img_bytes, np.uint8)
    flag = imread_flags[flag] if isinstance(flag, str) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == cv2.IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img
