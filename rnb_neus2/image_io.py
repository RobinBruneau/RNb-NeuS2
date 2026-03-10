"""Image I/O utilities for RNb-NeuS2.

Supports PNG (8/16-bit) and EXR (float32) formats.
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np


def load_image(path):
    """Load an image, auto-detecting format.

    - PNG 8-bit / 16-bit: normalized to [0, 1]
    - EXR float32: returned as-is (may contain values outside [0, 1])

    Returns:
        (H, W, C) float32 array in RGB(A) order.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("Cannot read image: {}".format(path))

    if image.dtype == np.float32:
        # EXR: already float, just fix channel order
        pass
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        raise ValueError("Unsupported dtype: {}".format(image.dtype))

    # BGR(A) -> RGB(A)
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image, path, bit_depth=16):
    """Save an image as PNG (8 or 16-bit).

    Args:
        image: (H, W, C) float32 array in RGB(A), values in [0, 1].
        path: Output path.
        bit_depth: 8 or 16.
    """
    image = np.nan_to_num(image, nan=0.0)
    image = np.clip(image, 0.0, 1.0)
    image = image * float(2 ** bit_depth - 1)

    if bit_depth == 8:
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint16)

    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_exr(image, path):
    """Save a float32 image as EXR.

    Args:
        image: (H, W, C) float32 array in RGB order.
        path: Output path (should end in .exr).
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # RGB -> BGR for cv2
    if len(image.shape) == 3 and image.shape[2] >= 3:
        image = image[:, :, ::-1].copy()

    cv2.imwrite(str(path), image,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])


def load_normal(path):
    """Load a normal map, auto-detecting format and value range.

    - EXR: values in [-1, 1], returned as-is
    - PNG: values in [0, 1] representing [-1, 1], remapped

    Returns:
        (H, W, 3) float32 array in [-1, 1].
    """
    ext = os.path.splitext(path)[1].lower()
    image = load_image(path)

    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    if ext == ".exr":
        # Already in [-1, 1]
        return image
    else:
        # PNG: [0, 1] -> [-1, 1]
        return image * 2.0 - 1.0


def save_normal_16bit(normal, path):
    """Save normal map as 16-bit PNG. Normals [-1,1] -> [0, 65535]."""
    normal_01 = 0.5 * (1.0 + normal)
    normal_16 = np.clip(normal_01 * 65535.0, 0, 65535).astype(np.uint16)
    # RGB -> BGR
    cv2.imwrite(str(path), normal_16[:, :, ::-1],
                [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_normal_exr(normal, path):
    """Save normal map as EXR with raw [-1, 1] values."""
    save_exr(normal.astype(np.float32), path)
