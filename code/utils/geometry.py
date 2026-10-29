"""Coordinate bookkeeping around the unchanged DepCSD networks."""
import numpy as np
from .numerics import require_finite


def inverse_augmentation(image, index):
    """Invert the exact order used by image_io.create_augmentations, for CHW arrays."""
    if not 0 <= index < 8:
        raise ValueError('augmentation index must be in [0, 7]')
    if index < 4:
        return np.rot90(image, -index, axes=(1, 2)).copy()
    rotated_back = np.rot90(image, -(index - 4), axes=(1, 2))
    return rotated_back[:, ::-1, :].copy()


def center_crop_input(image):
    """Match the original PIL center crop; retain no added border or padding."""
    require_finite(image, 'Input image')
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Input must be a C x H x W RGB NumPy array')
    if image.min() < 0 or image.max() > 1:
        raise ValueError('Input values must be in [0, 1]')
    h, w = image.shape[1:]
    hh, ww = h - h % 32, w - w % 32
    if min(hh, ww) < 64:
        raise ValueError('Center-cropped sides must be at least 64 for the original networks')
    top, left = (h - hh) // 2, (w - ww) // 2
    return image[:, top:top+hh, left:left+ww].copy()
