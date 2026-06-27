"""Horizontal ROI bounds to ignore conveyor-side cardboard / box-edge text."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox

from .config import DetectionConfig


def horizontal_roi_bounds(frame_width: int, config: DetectionConfig) -> tuple[int, int]:
    """Return inclusive-exclusive x range ``[x0, x1)`` where detection is allowed."""
    if frame_width <= 0:
        return 0, 0
    if config.exclude_left_frac > 0 or config.exclude_right_frac > 0:
        x0 = int(frame_width * config.exclude_left_frac)
        x1 = frame_width - int(frame_width * config.exclude_right_frac)
    elif config.lateral_margin_frac > 0:
        margin = int(frame_width * config.lateral_margin_frac)
        x0, x1 = margin, frame_width - margin
    else:
        return 0, frame_width
    x0 = max(0, min(x0, frame_width - 1))
    x1 = max(x0 + 1, min(x1, frame_width))
    return x0, x1


def box_inside_horizontal_roi(box: BBox, x0: int, x1: int) -> bool:
    x, _y, w, _h = box
    return x >= x0 and x + w <= x1


def mask_horizontal_roi(mask: NDArray[np.bool_], x0: int, x1: int) -> NDArray[np.bool_]:
    """Zero columns outside the valid horizontal ROI."""
    if x0 <= 0 and x1 >= mask.shape[1]:
        return mask
    cleared = mask.copy()
    if x0 > 0:
        cleared[:, :x0] = False
    if x1 < mask.shape[1]:
        cleared[:, x1:] = False
    return cleared


def apply_horizontal_exclusion_to_image(
    image: NDArray[np.uint8], x0: int, x1: int
) -> NDArray[np.uint8]:
    """Fill excluded side columns with the median of the valid central band."""
    _height, width = image.shape[:2]
    if x0 <= 0 and x1 >= width:
        return image
    out = image.copy()
    valid = image[:, x0:x1]
    fill = int(np.median(valid)) if valid.size else int(np.median(image))
    if x0 > 0:
        out[:, :x0] = fill
    if x1 < width:
        out[:, x1:] = fill
    return out
