"""Geometry primitives shared across detection, calibration, and review."""

from pdiseg.core.boxes import (
    box_area,
    box_center,
    clamp_box,
    dedupe_boxes,
    iou,
    merge_nearby_boxes,
    non_max_suppression,
    pad_box,
)
from pdiseg.core.imaging import (
    BBox,
    FrameInspection,
    boxes_to_json,
    crop,
    inspection_from_json,
    render_overlay,
)

__all__ = [
    "BBox",
    "FrameInspection",
    "box_area",
    "box_center",
    "boxes_to_json",
    "clamp_box",
    "crop",
    "dedupe_boxes",
    "inspection_from_json",
    "iou",
    "merge_nearby_boxes",
    "non_max_suppression",
    "pad_box",
    "render_overlay",
]
