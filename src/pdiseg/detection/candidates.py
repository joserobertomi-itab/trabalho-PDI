"""Connected components to bounding boxes for label candidates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import find_objects, label

from pdiseg.core.boxes import box_area, dedupe_boxes
from pdiseg.core.imaging import BBox

from .config import DetectionConfig
from .masks import build_candidate_masks, text_density_mask


def _area_ok(
    box: BBox, frame_area: int, config: DetectionConfig, *, pixel_area: int | None = None
) -> bool:
    area = pixel_area if pixel_area is not None else box_area(box)
    frac = area / frame_area
    if frac < config.min_area_frac:
        return False
    bbox_area = box_area(box)
    if bbox_area > config.label_max_area:
        return False
    bbox_frac = bbox_area / frame_area
    return bbox_frac <= 0.45


def boxes_from_mask(
    mask: NDArray[np.bool_],
    min_area: int,
    config: DetectionConfig | None = None,
    frame_area: int | None = None,
) -> list[BBox]:
    labels, _ = label(mask)
    boxes: list[BBox] = []
    for index, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        area = int((labels[slc] == index).sum())
        if area < min_area:
            continue
        ys, xs = slc
        box = (xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)
        if (
            config is not None
            and frame_area is not None
            and not _area_ok(box, frame_area, config, pixel_area=area)
        ):
            continue
        boxes.append(box)
    return boxes


def find_candidate_boxes(
    work: NDArray[np.uint8],
    config: DetectionConfig,
    *,
    text_source: NDArray[np.uint8] | None = None,
) -> list[BBox]:
    height, width = work.shape
    frame_area = height * width
    min_area = max(400, int(frame_area * config.cluster_min_area_frac))
    masks = build_candidate_masks(work, config)
    text_src = text_source if text_source is not None else work
    boxes = boxes_from_mask(masks.combined, min_area, config, frame_area)
    boxes.extend(boxes_from_mask(masks.black_hat, min_area, config, frame_area))
    boxes.extend(boxes_from_mask(text_density_mask(text_src, config), min_area, config, frame_area))
    return dedupe_boxes(boxes)


def detect_clusters(image: NDArray[np.uint8], config: DetectionConfig | None = None) -> list[BBox]:
    cfg = config or DetectionConfig()
    height, width = image.shape[:2]
    frame_area = height * width
    min_area = max(800, int(frame_area * cfg.cluster_min_area_frac))
    mask = text_density_mask(image, cfg)
    return boxes_from_mask(mask, min_area, cfg, frame_area)
