from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

BBox = tuple[int, int, int, int]

_REJECTED_COLOR = (220, 50, 50)
_KEPT_COLOR = (240, 200, 40)
_LABEL_COLOR = (40, 200, 80)


@dataclass
class FrameInspection:
    candidates: list[BBox]
    kept: list[BBox]
    labels: list[BBox]


def crop(image: NDArray[np.uint8], bbox: BBox) -> NDArray[np.uint8]:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w]


def _draw_box(rgb: NDArray[np.uint8], bbox: BBox, color: tuple[int, int, int]) -> None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return
    height, width = rgb.shape[:2]
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w - 1, width - 1)
    y1 = min(y + h - 1, height - 1)
    if x0 >= width or y0 >= height or x1 < x0 or y1 < y0:
        return
    rgb[y0, x0 : x1 + 1] = color
    rgb[y1, x0 : x1 + 1] = color
    rgb[y0 : y1 + 1, x0] = color
    rgb[y0 : y1 + 1, x1] = color


def render_overlay(image: NDArray[np.uint8], inspection: FrameInspection) -> NDArray[np.uint8]:

    from skimage.color import gray2rgb
    from skimage.util import img_as_ubyte

    rgb = cast(
        NDArray[np.uint8],
        gray2rgb(img_as_ubyte(image)),  # type: ignore[no-untyped-call]
    ).copy()
    kept_set = set(inspection.kept)
    for bbox in inspection.candidates:
        if bbox not in kept_set:
            _draw_box(rgb, bbox, _REJECTED_COLOR)
    for bbox in inspection.kept:
        _draw_box(rgb, bbox, _KEPT_COLOR)
    for bbox in inspection.labels:
        _draw_box(rgb, bbox, _LABEL_COLOR)
    return rgb


def boxes_to_json(boxes: list[BBox]) -> list[list[int]]:
    return [list(box) for box in boxes]


def inspection_from_json(data: dict[str, list[list[int]]]) -> FrameInspection:
    def _parse(key: str) -> list[BBox]:
        return [tuple(row) for row in data[key]]  # type: ignore[misc]

    return FrameInspection(
        candidates=_parse("candidates"),
        kept=_parse("kept"),
        labels=_parse("labels"),
    )
