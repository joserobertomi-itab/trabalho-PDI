"""Axis-aligned bounding-box utilities (IoU, NMS, merge, clamp)."""

from __future__ import annotations

import numpy as np

from pdiseg.core.imaging import BBox

BBoxF = tuple[float, float, float, float]


def box_area(box: BBox) -> int:
    return box[2] * box[3]


def box_center(box: BBox) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def clamp_box(box: BBox, width: int, height: int) -> BBox:
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def pad_box(box: BBox, width: int, height: int, pad_frac: float) -> BBox:
    x, y, w, h = box
    px = int(w * pad_frac)
    py = int(h * pad_frac)
    return clamp_box((x - px, y - py, w + 2 * px, h + 2 * py), width, height)


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes: list[BBox], scores: list[float], iou_threshold: float) -> list[BBox]:
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    kept: list[BBox] = []
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(boxes[i])
        for j in order:
            if i == j or suppressed[j]:
                continue
            if iou(boxes[i], boxes[j]) >= iou_threshold:
                suppressed[j] = True
    return kept


def merge_nearby_boxes(
    boxes: list[BBox], distance_frac: float, width: int, height: int
) -> list[BBox]:
    if not boxes:
        return []
    diag = float(np.hypot(width, height))
    max_dist = distance_frac * diag
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_boxes: list[BBox] = []
        used = [False] * len(merged)
        for i, box_a in enumerate(merged):
            if used[i]:
                continue
            xa, ya, wa, ha = box_a
            x0, y0, x1, y1 = xa, ya, xa + wa, ya + ha
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                xb, yb, wb, hb = merged[j]
                cx_a, cy_a = box_center(box_a)
                cx_b, cy_b = box_center(merged[j])
                if np.hypot(cx_a - cx_b, cy_a - cy_b) <= max_dist:
                    x0 = min(x0, xb)
                    y0 = min(y0, yb)
                    x1 = max(x1, xb + wb)
                    y1 = max(y1, yb + hb)
                    used[j] = True
                    changed = True
            used[i] = True
            next_boxes.append(clamp_box((x0, y0, x1 - x0, y1 - y0), width, height))
        merged = next_boxes
    return merged


def dedupe_boxes(boxes: list[BBox], iou_threshold: float = 0.85) -> list[BBox]:
    scores = [float(box_area(b)) for b in boxes]
    return non_max_suppression(boxes, scores, iou_threshold)
