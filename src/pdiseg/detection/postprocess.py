"""NMS, merge, Otsu refine, fallback, and score filtering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import find_objects, label
from skimage.filters import threshold_otsu

from pdiseg.core.boxes import merge_nearby_boxes, non_max_suppression, pad_box
from pdiseg.core.imaging import BBox

from .candidates import boxes_from_mask
from .config import DetectionConfig
from .masks import build_candidate_masks
from .scoring import ScoredCandidate, score_candidates


def keep_label_clusters(
    candidates: list[BBox], config: DetectionConfig | None = None
) -> list[BBox]:
    cfg = config or DetectionConfig()
    kept: list[BBox] = []
    for x, y, w, h in candidates:
        area = w * h
        elongation = max(w, h) / min(w, h)
        if (
            cfg.label_min_area <= area <= cfg.label_max_area
            and elongation <= cfg.label_max_elongation
        ):
            kept.append((x, y, w, h))
    return kept


def refine_to_name_label(
    image: NDArray[np.uint8], cluster_bbox: BBox, config: DetectionConfig | None = None
) -> BBox:
    cfg = config or DetectionConfig()
    x, y, w, h = cluster_bbox
    region = image[y : y + h, x : x + w]
    if region.size == 0 or region.max() == region.min():
        return cluster_bbox

    dark = region <= threshold_otsu(region)  # type: ignore[no-untyped-call]
    labels, count = label(dark)
    if count == 0:
        return cluster_bbox

    best_slice = None
    best_area = 0
    for index, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        area = int((labels[slc] == index).sum())
        if area > best_area:
            best_area, best_slice = area, slc
    if best_slice is None:
        return cluster_bbox

    ys, xs = best_slice
    rw, rh = xs.stop - xs.start, ys.stop - ys.start
    fraction = (rw * rh) / (w * h)
    if not (cfg.refine_min_fraction <= fraction <= cfg.refine_max_fraction):
        return cluster_bbox
    if rw < w * 0.35 or rh < h * 0.35:
        return cluster_bbox
    return (x + xs.start, y + ys.start, rw, rh)


def _select_scored(
    scored: list[ScoredCandidate],
    threshold: float,
    config: DetectionConfig,
    width: int,
    height: int,
) -> list[BBox]:
    filtered = [item for item in scored if item.score >= threshold]
    filtered.sort(key=lambda item: item.score, reverse=True)
    if len(filtered) > 1:
        top_score = filtered[0].score
        gap = max(0.05, 0.08 - 0.02 * (top_score - 0.55))
        filtered = [filtered[0]] + [item for item in filtered[1:] if top_score - item.score <= gap]
    boxes = [item.box for item in filtered]
    boxes = merge_nearby_boxes(boxes, config.merge_distance_frac, width, height)
    scores = [item.score for item in filtered]
    if len(scores) != len(boxes):
        score_map = {item.box: item.score for item in filtered}
        scores = [score_map.get(box, 0.4) for box in boxes]
    selected = non_max_suppression(boxes, scores, config.nms_iou)
    return selected[: config.max_labels_per_frame]


def _fallback_boxes(work: NDArray[np.uint8], config: DetectionConfig) -> list[BBox]:
    height, width = work.shape
    frame_area = height * width
    masks = build_candidate_masks(work, config)
    min_area = max(800, int(height * width * config.min_area_frac * 0.8))
    boxes = boxes_from_mask(masks.dark_luma & ~masks.glare, min_area, config, frame_area)
    boxes = keep_label_clusters(boxes, config)
    if not boxes:
        return []
    scored = score_candidates(work, work, boxes, config)
    scored = [item for item in scored if item.score >= config.score_threshold_fallback]
    scored.sort(key=lambda item: item.score, reverse=True)
    return [item.box for item in scored[: max(1, config.max_labels_per_frame // 2)]]


def postprocess_boxes(
    image: NDArray[np.uint8],
    work: NDArray[np.uint8],
    raw_boxes: list[BBox],
    config: DetectionConfig,
    *,
    threshold: float | None = None,
) -> tuple[list[BBox], list[ScoredCandidate], list[BBox]]:
    height, width = image.shape
    scored = score_candidates(image, work, raw_boxes, config)
    cut = threshold if threshold is not None else config.score_threshold
    selected = _select_scored(scored, cut, config, width, height)
    if not selected:
        selected = _select_scored(scored, config.score_threshold_fallback, config, width, height)
    if not selected and scored:
        best = max(scored, key=lambda item: item.score)
        if best.score >= 0.30:
            selected = [best.box]
    kept = list(selected)
    used_fallback = False
    if not selected:
        selected = _fallback_boxes(work, config)
        used_fallback = bool(selected)
    refined = [refine_to_name_label(image, box, config) for box in selected]
    refined = [pad_box(box, width, height, config.crop_padding_frac) for box in refined]
    rescored = score_candidates(image, work, refined, config)
    floor = config.refine_score_floor * (0.55 if used_fallback else 1.0)
    refined = [item.box for item in rescored if item.score >= floor]
    if not refined and rescored:
        best = max(rescored, key=lambda item: item.score)
        min_keep = 0.30
        if best.score >= min_keep:
            refined = [best.box]
    return refined, scored, kept
