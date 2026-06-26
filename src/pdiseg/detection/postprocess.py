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
    candidates: list[BBox],
    config: DetectionConfig | None = None,
    *,
    frame_width: int | None = None,
) -> list[BBox]:
    cfg = config or DetectionConfig()
    lateral = 0
    if cfg.lateral_margin_frac > 0 and frame_width is not None:
        lateral = int(frame_width * cfg.lateral_margin_frac)
    kept: list[BBox] = []
    for x, y, w, h in candidates:
        area = w * h
        elongation = max(w, h) / min(w, h)
        if not (cfg.label_min_area <= area <= cfg.label_max_area):
            continue
        if elongation > cfg.label_max_elongation:
            continue
        if lateral > 0 and (x < lateral or x + w > frame_width - lateral):  # type: ignore[operator]
            continue
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


def _passes_notebook_gates(item: ScoredCandidate, config: DetectionConfig) -> bool:
    if not config.use_notebook_gates:
        return True
    features = item.features
    bright = features.get("bright_on_dark", 0.0)
    background = features.get("background_level", 255.0)
    extent = features.get("extent", 0.0)
    edge = features.get("edge_density", 0.0)
    dark = features.get("dark_density", 0.0)
    if dark >= 0.42 and bright >= config.gate_min_bright_on_dark * 0.85:
        return True
    if bright < config.gate_min_bright_on_dark:
        return False
    if background > config.gate_max_background_level:
        return False
    if extent < config.gate_min_extent:
        return False
    return edge >= config.gate_min_edge_density


def _dedupe_scored(scored: list[ScoredCandidate], iou_threshold: float) -> list[ScoredCandidate]:
    if not scored:
        return []
    boxes = [item.box for item in scored]
    scores = [item.score for item in scored]
    kept_boxes = non_max_suppression(boxes, scores, iou_threshold)
    best: dict[BBox, ScoredCandidate] = {}
    for item in scored:
        if item.box not in kept_boxes:
            continue
        prev = best.get(item.box)
        if prev is None or item.score > prev.score:
            best[item.box] = item
    order = {box: index for index, box in enumerate(kept_boxes)}
    return sorted(best.values(), key=lambda item: order.get(item.box, 999))


def _select_scored(
    scored: list[ScoredCandidate],
    threshold: float,
    config: DetectionConfig,
    width: int,
    height: int,
) -> list[BBox]:
    if not scored:
        return []
    scored = _dedupe_scored(scored, iou_threshold=0.55)
    scored.sort(key=lambda item: item.score, reverse=True)
    top_score = scored[0].score
    relative_floor = max(threshold, top_score * config.score_relative_min)
    filtered = [item for item in scored if item.score >= relative_floor]
    if not filtered:
        filtered = [item for item in scored if item.score >= config.score_threshold_fallback]
    if not filtered:
        filtered = [scored[0]]
    gated = [item for item in filtered if _passes_notebook_gates(item, config)]
    if gated:
        filtered = gated
    boxes = [item.box for item in filtered]
    scores = [item.score for item in filtered]
    if config.merge_distance_frac > 0:
        boxes = merge_nearby_boxes(boxes, config.merge_distance_frac, width, height)
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
    boxes = keep_label_clusters(boxes, config, frame_width=width)
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
    padded = [pad_box(box, width, height, config.crop_padding_frac) for box in refined]
    rescored = score_candidates(image, work, padded, config)
    floor = config.refine_score_floor * (0.55 if used_fallback else 1.0)
    refined = [item.box for item in rescored if item.score >= floor]
    if not refined and rescored:
        best = max(rescored, key=lambda item: item.score)
        min_keep = 0.30
        if best.score >= min_keep:
            refined = [best.box]
    if not refined and padded:
        refined = padded
    return refined, scored, kept
