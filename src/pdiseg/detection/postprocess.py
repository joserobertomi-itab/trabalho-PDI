"""NMS, merge, Otsu refine, fallback, and score filtering."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_closing, binary_opening, find_objects, label
from skimage.filters import threshold_otsu

from pdiseg.core.boxes import box_area, clamp_box, dedupe_boxes, iou, non_max_suppression, pad_box
from pdiseg.core.imaging import BBox

from .candidates import find_candidate_boxes
from .config import DetectionConfig
from .masks import dark_body_mask, opened_background
from .roi import box_inside_horizontal_roi, horizontal_roi_bounds
from .scoring import ScoredCandidate, score_candidates


@dataclass(frozen=True)
class _ClusterDetection:
    cluster: BBox
    anchor: BBox
    score: float


def keep_label_clusters(
    candidates: list[BBox],
    config: DetectionConfig | None = None,
    *,
    frame_width: int | None = None,
) -> list[BBox]:
    cfg = config or DetectionConfig()
    x0, x1 = (0, frame_width or 0)
    if frame_width is not None:
        x0, x1 = horizontal_roi_bounds(frame_width, cfg)
    kept: list[BBox] = []
    for x, y, w, h in candidates:
        area = w * h
        elongation = max(w, h) / min(w, h)
        if not (cfg.label_min_area <= area <= cfg.label_max_area):
            continue
        if elongation > cfg.label_max_elongation:
            continue
        if frame_width is not None and not box_inside_horizontal_roi((x, y, w, h), x0, x1):
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
    close_shape = (
        max(3, min(9, h // 8)),
        max(3, min(15, w // 8)),
    )
    dark = binary_closing(dark, structure=np.ones(close_shape, dtype=bool))
    dark = binary_opening(dark, structure=np.ones((3, 3), dtype=bool))
    labels, count = label(dark)
    if count == 0:
        return cluster_bbox

    best_box: BBox | None = None
    best_score = 0.0
    for index, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        component = labels[slc] == index
        candidates = _component_plaque_candidates(region, slc, component, cfg)
        for candidate, component_area in candidates:
            score = _score_plaque_candidate(region, candidate, component_area, cfg)
            if score > best_score:
                best_score = score
                best_box = candidate
    if best_box is None:
        return cluster_bbox

    rx, ry, rw, rh = best_box
    return (x + rx, y + ry, rw, rh)


def _component_plaque_candidates(
    region: NDArray[np.uint8],
    slc: tuple[slice, slice],
    component: NDArray[np.bool_],
    config: DetectionConfig,
) -> list[tuple[BBox, int]]:
    ys, xs = slc
    base_box = (xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)
    candidates: list[tuple[BBox, int]] = [(base_box, int(component.sum()))]

    # IPython autoreload may update this function before injecting the new private
    # helper into an already-loaded module. Skip split candidates in that stale
    # kernel state instead of breaking notebook debug.
    split_fn = globals().get("_projection_split_boxes")
    split_boxes = split_fn(component, xs.start, ys.start, config) if split_fn else []
    for split_box in split_boxes:
        sx, sy, sw, sh = split_box
        local = component[sy - ys.start : sy - ys.start + sh, sx - xs.start : sx - xs.start + sw]
        candidates.append((split_box, int(local.sum())))
    return candidates


def _projection_split_boxes(
    mask: NDArray[np.bool_], x_offset: int, y_offset: int, config: DetectionConfig
) -> list[BBox]:
    boxes: list[BBox] = []
    for axis in (0, 1):
        length = mask.shape[axis]
        if length < 24:
            continue
        projection = mask.mean(axis=1 - axis)
        window = max(3, min(11, length // 12))
        if window % 2 == 0:
            window += 1
        smooth = np.convolve(projection, np.ones(window) / window, mode="same")
        start = int(length * 0.28)
        stop = int(length * 0.72)
        if stop <= start:
            continue
        split = start + int(np.argmin(smooth[start:stop]))
        if smooth[split] > max(0.08, float(smooth.max()) * 0.55):
            continue
        parts = (
            (slice(0, split), slice(None)) if axis == 0 else (slice(None), slice(0, split)),
            (slice(split, length), slice(None))
            if axis == 0
            else (slice(None), slice(split, length)),
        )
        for rows, cols in parts:
            part = mask[rows, cols]
            if int(part.sum()) < max(80, int(mask.size * config.refine_min_fraction * 0.25)):
                continue
            coords = np.argwhere(part)
            if coords.size == 0:
                continue
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            if axis == 0 and rows.start is not None:
                y0 += rows.start
                y1 += rows.start
            if axis == 1 and cols.start is not None:
                x0 += cols.start
                x1 += cols.start
            boxes.append((x_offset + int(x0), y_offset + int(y0), int(x1 - x0), int(y1 - y0)))
    return boxes


def _score_plaque_candidate(
    region: NDArray[np.uint8], box: BBox, component_area: int, config: DetectionConfig
) -> float:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return 0.0
    cluster_h, cluster_w = region.shape
    fraction = (w * h) / max(region.size, 1)
    if not (config.refine_min_fraction <= fraction <= config.refine_max_fraction):
        return 0.0
    if w < cluster_w * 0.18 or h < cluster_h * 0.24:
        return 0.0

    aspect = w / max(h, 1)
    if aspect < config.final_min_aspect * 0.75 or aspect > config.final_max_aspect * 1.25:
        return 0.0

    fill = component_area / max(w * h, 1)
    if fill < 0.25:
        return 0.0

    roi = region[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    mean = float(roi.mean())
    bright = float((roi > mean + config.bright_on_dark_offset).mean())

    bright_score = min(1.0, bright / max(config.final_min_bright_on_dark, 1e-6))
    fill_score = max(0.0, 1.0 - abs(fill - 0.68) / 0.42)
    if fraction < 0.12:
        area_score = fraction / 0.12
    elif fraction > 0.68:
        area_score = max(0.1, 1.0 - (fraction - 0.68) / 0.32)
    else:
        area_score = 1.0
    aspect_score = 1.0
    if aspect < config.final_min_aspect:
        aspect_score = aspect / config.final_min_aspect
    elif aspect > config.final_max_aspect:
        aspect_score = config.final_max_aspect / aspect
    dark_score = max(0.0, (125.0 - mean) / 125.0)

    score = (
        0.45 * bright_score
        + 0.22 * fill_score
        + 0.18 * area_score
        + 0.10 * aspect_score
        + 0.05 * dark_score
    )
    if bright < config.final_min_bright_on_dark * 0.5 and fill > 0.92:
        score *= 0.25
    return float(score)


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


def _final_area_bounds(config: DetectionConfig, frame_area: float) -> tuple[float, float]:
    min_area = config.final_min_area_frac
    max_area = config.final_max_area_frac
    if frame_area and frame_area < 300_000:
        min_area = min(min_area, config.min_area_frac)
        max_area = max(max_area, 0.20)
    return min_area, max_area


def _passes_final_product_badge_gate(item: ScoredCandidate, config: DetectionConfig) -> bool:
    if not config.use_final_product_badge_gate:
        return True

    features = item.features
    bright = features.get("bright_on_dark", 0.0)
    background = features.get("background_level", 255.0)
    edge = features.get("edge_density", 0.0)
    extent = features.get("extent", 0.0)
    frame_area = features.get("frame_area", 0.0)
    area_frac = features.get("area_frac", box_area(item.box) / max(frame_area, 1.0))
    aspect = features.get("aspect", item.box[2] / max(item.box[3], 1))

    min_area, max_area = _final_area_bounds(config, frame_area)
    if bright < config.final_min_bright_on_dark:
        return False
    if background > config.final_max_background_level:
        return False
    if edge < config.final_min_edge_density:
        return False
    if extent < config.final_min_extent:
        return False
    if not (min_area <= area_frac <= max_area):
        return False
    if not (config.final_min_aspect <= aspect <= config.final_max_aspect):
        return False
    if config.min_body_overlap > 0 and features.get("body_overlap", 0.0) < config.min_body_overlap:
        return False
    return _passes_notebook_gates(item, config)


def _passes_relaxed_product_anchor_gate(item: ScoredCandidate, config: DetectionConfig) -> bool:
    features = item.features
    frame_area = features.get("frame_area", 0.0)
    area_frac = features.get("area_frac", box_area(item.box) / max(frame_area, 1.0))
    aspect = features.get("aspect", item.box[2] / max(item.box[3], 1))
    min_area, max_area = _final_area_bounds(config, frame_area)
    if frame_area >= 300_000:
        min_area = max(min_area, 0.006)
    frame_width = features.get("frame_width", 0.0)
    if frame_width > 0:
        x0, x1 = horizontal_roi_bounds(int(frame_width), config)
        x, _, w, _ = item.box
        if not box_inside_horizontal_roi((x, item.box[1], w, item.box[3]), x0, x1):
            return False

    if not (min_area <= area_frac <= max_area):
        return False
    if not (config.final_min_aspect <= aspect <= config.final_max_aspect):
        return False
    if features.get("background_level", 255.0) > config.final_max_background_level + 18.0:
        return False
    if features.get("edge_density", 0.0) < max(0.20, config.final_min_edge_density * 0.90):
        return False
    if features.get("text_density", 0.0) < 0.18:
        return False
    if features.get("dark_density", 0.0) < 0.20:
        return False
    extent = features.get("extent", 0.0)
    if not (0.30 <= extent <= 0.62):
        return False
    return not (
        features.get("bimodal_score", 0.0) < 0.22 and features.get("bimodal_contrast", 0.0) < 24.0
    )


def _passes_product_anchor_gate(
    item: ScoredCandidate, config: DetectionConfig, *, allow_relaxed: bool
) -> bool:
    return _passes_final_product_badge_gate(item, config) or (
        allow_relaxed and _passes_relaxed_product_anchor_gate(item, config)
    )


def extract_product_anchor(
    image: NDArray[np.uint8],
    work: NDArray[np.uint8],
    cluster_bbox: BBox,
    config: DetectionConfig | None = None,
    *,
    allow_relaxed: bool = False,
    opened: NDArray[np.uint8] | None = None,
    body: NDArray[np.bool_] | None = None,
) -> BBox | None:
    """Return a product-type badge anchor inside a cluster, or None for brand/noise."""
    cfg = config or DetectionConfig()
    height, width = image.shape[:2]
    cluster = clamp_box(cluster_bbox, width, height)
    refined = refine_to_name_label(image, cluster, cfg)

    if refined == cluster:
        anchor_candidates = [cluster]
    else:
        anchor_candidates = [
            refined,
            pad_box(refined, width, height, cfg.crop_padding_frac * 0.5),
        ]

    unique = dedupe_boxes(anchor_candidates, iou_threshold=0.80)
    scored = score_candidates(image, work, unique, cfg, opened=opened, body=body)
    anchors = [
        item
        for item in scored
        if _passes_product_anchor_gate(item, cfg, allow_relaxed=allow_relaxed)
    ]
    if not anchors:
        return None
    anchors.sort(key=lambda item: item.score, reverse=True)
    return anchors[0].box


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


def _intersection_fraction(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / max(1, min(box_area(a), box_area(b)))


def _axis_overlap_fraction(a0: int, a1: int, b0: int, b1: int) -> float:
    overlap = max(0, min(a1, b1) - max(a0, b0))
    return overlap / max(1, min(a1 - a0, b1 - b0))


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, max(a0, b0) - min(a1, b1))


def _union_box(a: BBox, b: BBox, width: int, height: int) -> BBox:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return clamp_box((x0, y0, x1 - x0, y1 - y0), width, height)


def _contains_anchor_context(anchor: BBox, candidate: BBox, config: DetectionConfig) -> bool:
    if _intersection_fraction(anchor, candidate) < 0.55:
        return False
    scale = box_area(candidate) / max(1, box_area(anchor))
    return scale <= config.cluster_context_max_area_scale


def _has_context_signal(item: ScoredCandidate, config: DetectionConfig) -> bool:
    features = item.features
    return (
        features.get("edge_density", 0.0) >= config.cluster_context_min_edge_density
        or features.get("bimodal_score", 0.0) >= config.cluster_context_min_bimodal_score
        or features.get("text_density", 0.0) >= 0.025
        or features.get("bright_on_dark", 0.0) >= config.final_min_bright_on_dark * 0.5
    )


def _cluster_max_area_frac(config: DetectionConfig, frame_area: int) -> float:
    max_frac = max(config.max_area_frac, config.final_max_area_frac * 2.5)
    if frame_area < 300_000:
        max_frac = max(max_frac, 0.50)
    return max_frac


def _is_adjacent_label_context(
    anchor: BBox, candidate: BBox, item: ScoredCandidate, config: DetectionConfig
) -> bool:
    if not _has_context_signal(item, config):
        return False

    ax, ay, aw, ah = anchor
    bx, by, bw, bh = candidate
    gap_x = _axis_gap(ax, ax + aw, bx, bx + bw)
    gap_y = _axis_gap(ay, ay + ah, by, by + bh)
    overlap_x = _axis_overlap_fraction(ax, ax + aw, bx, bx + bw)
    overlap_y = _axis_overlap_fraction(ay, ay + ah, by, by + bh)
    max_gap_x = max(8.0, aw * config.cluster_context_max_gap_frac)
    max_gap_y = max(8.0, ah * config.cluster_context_max_gap_frac)

    above_or_touching = by + bh <= ay + int(ah * 0.35)
    upper_context = (
        above_or_touching
        and gap_y <= max_gap_y
        and overlap_x >= (config.cluster_context_min_axis_overlap * 0.65)
    )
    side_context = (
        gap_x <= max_gap_x
        and overlap_y >= config.cluster_context_min_axis_overlap
        and by + bh / 2 <= ay + ah * 0.65
    )
    upper_diagonal = above_or_touching and gap_x <= max_gap_x and gap_y <= max_gap_y
    return upper_context or side_context or upper_diagonal


def _cluster_area_ok(cluster: BBox, anchor: BBox, config: DetectionConfig, frame_area: int) -> bool:
    area = box_area(cluster)
    if area > config.label_max_area:
        return False
    if area / max(frame_area, 1) > _cluster_max_area_frac(config, frame_area):
        return False
    scale = area / max(1, box_area(anchor))
    return scale <= config.cluster_context_max_area_scale


def expand_to_label_cluster(
    image: NDArray[np.uint8],
    work: NDArray[np.uint8],
    anchor_box: BBox,
    candidate_pool: list[BBox],
    config: DetectionConfig | None = None,
    *,
    scored_pool: list[ScoredCandidate] | None = None,
    opened: NDArray[np.uint8] | None = None,
    body: NDArray[np.bool_] | None = None,
) -> BBox:
    """Expand a product anchor toward adjacent brand/context without losing the anchor."""
    cfg = config or DetectionConfig()
    height, width = image.shape[:2]
    frame_area = height * width
    anchor = clamp_box(anchor_box, width, height)
    cluster = anchor
    context_found = False

    pool = dedupe_boxes([*candidate_pool, anchor], iou_threshold=0.70)
    if scored_pool is None:
        scored_pool = score_candidates(image, work, pool, cfg, opened=opened, body=body)
    compatible_contexts: list[tuple[float, ScoredCandidate, BBox]] = []
    for item in scored_pool:
        candidate = clamp_box(item.box, width, height)
        if candidate == anchor:
            continue
        contains_anchor = _contains_anchor_context(anchor, candidate, cfg)
        adjacent_context = _is_adjacent_label_context(anchor, candidate, item, cfg)
        if not (contains_anchor or adjacent_context):
            continue
        ax, ay, _, ah = anchor
        bx, by, bw, bh = candidate
        rank = item.score
        if by + bh <= ay + int(ah * 0.35):
            rank += 0.30
        elif by < ay:
            rank += 0.10
        if contains_anchor:
            rank += 0.08
        if bx <= ax <= bx + bw:
            rank += 0.03
        compatible_contexts.append((rank, item, candidate))

    compatible_contexts.sort(key=lambda item: item[0], reverse=True)
    for _, _, candidate in compatible_contexts:
        merged = _union_box(cluster, candidate, width, height)
        if not _cluster_area_ok(merged, anchor, cfg, frame_area):
            continue
        cluster = merged
        context_found = True

    if context_found:
        return pad_box(cluster, width, height, cfg.crop_padding_frac * 0.5)

    x, y, w, h = anchor
    left = int(w * cfg.cluster_expand_side_frac)
    right = int(w * cfg.cluster_expand_side_frac)
    up = int(h * cfg.cluster_expand_up_frac)
    down = int(h * cfg.cluster_expand_down_frac)
    return clamp_box((x - left, y - up, w + left + right, h + up + down), width, height)


def _compatible_fragments(
    a: BBox, b: BBox, config: DetectionConfig, median_w: float, median_h: float
) -> bool:
    if iou(a, b) >= config.fragment_group_iou:
        return True
    if _intersection_fraction(a, b) >= config.fragment_group_containment:
        return True

    area_a = box_area(a)
    area_b = box_area(b)
    scale_ratio = max(area_a, area_b) / max(1, min(area_a, area_b))
    if scale_ratio > config.fragment_group_max_scale_ratio:
        return False

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    gap_x = _axis_gap(ax, ax + aw, bx, bx + bw)
    gap_y = _axis_gap(ay, ay + ah, by, by + bh)
    overlap_x = _axis_overlap_fraction(ax, ax + aw, bx, bx + bw)
    overlap_y = _axis_overlap_fraction(ay, ay + ah, by, by + bh)
    max_gap_x = max(8.0, median_w * config.fragment_group_gap_frac)
    max_gap_y = max(8.0, median_h * config.fragment_group_gap_frac)
    same_row = gap_x <= max_gap_x and overlap_y >= 0.35
    same_column = gap_y <= max_gap_y and overlap_x >= 0.35
    return same_row or same_column


def _group_scored_fragments(
    scored: list[ScoredCandidate], config: DetectionConfig, frame_size: tuple[int, int]
) -> list[tuple[BBox, float]]:
    if not scored:
        return []

    height, width = frame_size
    median_w = float(np.median([item.box[2] for item in scored]))
    median_h = float(np.median([item.box[3] for item in scored]))
    parent = list(range(len(scored)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, item_a in enumerate(scored):
        for j in range(i + 1, len(scored)):
            if _compatible_fragments(item_a.box, scored[j].box, config, median_w, median_h):
                union(i, j)

    groups: dict[int, list[ScoredCandidate]] = {}
    for index, item in enumerate(scored):
        groups.setdefault(find(index), []).append(item)

    grouped: list[tuple[BBox, float]] = []
    for members in groups.values():
        x0 = min(item.box[0] for item in members)
        y0 = min(item.box[1] for item in members)
        x1 = max(item.box[0] + item.box[2] for item in members)
        y1 = max(item.box[1] + item.box[3] for item in members)
        score = max(item.score for item in members)
        grouped.append((clamp_box((x0, y0, x1 - x0, y1 - y0), width, height), score))

    grouped.sort(key=lambda item: item[1], reverse=True)
    return grouped


def group_label_fragments(
    scored_candidates: list[ScoredCandidate],
    config: DetectionConfig,
    frame_size: tuple[int, int],
) -> list[BBox]:
    """Merge eligible word fragments into one candidate per visible package label."""
    eligible = [
        item for item in scored_candidates if _passes_final_product_badge_gate(item, config)
    ]
    return [box for box, _ in _group_scored_fragments(eligible, config, frame_size)]


def _recall_candidate_boxes(
    image: NDArray[np.uint8], work: NDArray[np.uint8], config: DetectionConfig
) -> list[BBox]:
    attempts = [
        replace(config, use_dog_text=True),
        replace(
            config,
            use_dog_text=True,
            dog_contrast_threshold=max(14.0, config.dog_contrast_threshold * 0.75),
            dog_bg_percentile=max(config.dog_bg_percentile, 72.0),
            cluster_min_area_frac=config.cluster_min_area_frac * 0.75,
        ),
        replace(
            config,
            use_dog_text=True,
            dog_contrast_threshold=max(12.0, config.dog_contrast_threshold * 0.62),
            dog_bg_percentile=max(config.dog_bg_percentile, 78.0),
            text_offset=config.text_offset * 0.80,
            dark_percentile=max(config.dark_percentile, 42.0),
            cluster_min_area_frac=config.cluster_min_area_frac * 0.55,
        ),
    ]
    boxes: list[BBox] = []
    for attempt in attempts:
        boxes.extend(find_candidate_boxes(work, attempt, text_source=image))
    return dedupe_boxes(
        keep_label_clusters(boxes, config, frame_width=work.shape[1]) or boxes, 0.55
    )


def _build_cluster_detections(
    image: NDArray[np.uint8],
    work: NDArray[np.uint8],
    raw_boxes: list[BBox],
    scored: list[ScoredCandidate],
    config: DetectionConfig,
    *,
    allow_relaxed_anchor: bool = False,
    opened: NDArray[np.uint8] | None = None,
    body: NDArray[np.bool_] | None = None,
) -> list[_ClusterDetection]:
    if not scored:
        return []

    height, width = image.shape[:2]
    frame_area = height * width
    context_scored = list(scored)
    scored = _dedupe_scored(scored, iou_threshold=0.55)
    scored.sort(key=lambda item: item.score, reverse=True)
    top_score = scored[0].score if scored else 0.0
    score_floor = min(config.score_threshold_fallback * 0.55, top_score * 0.28)

    detections: list[_ClusterDetection] = []
    anchor_scores: dict[BBox, ScoredCandidate] = {}
    pool = dedupe_boxes(raw_boxes + [item.box for item in context_scored], iou_threshold=0.70)
    scored_pool = context_scored
    search_items = [item for item in scored if item.score >= score_floor]
    search_items = search_items[: config.anchor_search_max_candidates]
    for item in search_items:
        if item.score < score_floor:
            continue
        anchor = extract_product_anchor(
            image,
            work,
            item.box,
            config,
            allow_relaxed=allow_relaxed_anchor,
            opened=opened,
            body=body,
        )
        if anchor is None:
            continue
        anchor_score = anchor_scores.get(anchor)
        if anchor_score is None:
            anchor_score = score_candidates(
                image, work, [anchor], config, opened=opened, body=body
            )[0]
            anchor_scores[anchor] = anchor_score
        if not _passes_product_anchor_gate(
            anchor_score, config, allow_relaxed=allow_relaxed_anchor
        ):
            continue
        if anchor_score.score < config.refine_score_floor:
            continue
        cluster = expand_to_label_cluster(
            image,
            work,
            anchor,
            pool,
            config,
            scored_pool=list(scored_pool),
            opened=opened,
            body=body,
        )
        if box_area(cluster) <= 0 or box_area(cluster) / max(
            frame_area, 1
        ) > _cluster_max_area_frac(config, frame_area):
            continue
        expansion_bonus = min(0.10, max(0, box_area(cluster) - box_area(anchor)) / frame_area * 4.0)
        score = min(1.0, anchor_score.score * 0.76 + item.score * 0.18 + expansion_bonus)
        detections.append(_ClusterDetection(cluster=cluster, anchor=anchor, score=score))

    detections.sort(key=lambda item: item.score, reverse=True)
    return detections


def _select_clusters(
    detections: list[_ClusterDetection], config: DetectionConfig
) -> list[_ClusterDetection]:
    if not detections:
        return []

    boxes = [item.cluster for item in detections]
    scores = [item.score for item in detections]
    kept_boxes = non_max_suppression(boxes, scores, config.nms_iou)
    best_by_box = {item.cluster: item for item in detections}
    selected = [best_by_box[box] for box in kept_boxes if box in best_by_box]
    if not selected:
        return []

    top_score = selected[0].score
    if config.primary_cluster_only:
        return [selected[0]]

    floor = top_score * config.additional_cluster_score_ratio
    return [item for item in selected if item.score >= floor][: config.max_labels_per_frame]


def _selected_anchors(detections: list[_ClusterDetection]) -> list[BBox]:
    return [item.anchor for item in detections]


def _selected_clusters(detections: list[_ClusterDetection]) -> list[BBox]:
    return [item.cluster for item in detections]


def postprocess_boxes(
    image: NDArray[np.uint8],
    work: NDArray[np.uint8],
    raw_boxes: list[BBox],
    config: DetectionConfig,
    *,
    threshold: float | None = None,
) -> tuple[list[BBox], list[ScoredCandidate], list[BBox]]:
    opened = opened_background(image, config)
    body: NDArray[np.bool_] | None = None
    if config.min_body_overlap > 0:
        body = dark_body_mask(image, config)

    scored = score_candidates(image, work, raw_boxes, config, opened=opened, body=body)
    cut = threshold if threshold is not None else config.score_threshold
    if threshold is not None:
        scored = [item for item in scored if item.score >= cut]

    detections = _build_cluster_detections(
        image, work, raw_boxes, scored, config, opened=opened, body=body
    )
    if not detections:
        detections = _build_cluster_detections(
            image,
            work,
            raw_boxes,
            scored,
            config,
            allow_relaxed_anchor=True,
            opened=opened,
            body=body,
        )

    if not detections:
        recall_boxes = _recall_candidate_boxes(image, work, config)
        if recall_boxes:
            raw_boxes = dedupe_boxes(raw_boxes + recall_boxes, iou_threshold=0.85)
            scored = score_candidates(image, work, raw_boxes, config, opened=opened, body=body)
            detections = _build_cluster_detections(
                image,
                work,
                raw_boxes,
                scored,
                config,
                allow_relaxed_anchor=True,
                opened=opened,
                body=body,
            )

    selected = _select_clusters(detections, config)
    return _selected_clusters(selected), scored, _selected_anchors(selected)
