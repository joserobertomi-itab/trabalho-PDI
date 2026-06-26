"""Multi-feature score for each candidate bounding box."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import sobel, uniform_filter
from skimage.filters import threshold_otsu

from pdiseg.core.boxes import box_area
from pdiseg.core.imaging import BBox

from .config import DetectionConfig
from .masks import dark_body_mask, glare_mask, opened_background


@dataclass(frozen=True)
class ScoredCandidate:
    box: BBox
    score: float
    features: dict[str, float]


def analyze_bimodality(
    roi: NDArray[np.uint8], config: DetectionConfig
) -> tuple[float, float, float]:
    """Return (score, dark_fraction, contrast) for a bimodal dark+light region."""
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    try:
        thresh = float(threshold_otsu(roi))  # type: ignore[no-untyped-call]
    except ValueError:
        thresh = float(np.mean(roi))
    dark = roi[roi <= thresh]
    light = roi[roi > thresh]
    if dark.size == 0 or light.size == 0:
        return 0.0, 0.0, 0.0
    dark_frac = float(dark.size / roi.size)
    light_frac = 1.0 - dark_frac
    if min(dark_frac, light_frac) < config.bimodality_min_class_frac:
        return 0.0, dark_frac, 0.0
    contrast = float(light.mean() - dark.mean())
    balance = 1.0 - abs(0.5 - dark_frac)
    score = contrast * balance
    return score, dark_frac, contrast


def _ring_mean(image: NDArray[np.uint8], box: BBox, margin: int = 6) -> float:
    x, y, w, h = box
    height, width = image.shape
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + w + margin)
    y1 = min(height, y + h + margin)
    ring = image[y0:y1, x0:x1].copy()
    ring[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = 0
    values = ring[ring > 0]
    return float(values.mean()) if values.size else float(image.mean())


def score_candidate(
    image: NDArray[np.uint8],
    work: NDArray[np.uint8],
    box: BBox,
    config: DetectionConfig,
    *,
    opened: NDArray[np.uint8] | None = None,
    body: NDArray[np.bool_] | None = None,
) -> ScoredCandidate:
    height, width = image.shape
    frame_area = height * width
    x, y, w, h = box
    region = work[y : y + h, x : x + w]
    region_orig = image[y : y + h, x : x + w]
    if region.size == 0:
        return ScoredCandidate(box=box, score=0.0, features={})

    area_frac = box_area(box) / frame_area
    aspect = w / max(h, 1)
    elongation = max(aspect, 1 / max(aspect, 1e-6))

    global_dark = float(np.percentile(work, config.dark_percentile))
    dark_density = float((region <= global_dark).mean())

    local = region.astype(np.float32)
    local_mean = uniform_filter(local, size=min(config.text_local_window, max(3, min(w, h) // 3)))
    text_pixels = local > local_mean + config.text_offset * 0.55
    text_density = float(text_pixels.mean())

    gx = sobel(region.astype(np.float32), axis=1)
    gy = sobel(region.astype(np.float32), axis=0)
    edge_density = float(np.hypot(gx, gy).mean()) / 255.0
    if region.size < 2500:
        edge_density = max(edge_density, float(np.std(region)) / 96.0)
    texture = float(region.std()) / 128.0
    if region.size < 2500:
        texture = max(texture, float(np.std(region_orig)) / 96.0)

    glare = glare_mask(region_orig, config)
    glare_fraction = float(glare.mean())
    if dark_density > 0.72:
        glare_fraction *= 0.25

    inner_mean = float(region.mean())
    ring_mean = _ring_mean(work, box)
    contrast = max(0.0, (ring_mean - inner_mean) / 128.0)

    opened_patch = (
        opened[y : y + h, x : x + w]
        if opened is not None
        else opened_background(image, config)[y : y + h, x : x + w]
    )
    background_level = float(opened_patch.mean())
    background_score = max(0.0, (config.background_level_max - background_level) / config.background_level_max)
    bright_on_dark = float((region_orig > opened_patch + config.bright_on_dark_offset).mean())

    try:
        fg_thresh = float(threshold_otsu(region_orig))  # type: ignore[no-untyped-call]
    except ValueError:
        fg_thresh = float(region_orig.mean())
    extent = float((region_orig <= fg_thresh).mean()) if region_orig.size else 0.0
    extent_score = math.exp(-((extent - config.extent_target) / 0.15) ** 2)

    bimodal_raw, bimodal_dark_frac, bimodal_contrast = analyze_bimodality(region_orig, config)
    bimodal_score = min(1.0, bimodal_raw / 80.0)

    body_overlap = (
        float(body[y : y + h, x : x + w].mean()) if body is not None else dark_density
    )

    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    position_score = 1.0 - 0.35 * abs(cx - 0.52) - 0.25 * max(0.0, cy - 0.72)

    area_score = 1.0
    if area_frac < config.min_area_frac:
        area_score = area_frac / config.min_area_frac
    elif area_frac > config.max_area_frac:
        overshoot = area_frac - config.max_area_frac
        area_score = max(0.35, 1.0 - overshoot / 0.20)

    aspect_score = 1.0
    if aspect < config.min_aspect:
        aspect_score = aspect / config.min_aspect
    elif aspect > config.max_aspect:
        aspect_score = max(0.0, config.max_aspect / aspect)
    if elongation > config.max_aspect:
        aspect_score *= max(0.0, config.max_aspect / elongation)

    score = (
        0.17 * min(1.0, dark_density * 1.35)
        + 0.15 * min(1.0, text_density * 3.5)
        + 0.11 * min(1.0, edge_density * 2.5)
        + 0.08 * min(1.0, texture)
        + 0.09 * min(1.0, contrast)
        + 0.07 * area_score
        + 0.05 * aspect_score
        + 0.05 * max(0.0, position_score)
        + 0.07 * min(1.0, background_score)
        + 0.08 * min(1.0, bright_on_dark / 0.20)
        + 0.06 * extent_score
        + 0.07 * bimodal_score
        + 0.05 * min(1.0, body_overlap * 1.2)
        - 0.40 * min(1.0, glare_fraction * 2.2)
    )
    if dark_density < 0.18:
        score *= 0.55
    if text_density < 0.02 and edge_density < 0.06:
        score *= 0.65
    if glare_fraction > 0.35:
        score *= 0.5
    if config.min_body_overlap > 0 and body_overlap < config.min_body_overlap:
        score *= 0.45
    score = float(np.clip(score, 0.0, 1.0))

    features = {
        "area_frac": area_frac,
        "aspect": aspect,
        "dark_density": dark_density,
        "text_density": text_density,
        "edge_density": edge_density,
        "texture": texture,
        "glare_fraction": glare_fraction,
        "contrast": contrast,
        "position_score": position_score,
        "area_score": area_score,
        "aspect_score": aspect_score,
        "background_level": background_level,
        "background_score": background_score,
        "bright_on_dark": bright_on_dark,
        "extent": extent,
        "extent_score": extent_score,
        "bimodal_score": bimodal_score,
        "bimodal_dark_frac": bimodal_dark_frac,
        "bimodal_contrast": bimodal_contrast,
        "body_overlap": body_overlap,
    }
    return ScoredCandidate(box=box, score=score, features=features)


def score_candidates(
    image: NDArray[np.uint8],
    work: NDArray[np.uint8],
    boxes: list[BBox],
    config: DetectionConfig,
) -> list[ScoredCandidate]:
    if not boxes:
        return []
    opened = opened_background(image, config)
    body: NDArray[np.bool_] | None = None
    if config.min_body_overlap > 0:
        body = dark_body_mask(image, config)
    return [
        score_candidate(image, work, box, config, opened=opened, body=body) for box in boxes
    ]
