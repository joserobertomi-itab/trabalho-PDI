"""Tunable detection thresholds (relative to frame size and percentiles)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """Relative thresholds for label detection (frame-size aware, no per-class tuning)."""

    score_threshold: float = 0.54
    score_threshold_fallback: float = 0.44
    refine_score_floor: float = 0.38
    nms_iou: float = 0.45
    merge_distance_frac: float = 0.018
    min_area_frac: float = 0.0010
    max_area_frac: float = 0.09
    min_aspect: float = 0.25
    max_aspect: float = 4.5
    max_labels_per_frame: int = 2
    crop_padding_frac: float = 0.03
    glare_percentile: float = 92.0
    dark_percentile: float = 36.0
    text_local_window: int = 25
    text_offset: float = 14.0
    cluster_min_area_frac: float = 0.00045
    label_min_area: int = 3200
    label_max_area: int = 140000
    label_max_elongation: float = 4.0
    refine_min_fraction: float = 0.05
    refine_max_fraction: float = 0.90
