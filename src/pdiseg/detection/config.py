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
    # TASK-01: clear_border on combined mask
    clear_border_buffer_frac: float = 0.04
    # TASK-02: Sobel edge-density candidate mask
    edge_mag_threshold: float = 55.0
    edge_density_window_frac: float = 0.087
    edge_density_min: float = 0.19
    edge_close_size: int = 21
    edge_open_size: int = 9
    # TASK-03: opened background + bright-on-dark scoring
    opened_background_size: int = 13
    bright_on_dark_offset: float = 50.0
    background_level_max: float = 114.0
    # TASK-04: extent + bimodality
    extent_target: float = 0.48
    bimodality_min_class_frac: float = 0.30
    # TASK-06: optional DoG text mask (off by default)
    use_dog_text: bool = False
    dog_sigma: float = 45.0
    dog_contrast_threshold: float = 24.0
    dog_bg_percentile: float = 66.0
    dog_bold_size: int = 3
    dog_dilate_size: int = 11
    # TASK-07: adaptive dark-body + lateral margin
    lateral_margin_frac: float = 0.025
    min_body_overlap: float = 0.0
    body_block_div: int = 6
    body_C: int = 18
