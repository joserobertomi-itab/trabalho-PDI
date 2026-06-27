"""Tunable detection thresholds (relative to frame size and percentiles)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """Relative thresholds for label detection (frame-size aware, no per-class tuning)."""

    score_threshold: float = 0.48
    score_threshold_fallback: float = 0.36
    score_relative_min: float = 0.58
    refine_score_floor: float = 0.36
    anchor_search_max_candidates: int = 28
    nms_iou: float = 0.30
    merge_distance_frac: float = 0.010
    min_area_frac: float = 0.0008
    max_area_frac: float = 0.10
    min_aspect: float = 0.20
    max_aspect: float = 5.0
    max_labels_per_frame: int = 6
    primary_cluster_only: bool = True
    additional_cluster_score_ratio: float = 0.92
    crop_padding_frac: float = 0.04
    glare_percentile: float = 92.0
    dark_percentile: float = 36.0
    text_local_window: int = 25
    text_offset: float = 14.0
    cluster_min_area_frac: float = 0.00045
    label_min_area: int = 1800
    label_max_area: int = 160000
    label_max_elongation: float = 5.0
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
    # TASK-06: DoG text mask for bright product text on a locally dark field.
    use_dog_text: bool = True
    dog_sigma: float = 45.0
    dog_contrast_threshold: float = 24.0
    dog_bg_percentile: float = 66.0
    dog_bold_size: int = 3
    dog_dilate_size: int = 11
    # TASK-07: adaptive dark-body + lateral margin
    lateral_margin_frac: float = 0.0
    # Ignore conveyor-side cardboard / box-edge text (asymmetric ROI).
    exclude_left_frac: float = 0.20
    exclude_right_frac: float = 0.08
    min_body_overlap: float = 0.0
    body_block_div: int = 6
    body_C: int = 18
    # Notebook-style hard gates (dark strip + bright glyphs)
    use_notebook_gates: bool = True
    gate_min_bright_on_dark: float = 0.035
    gate_max_background_level: float = 118.0
    gate_min_extent: float = 0.40
    gate_min_edge_density: float = 0.05
    # Final product-badge gate: precision-first emission filter.
    use_final_product_badge_gate: bool = True
    final_min_bright_on_dark: float = 0.040
    final_max_background_level: float = 114.0
    final_min_edge_density: float = 0.18
    final_min_extent: float = 0.45
    final_min_area_frac: float = 0.0025
    final_max_area_frac: float = 0.045
    final_min_aspect: float = 0.25
    final_max_aspect: float = 4.5
    cluster_expand_up_frac: float = 0.90
    cluster_expand_side_frac: float = 0.55
    cluster_expand_down_frac: float = 0.18
    cluster_context_max_gap_frac: float = 1.15
    cluster_context_min_axis_overlap: float = 0.22
    cluster_context_min_edge_density: float = 0.05
    cluster_context_min_bimodal_score: float = 0.10
    cluster_context_max_area_scale: float = 6.5
    fragment_group_iou: float = 0.12
    fragment_group_containment: float = 0.55
    fragment_group_gap_frac: float = 0.16
    fragment_group_max_scale_ratio: float = 3.0
