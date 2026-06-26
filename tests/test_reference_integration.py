"""Tests for reference-segmentation integration (TASK-01 through TASK-07)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from pdiseg.detection.config import DetectionConfig
from pdiseg.detection.masks import (
    build_candidate_masks,
    dark_body_mask,
    dog_text_mask,
    edge_density_mask,
)
from pdiseg.detection.scoring import analyze_bimodality, score_candidate


def _draw_text_block(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, bg: int = 30, ink: int = 220
) -> None:
    img[y0:y1, x0:x1] = bg
    for x in range(x0, x1, 8):
        img[y0:y1, x : x + 3] = ink


def test_clear_border_removes_mask_touching_frame_edge() -> None:
    work = np.full((120, 200), 80, dtype=np.uint8)
    work[0:40, 0:80] = 25
    _draw_text_block(work, 5, 5, 75, 35)

    cfg = DetectionConfig(clear_border_buffer_frac=0.04)
    masks = build_candidate_masks(work, cfg, gray=work)
    _, count = label(masks.combined)
    assert count == 0


def test_clear_border_keeps_interior_mask_component() -> None:
    work = np.full((200, 400), 80, dtype=np.uint8)
    work[60:140, 120:280] = 25
    _draw_text_block(work, 130, 70, 270, 130)

    cfg = DetectionConfig(clear_border_buffer_frac=0.04)
    masks = build_candidate_masks(work, cfg, gray=work)
    _, count = label(masks.combined)
    assert count >= 1


def test_edge_density_mask_highlights_text_block() -> None:
    gray = np.full((200, 400), 30, dtype=np.uint8)
    _draw_text_block(gray, 150, 80, 300, 140)

    mask = edge_density_mask(gray, DetectionConfig())
    assert mask[110, 220]
    assert int(mask.sum()) > 500


def test_edge_density_mask_is_quiet_on_uniform_field() -> None:
    gray = np.full((200, 400), 120, dtype=np.uint8)
    mask = edge_density_mask(gray, DetectionConfig())
    assert int(mask.sum()) == 0


def test_analyze_bimodality_detects_dark_and_light_regions() -> None:
    roi = np.concatenate(
        [np.full(1200, 40, dtype=np.uint8), np.full(800, 210, dtype=np.uint8)]
    )
    score, dark_frac, contrast = analyze_bimodality(roi, DetectionConfig())
    assert score > 0
    assert 0.4 < dark_frac < 0.7
    assert contrast > 50


def test_analyze_bimodality_rejects_uniform_patch() -> None:
    roi = np.full(400, 100, dtype=np.uint8)
    score, _, _ = analyze_bimodality(roi, DetectionConfig())
    assert score == 0.0


def test_bright_on_dark_scores_higher_than_flat_region() -> None:
    image = np.full((200, 400), 90, dtype=np.uint8)
    image[60:140, 120:280] = 35
    _draw_text_block(image, 130, 70, 270, 130, bg=35, ink=220)

    cfg = DetectionConfig()
    label_box = (120, 60, 160, 80)
    flat_box = (10, 10, 80, 60)
    label_score = score_candidate(image, image, label_box, cfg).score
    flat_score = score_candidate(image, image, flat_box, cfg).score
    assert label_score > flat_score


def test_dark_body_mask_marks_dark_rectangle() -> None:
    gray = np.full((200, 400), 150, dtype=np.uint8)
    gray[70:130, 140:260] = 30
    body = dark_body_mask(gray, DetectionConfig())
    assert body[100, 200]
    assert not body[10, 10]


def test_dog_text_mask_disabled_by_default() -> None:
    cfg = DetectionConfig(use_dog_text=False)
    assert cfg.use_dog_text is False


def test_dog_text_mask_finds_bright_text_on_dark_strip() -> None:
    gray = np.full((200, 400), 30, dtype=np.uint8)
    gray[60:140, 120:280] = 22
    _draw_text_block(gray, 130, 70, 270, 130, bg=22, ink=220)
    cfg = DetectionConfig(dog_bg_percentile=88.0, dog_contrast_threshold=20.0)
    mask = dog_text_mask(gray, cfg)
    assert int(mask.sum()) > 50


def test_lateral_margin_rejects_edge_hugging_box() -> None:
    from pdiseg.detection.postprocess import keep_label_clusters

    cfg = DetectionConfig(lateral_margin_frac=0.05)
    edge_box = (5, 80, 150, 100)
    inner_box = (200, 80, 150, 100)
    kept = keep_label_clusters([edge_box, inner_box], cfg, frame_width=500)
    assert inner_box in kept
    assert edge_box not in kept
