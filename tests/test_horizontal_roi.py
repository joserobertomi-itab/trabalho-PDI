import numpy as np

from pdiseg.detection.config import DetectionConfig
from pdiseg.detection.masks import build_candidate_masks
from pdiseg.detection.postprocess import keep_label_clusters
from pdiseg.detection.preprocess import preprocess_image
from pdiseg.detection.roi import horizontal_roi_bounds


def test_horizontal_roi_bounds_asymmetric_defaults() -> None:
    cfg = DetectionConfig()
    x0, x1 = horizontal_roi_bounds(1000, cfg)
    assert x0 == 200
    assert x1 == 920


def test_exclude_left_rejects_cardboard_side_box() -> None:
    cfg = DetectionConfig(exclude_left_frac=0.18, exclude_right_frac=0.08)
    left_box = (20, 80, 150, 100)
    inner_box = (250, 80, 150, 100)
    kept = keep_label_clusters([left_box, inner_box], cfg, frame_width=1000)
    assert inner_box in kept
    assert left_box not in kept


def test_preprocess_blanks_excluded_side_columns() -> None:
    frame = np.zeros((120, 200), dtype=np.uint8)
    frame[:, :40] = 220
    frame[:, 40:] = 40
    prep = preprocess_image(frame, DetectionConfig(exclude_left_frac=0.18, exclude_right_frac=0.0))
    x0, _ = horizontal_roi_bounds(200, DetectionConfig(exclude_left_frac=0.18))
    assert prep.gray[:, :x0].max() <= 45
    assert prep.work[:, :x0].max() <= 45


def test_masks_ignore_excluded_columns() -> None:
    work = np.full((100, 200), 40, dtype=np.uint8)
    work[30:70, 10:60] = 180
    cfg = DetectionConfig(exclude_left_frac=0.18, exclude_right_frac=0.0)
    masks = build_candidate_masks(work, cfg, gray=work)
    x0, _ = horizontal_roi_bounds(200, cfg)
    assert not masks.combined[:, :x0].any()
