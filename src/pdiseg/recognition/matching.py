"""Descriptor matching: Lowe ratio + cross-check + RANSAC geometric verification."""

from __future__ import annotations

import numpy as np
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform

from .config import RecognitionConfig
from .features import Features


def match_fraction(
    template: Features,
    segment: Features,
    config: RecognitionConfig | None = None,
) -> float:
    """Return valid matches as a fraction of the template's keypoints (0.0 to 1.0).

    This is the T2 calibration metric: the percentage of template interest
    points with a valid correspondence in the segment. Valid means it survives
    the Lowe ratio test (``max_ratio``), the cross-check, and — decisive for
    false positives — agrees with one affine transform (RANSAC inlier):
    spurious matches on words shared by every label (brand, "CONGELADO", the
    weight table font) do not form a geometric consensus and are discarded.
    """
    cfg = config or RecognitionConfig()
    matches = match_descriptors(  # type: ignore[no-untyped-call]
        template.descriptors,
        segment.descriptors,
        cross_check=cfg.cross_check,
        max_ratio=cfg.max_ratio,
    )
    if len(matches) < cfg.ransac_min_matches:
        return 0.0
    src = template.keypoints[matches[:, 0]]
    dst = segment.keypoints[matches[:, 1]]
    try:
        _, inliers = ransac(  # type: ignore[no-untyped-call]
            (src, dst),
            AffineTransform,
            min_samples=3,
            residual_threshold=cfg.ransac_residual_px,
            max_trials=200,
            rng=0,
        )
    except ValueError:
        return 0.0
    if inliers is None:
        return 0.0
    return float(np.count_nonzero(inliers)) / float(len(template))
