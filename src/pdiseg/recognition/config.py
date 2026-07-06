"""Tunable recognition thresholds (local-descriptor matching, precision-first)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognitionConfig:
    """Matching knobs shared by all classes (no per-class tuning).

    ``min_match_frac`` is the single calibration knob required by the T2 brief:
    the fraction of template keypoints that must find a valid match in the
    segment for a classification to be emitted at all (otherwise ``unknown``).
    """

    descriptor: str = "sift"  # "sift" | "orb"
    max_ratio: float = 0.72
    cross_check: bool = True
    min_match_frac: float = 0.05
    min_keypoints: int = 8
    orb_keypoints: int = 500
    # Geometric verification: matches must agree on one affine transform.
    ransac_min_matches: int = 6
    ransac_residual_px: float = 8.0
