"""Keypoint detection and descriptor extraction (skimage SIFT / ORB)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.feature import ORB, SIFT

from .config import RecognitionConfig

Descriptors = NDArray[Any]


@dataclass(frozen=True)
class Features:
    """Keypoint coordinates (row, col) plus their descriptors for one crop."""

    keypoints: NDArray[np.float64]
    descriptors: Descriptors

    def __len__(self) -> int:
        return int(self.descriptors.shape[0])


def extract_features(
    image: NDArray[np.uint8], config: RecognitionConfig | None = None
) -> Features | None:
    """Return keypoints + descriptors for one grayscale crop.

    Returns ``None`` when the image yields fewer than ``min_keypoints``
    keypoints (flat or degenerate crops) — callers treat that as "no evidence",
    never as a match.
    """
    cfg = config or RecognitionConfig()
    extractor: SIFT | ORB
    if cfg.descriptor == "orb":
        extractor = ORB(n_keypoints=cfg.orb_keypoints)  # type: ignore[no-untyped-call]
    elif cfg.descriptor == "sift":
        extractor = SIFT()  # type: ignore[no-untyped-call]
    else:
        raise ValueError(f"unknown descriptor {cfg.descriptor!r}")
    try:
        extractor.detect_and_extract(image)  # type: ignore[no-untyped-call]
    except (RuntimeError, ValueError):
        # skimage raises when no keypoints/octaves can be computed.
        return None
    descriptors = np.asarray(extractor.descriptors)
    if descriptors.ndim != 2 or descriptors.shape[0] < cfg.min_keypoints:
        return None
    return Features(
        keypoints=np.asarray(extractor.keypoints, dtype=np.float64),
        descriptors=descriptors,
    )


def extract_descriptors(
    image: NDArray[np.uint8], config: RecognitionConfig | None = None
) -> Descriptors | None:
    """Descriptor-only convenience wrapper around :func:`extract_features`."""
    features = extract_features(image, config)
    return None if features is None else features.descriptors
