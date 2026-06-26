"""High-level detection API: inspect and return final label boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox, FrameInspection

from .candidates import find_candidate_boxes
from .config import DetectionConfig
from .postprocess import keep_label_clusters, postprocess_boxes
from .preprocess import preprocess_image
from .scoring import ScoredCandidate


@dataclass(frozen=True)
class DetectionResult:
    labels: list[BBox]
    candidates: list[BBox]
    kept: list[BBox]
    scored: list[ScoredCandidate]
    work: NDArray[np.uint8]


def detect(image: NDArray[np.uint8]) -> list[BBox]:
    height, width = image.shape[:2]
    return [(0, 0, width, height)]


def detect_name_labels(
    image: NDArray[np.uint8], config: DetectionConfig | None = None
) -> list[BBox]:
    """Return final name-label bounding boxes for one frame."""
    return inspect_detection(image, config).labels


def inspect_detection(
    image: NDArray[np.uint8], config: DetectionConfig | None = None
) -> DetectionResult:
    cfg = config or DetectionConfig()
    prep = preprocess_image(image, cfg)
    raw = find_candidate_boxes(prep.work, cfg, text_source=prep.gray)
    geometry = keep_label_clusters(raw, cfg, frame_width=image.shape[1])
    labels, scored, kept = postprocess_boxes(image, prep.work, geometry or raw, cfg)
    return DetectionResult(
        labels=labels,
        candidates=raw,
        kept=kept,
        scored=scored,
        work=prep.work,
    )


def inspect_frame(
    image: NDArray[np.uint8], config: DetectionConfig | None = None
) -> FrameInspection:
    result = inspect_detection(image, config)
    return FrameInspection(
        candidates=result.candidates,
        kept=result.kept,
        labels=result.labels,
    )
