"""Classify one segment against the template set; below threshold → unknown."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pdiseg.io.dataset import load_image

from .config import RecognitionConfig
from .features import Features, extract_features
from .matching import match_fraction

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Template:
    name: str
    features: Features


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)


def load_templates(
    templates_root: str | Path, config: RecognitionConfig | None = None
) -> list[Template]:
    """Load ``<templates_root>/<Class>.png`` templates and extract features once."""
    root = Path(templates_root)
    templates: list[Template] = []
    for path in sorted(root.glob("*.png")):
        features = extract_features(load_image(path), config)
        if features is None:
            raise ValueError(f"template {path} yields too few keypoints; re-crop it")
        templates.append(Template(name=path.stem, features=features))
    if not templates:
        raise ValueError(f"no *.png templates found under {root}")
    return templates


def classify_features(
    segment_features: Features | None,
    templates: list[Template],
    config: RecognitionConfig | None = None,
) -> Prediction:
    """Best-scoring template wins, but only above ``min_match_frac``; else unknown."""
    cfg = config or RecognitionConfig()
    if segment_features is None:
        return Prediction(label=UNKNOWN, score=0.0)
    scores = {
        template.name: match_fraction(template.features, segment_features, cfg)
        for template in templates
    }
    best = max(scores, key=lambda name: scores[name])
    if scores[best] < cfg.min_match_frac:
        return Prediction(label=UNKNOWN, score=scores[best], scores=scores)
    return Prediction(label=best, score=scores[best], scores=scores)


def classify_segment(
    image: NDArray[np.uint8],
    templates: list[Template],
    config: RecognitionConfig | None = None,
) -> Prediction:
    return classify_features(extract_features(image, config), templates, config)
