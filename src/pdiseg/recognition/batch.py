"""Batch recognition over T1 segment crops: scoring, image-level aggregation, sweep."""

from __future__ import annotations

import csv
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pdiseg.io.dataset import load_image

from .classify import UNKNOWN, Template
from .config import RecognitionConfig
from .features import extract_features
from .matching import match_fraction

SEGMENT_SUFFIX_RE = re.compile(r"_(?:segmented|segmentada)_\d+$")


@dataclass(frozen=True)
class SegmentScores:
    """Match fractions of one segment crop against every template.

    ``class_name`` is the dataset folder and is used for evaluation only —
    predictions never look at it.
    """

    class_name: str
    source_stem: str
    scores: dict[str, float]


@dataclass(frozen=True)
class ImageBest:
    """Per-source-image aggregation: best score per template across its segments."""

    class_name: str
    source_stem: str
    scores: dict[str, float]


@dataclass(frozen=True)
class ImagePrediction:
    class_name: str
    source_stem: str
    predicted: str
    score: float
    true_score: float

    @property
    def correct(self) -> bool:
        return self.predicted == self.class_name


@dataclass(frozen=True)
class RecognitionSummary:
    images: int
    correct: int
    unknown: int
    false_positives: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.images if self.images else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.images if self.images else 0.0


def find_segment_crops(segments_root: str | Path) -> list[Path]:
    """Return sorted ``<Class>/<stem>_segment{ed,ada}_<N>.png`` crops (one level deep)."""
    root = Path(segments_root)
    return sorted(
        p for p in root.glob("*/*.png") if p.is_file() and SEGMENT_SUFFIX_RE.search(p.stem)
    )


def score_segment(
    path: Path, templates: list[Template], config: RecognitionConfig
) -> SegmentScores:
    features = extract_features(load_image(path), config)
    if features is None:
        scores = {template.name: 0.0 for template in templates}
    else:
        scores = {
            template.name: match_fraction(template.features, features, config)
            for template in templates
        }
    return SegmentScores(
        class_name=path.parent.name,
        source_stem=SEGMENT_SUFFIX_RE.sub("", path.stem),
        scores=scores,
    )


def score_dataset(
    segments_root: str | Path,
    templates: list[Template],
    config: RecognitionConfig,
    *,
    limit: int | None = None,
    workers: int = 1,
    progress_every: int = 0,
) -> list[SegmentScores]:
    """Score every segment crop against every template (thread pool like calibrate)."""
    crops = find_segment_crops(segments_root)
    if limit is not None:
        crops = crops[: max(0, limit)]
    total = len(crops)
    results: list[SegmentScores] = []

    def record(index: int, scored: SegmentScores) -> None:
        results.append(scored)
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(
                f"pdiseg-recognize: scored {index}/{total} segments",
                file=sys.stderr,
                flush=True,
            )

    worker_count = max(1, min(int(workers), total)) if total > 1 else 1
    if worker_count == 1:
        for index, crop_path in enumerate(crops, start=1):
            record(index, score_segment(crop_path, templates, config))
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: list[Future[SegmentScores]] = [
                executor.submit(score_segment, crop_path, templates, config) for crop_path in crops
            ]
            for future in as_completed(futures):
                completed += 1
                record(completed, future.result())
    return results


def aggregate_by_image(segment_scores: list[SegmentScores]) -> list[ImageBest]:
    """Collapse multiple segments of one source image into per-template max scores."""
    merged: dict[tuple[str, str], dict[str, float]] = {}
    for scored in segment_scores:
        key = (scored.class_name, scored.source_stem)
        acc = merged.setdefault(key, dict.fromkeys(scored.scores, 0.0))
        for name, value in scored.scores.items():
            acc[name] = max(acc.get(name, 0.0), value)
    return [
        ImageBest(class_name=class_name, source_stem=stem, scores=scores)
        for (class_name, stem), scores in sorted(merged.items())
    ]


def predict_images(aggregated: list[ImageBest], config: RecognitionConfig) -> list[ImagePrediction]:
    """Best template wins if it clears ``min_match_frac``; otherwise unknown."""
    predictions: list[ImagePrediction] = []
    for image in aggregated:
        best = max(image.scores, key=lambda name: image.scores[name])
        best_score = image.scores[best]
        label = best if best_score >= config.min_match_frac else UNKNOWN
        predictions.append(
            ImagePrediction(
                class_name=image.class_name,
                source_stem=image.source_stem,
                predicted=label,
                score=best_score,
                true_score=image.scores.get(image.class_name, 0.0),
            )
        )
    return predictions


def summarize(predictions: list[ImagePrediction]) -> RecognitionSummary:
    correct = sum(1 for p in predictions if p.correct)
    unknown = sum(1 for p in predictions if p.predicted == UNKNOWN)
    false_positives = sum(1 for p in predictions if p.predicted != UNKNOWN and not p.correct)
    return RecognitionSummary(
        images=len(predictions),
        correct=correct,
        unknown=unknown,
        false_positives=false_positives,
    )


def sweep_thresholds(
    aggregated: list[ImageBest],
    config: RecognitionConfig,
    thresholds: list[float] | None = None,
) -> list[tuple[float, RecognitionSummary]]:
    """Re-apply ``min_match_frac`` candidates over cached scores (cheap sweep)."""
    if thresholds is None:
        thresholds = [i / 100 for i in range(51)]
    rows: list[tuple[float, RecognitionSummary]] = []
    for threshold in thresholds:
        cfg = RecognitionConfig(
            descriptor=config.descriptor,
            max_ratio=config.max_ratio,
            cross_check=config.cross_check,
            min_match_frac=threshold,
        )
        rows.append((threshold, summarize(predict_images(aggregated, cfg))))
    return rows


def write_predictions_csv(path: str | Path, predictions: list[ImagePrediction]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "source", "predicted", "score", "true_score", "correct"])
        for p in predictions:
            writer.writerow(
                [
                    p.class_name,
                    p.source_stem,
                    p.predicted,
                    f"{p.score:.4f}",
                    f"{p.true_score:.4f}",
                    int(p.correct),
                ]
            )


def write_sweep_csv(path: str | Path, rows: list[tuple[float, RecognitionSummary]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "min_match_frac",
                "images",
                "correct",
                "unknown",
                "false_positives",
                "accuracy",
                "false_positive_rate",
            ]
        )
        for threshold, s in rows:
            writer.writerow(
                [
                    f"{threshold:.2f}",
                    s.images,
                    s.correct,
                    s.unknown,
                    s.false_positives,
                    f"{s.accuracy:.4f}",
                    f"{s.false_positive_rate:.4f}",
                ]
            )
