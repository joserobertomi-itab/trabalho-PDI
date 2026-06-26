"""Load calibration bundle and resolve frames for the review UI."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pdiseg.core.imaging import BBox, FrameInspection, inspection_from_json
from pdiseg.io.dataset import SEGMENTED_CROP_GLOB

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SEGMENTED_RE = re.compile(r"^(?P<stem>.+)_segmented_(?P<index>\d+)$")


@dataclass(frozen=True)
class ReviewBundle:
    dataset_root: Path
    calibration_root: Path
    result_root: Path | None = None
    boxes: dict[str, dict[str, list[list[int]]]] = field(default_factory=dict)
    stats_by_class: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassSummary:
    class_name: str
    frames: int
    candidates: int
    kept: int
    labels: int


@dataclass(frozen=True)
class FrameReview:
    class_name: str
    stem: str
    rel_path: str
    source_exists: bool
    boxes: FrameInspection | None
    crop_paths: tuple[Path, ...]
    rejected_count: int
    candidate_count: int
    kept_count: int
    label_count: int


@dataclass(frozen=True)
class TotalsSummary:
    frames: int
    candidates: int
    kept: int
    labels: int


def load_bundle(
    dataset_root: str | Path,
    calibration_root: str | Path,
    result_root: str | Path | None = None,
) -> ReviewBundle:
    dataset_root = Path(dataset_root)
    calibration_root = Path(calibration_root)
    result = Path(result_root) if result_root is not None else None

    boxes: dict[str, dict[str, list[list[int]]]] = {}
    boxes_path = calibration_root / "boxes.json"
    if boxes_path.is_file():
        raw = json.loads(boxes_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            boxes = raw

    stats_by_class = _load_stats(calibration_root / "stats.csv")
    return ReviewBundle(
        dataset_root=dataset_root,
        calibration_root=calibration_root,
        result_root=result,
        boxes=boxes,
        stats_by_class=stats_by_class,
    )


def list_classes(bundle: ReviewBundle) -> list[ClassSummary]:
    classes = _discover_classes(bundle)
    summaries: list[ClassSummary] = []
    for class_name in classes:
        frames = list_frames(bundle, class_name)
        if class_name in bundle.stats_by_class:
            row = bundle.stats_by_class[class_name]
            summaries.append(
                ClassSummary(
                    class_name=class_name,
                    frames=row["frames"],
                    candidates=row["candidates"],
                    kept=row["kept"],
                    labels=row["labels"],
                )
            )
        else:
            summaries.append(
                ClassSummary(
                    class_name=class_name,
                    frames=len(frames),
                    candidates=sum(f.candidate_count for f in frames),
                    kept=sum(f.kept_count for f in frames),
                    labels=sum(f.label_count for f in frames),
                )
            )
    return summaries


def totals(bundle: ReviewBundle) -> TotalsSummary:
    classes = list_classes(bundle)
    return TotalsSummary(
        frames=sum(c.frames for c in classes),
        candidates=sum(c.candidates for c in classes),
        kept=sum(c.kept for c in classes),
        labels=sum(c.labels for c in classes),
    )


def list_frames(
    bundle: ReviewBundle,
    class_name: str,
    *,
    min_labels: int = 0,
    only_rejected: bool = False,
) -> list[FrameReview]:
    frames = [_build_frame(bundle, class_name, rel) for rel in _frame_keys(bundle, class_name)]
    if min_labels:
        frames = [frame for frame in frames if frame.label_count >= min_labels]
    if only_rejected:
        frames = [frame for frame in frames if frame.rejected_count > 0]
    return sorted(frames, key=lambda frame: frame.stem)


def get_frame(bundle: ReviewBundle, class_name: str, stem: str) -> FrameReview | None:
    for rel in sorted(_frame_keys(bundle, class_name)):
        if Path(rel).stem == stem:
            return _build_frame(bundle, class_name, rel)
    return None


def source_path(bundle: ReviewBundle, class_name: str, stem: str) -> Path | None:
    frame = get_frame(bundle, class_name, stem)
    if frame is None:
        return None
    path = bundle.dataset_root / frame.rel_path
    return path if path.is_file() else None


def _discover_classes(bundle: ReviewBundle) -> list[str]:
    names: set[str] = set()
    names.update(key.split("/", 1)[0] for key in bundle.boxes if "/" in key)
    names.update(bundle.stats_by_class.keys())
    if bundle.dataset_root.is_dir():
        names.update(
            p.name
            for p in bundle.dataset_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    if bundle.result_root and bundle.result_root.is_dir():
        names.update(
            p.name
            for p in bundle.result_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    return sorted(names)


def _frame_keys(bundle: ReviewBundle, class_name: str) -> set[str]:
    keys: set[str] = {key for key in bundle.boxes if key.startswith(f"{class_name}/")}
    class_dataset = bundle.dataset_root / class_name
    if class_dataset.is_dir():
        for path in class_dataset.iterdir():
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                keys.add(f"{class_name}/{path.name}")
    if bundle.result_root is None:
        return keys
    class_result = bundle.result_root / class_name
    if not class_result.is_dir():
        return keys
    for path in class_result.glob(SEGMENTED_CROP_GLOB):
        match = _SEGMENTED_RE.match(path.stem)
        if match is None:
            continue
        stem = match.group("stem")
        if any(Path(key).stem == stem for key in keys):
            continue
        if class_dataset.is_dir():
            for source in class_dataset.glob(f"{stem}.*"):
                if source.suffix.lower() in _IMAGE_SUFFIXES:
                    keys.add(f"{class_name}/{source.name}")
        else:
            keys.add(f"{class_name}/{stem}.png")
    return keys


def _build_frame(bundle: ReviewBundle, class_name: str, rel_path: str) -> FrameReview:
    stem = Path(rel_path).stem
    raw = bundle.boxes.get(rel_path)
    inspection = inspection_from_json(raw) if raw is not None else None
    crop_paths = _crop_paths(bundle, class_name, stem)
    candidate_count = len(inspection.candidates) if inspection else 0
    kept_count = len(inspection.kept) if inspection else 0
    label_count = len(inspection.labels) if inspection else len(crop_paths)
    rejected_count = 0
    if inspection:
        kept_set = set(inspection.kept)
        rejected_count = sum(1 for box in inspection.candidates if box not in kept_set)
    return FrameReview(
        class_name=class_name,
        stem=stem,
        rel_path=rel_path,
        source_exists=_source_exists(bundle, rel_path),
        boxes=inspection,
        crop_paths=tuple(crop_paths),
        rejected_count=rejected_count,
        candidate_count=candidate_count,
        kept_count=kept_count,
        label_count=label_count,
    )


def _source_exists(bundle: ReviewBundle, rel_path: str) -> bool:
    return (bundle.dataset_root / rel_path).is_file()


def _crop_paths(bundle: ReviewBundle, class_name: str, stem: str) -> list[Path]:
    if bundle.result_root is None:
        return []
    class_dir = bundle.result_root / class_name
    if not class_dir.is_dir():
        return []

    crops: list[tuple[int, Path]] = []
    for path in class_dir.iterdir():
        if not path.is_file():
            continue
        match = _SEGMENTED_RE.match(path.stem)
        if match and match.group("stem") == stem:
            crops.append((int(match.group("index")), path))
    return [path for _, path in sorted(crops)]


def label_box(frame: FrameReview, index: int) -> BBox | None:
    if frame.boxes is None or index < 1 or index > len(frame.boxes.labels):
        return None
    return frame.boxes.labels[index - 1]


def _load_stats(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        return {}
    by_class: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = row["class_name"]
            by_class[class_name] = {
                "frames": int(row["frames"]),
                "candidates": int(row["candidates"]),
                "kept": int(row["kept"]),
                "labels": int(row["labels"]),
            }
    return by_class
