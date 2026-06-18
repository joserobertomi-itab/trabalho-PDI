from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .imaging import FrameInspection, boxes_to_json, render_overlay
from .pipeline import (
    detect_clusters,
    find_source_images,
    keep_label_clusters,
    preprocess,
    refine_to_name_label,
)


def inspect_frame(image: NDArray[np.uint8]) -> FrameInspection:

    working = preprocess(image)
    candidates = detect_clusters(working)
    kept = keep_label_clusters(candidates)
    labels = [refine_to_name_label(working, cluster) for cluster in kept]
    return FrameInspection(candidates=candidates, kept=kept, labels=labels)


@dataclass
class ClassStats:
    class_name: str
    frames: int
    candidates: int
    kept: int
    labels: int


def calibrate(
    input_root: str | Path, output_dir: str | Path, per_class_limit: int = 3
) -> list[ClassStats]:

    import imageio.v3 as iio

    input_root = Path(input_root)
    output_dir = Path(output_dir)
    totals: dict[str, dict[str, int]] = {}
    written_per_class: dict[str, int] = {}
    boxes_by_source: dict[str, dict[str, list[list[int]]]] = {}

    for source in find_source_images(input_root):
        class_name = source.parent.name
        image = iio.imread(source)
        inspection = inspect_frame(image)
        rel_path = source.relative_to(input_root).as_posix()
        boxes_by_source[rel_path] = {
            "candidates": boxes_to_json(inspection.candidates),
            "kept": boxes_to_json(inspection.kept),
            "labels": boxes_to_json(inspection.labels),
        }

        acc = totals.setdefault(class_name, {"frames": 0, "candidates": 0, "kept": 0, "labels": 0})
        acc["frames"] += 1
        acc["candidates"] += len(inspection.candidates)
        acc["kept"] += len(inspection.kept)
        acc["labels"] += len(inspection.labels)

        if written_per_class.get(class_name, 0) < per_class_limit:
            dest = output_dir / class_name / f"{source.stem}_overlay.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(dest, render_overlay(image, inspection))
            written_per_class[class_name] = written_per_class.get(class_name, 0) + 1

    stats = [
        ClassStats(
            class_name=name,
            frames=acc["frames"],
            candidates=acc["candidates"],
            kept=acc["kept"],
            labels=acc["labels"],
        )
        for name, acc in sorted(totals.items())
    ]
    _write_stats_csv(output_dir / "stats.csv", stats)
    _write_boxes_json(output_dir / "boxes.json", boxes_by_source)
    return stats


def _write_stats_csv(path: Path, stats: list[ClassStats]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "frames", "candidates", "kept", "labels"])
        for s in stats:
            writer.writerow([s.class_name, s.frames, s.candidates, s.kept, s.labels])


def _write_boxes_json(path: Path, boxes_by_source: dict[str, dict[str, list[list[int]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(boxes_by_source, indent=2), encoding="utf-8")
