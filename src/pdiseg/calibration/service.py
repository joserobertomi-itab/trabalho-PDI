"""Export overlays, boxes.json, and per-class stats for review."""

from __future__ import annotations

import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pdiseg.core.imaging import boxes_to_json, render_overlay
from pdiseg.detection.detector import inspect_frame
from pdiseg.io.dataset import find_source_images, load_image


@dataclass
class ClassStats:
    class_name: str
    frames: int
    candidates: int
    kept: int
    labels: int


@dataclass(frozen=True)
class _CalibrationFrame:
    class_name: str
    rel_path: str
    candidates: list[list[int]]
    kept: list[list[int]]
    labels: list[list[int]]


def calibrate(
    input_root: str | Path,
    output_dir: str | Path,
    per_class_limit: int = 3,
    *,
    limit: int | None = None,
    offset: int = 0,
    progress_every: int = 0,
    workers: int = 1,
) -> list[ClassStats]:
    """Write sample overlays, ``boxes.json``, and ``stats.csv`` under ``output_dir``."""
    import imageio.v3 as iio

    input_root = Path(input_root)
    output_dir = Path(output_dir)
    totals: dict[str, dict[str, int]] = {}
    boxes_by_source: dict[str, dict[str, list[list[int]]]] = {}

    sources = find_source_images(input_root)[max(0, offset) :]
    if limit is not None:
        sources = sources[: max(0, limit)]
    total = len(sources)
    overlay_targets = _overlay_targets(sources, per_class_limit)

    def process_source(source: Path) -> _CalibrationFrame:
        class_name = source.parent.name
        image = load_image(source)
        inspection = inspect_frame(image)
        rel_path = source.relative_to(input_root).as_posix()
        candidates = boxes_to_json(inspection.candidates)
        kept = boxes_to_json(inspection.kept)
        labels = boxes_to_json(inspection.labels)

        if source in overlay_targets:
            dest = output_dir / class_name / f"{source.stem}_overlay.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(dest, render_overlay(image, inspection))
        return _CalibrationFrame(
            class_name=class_name,
            rel_path=rel_path,
            candidates=candidates,
            kept=kept,
            labels=labels,
        )

    def record(index: int, frame: _CalibrationFrame) -> None:
        boxes_by_source[frame.rel_path] = {
            "candidates": frame.candidates,
            "kept": frame.kept,
            "labels": frame.labels,
        }
        acc = totals.setdefault(
            frame.class_name, {"frames": 0, "candidates": 0, "kept": 0, "labels": 0}
        )
        acc["frames"] += 1
        acc["candidates"] += len(frame.candidates)
        acc["kept"] += len(frame.kept)
        acc["labels"] += len(frame.labels)
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(
                f"pdiseg-calibrate: processed {index}/{total} images",
                file=sys.stderr,
                flush=True,
            )

    worker_count = _effective_workers(workers, total)
    if worker_count == 1:
        for index, source in enumerate(sources, start=1):
            record(index, process_source(source))
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: list[Future[_CalibrationFrame]] = [
                executor.submit(process_source, source) for source in sources
            ]
            for future in as_completed(futures):
                completed += 1
                record(completed, future.result())

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
    _write_boxes_json(output_dir / "boxes.json", dict(sorted(boxes_by_source.items())))
    return stats


def _overlay_targets(sources: list[Path], per_class_limit: int) -> set[Path]:
    if per_class_limit <= 0:
        return set()
    targets: set[Path] = set()
    written_per_class: dict[str, int] = {}
    for source in sources:
        class_name = source.parent.name
        if written_per_class.get(class_name, 0) >= per_class_limit:
            continue
        targets.add(source)
        written_per_class[class_name] = written_per_class.get(class_name, 0) + 1
    return targets


def _effective_workers(workers: int, total: int) -> int:
    if total <= 1:
        return 1
    return max(1, min(int(workers), total))


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
