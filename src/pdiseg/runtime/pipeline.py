"""Batch dataset runner: detect, crop, write ``result/`` tree."""

from __future__ import annotations

import sys
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox, crop
from pdiseg.detection.detector import detect_name_labels
from pdiseg.io.dataset import find_source_images, load_image, segmented_crop_filename


@dataclass
class RunSummary:
    images_processed: int
    crops_written: int


@dataclass
class ClassReport:
    class_name: str
    frames: int
    crops: int
    empty_frames: int


@dataclass
class DatasetReport:
    classes: list[ClassReport]
    total_frames: int
    total_crops: int
    empty_frames: int

    @property
    def avg_crops_per_frame(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.total_crops / self.total_frames


@dataclass(frozen=True)
class _ProcessedSource:
    class_name: str
    crops: int


def output_path(
    output_root: str | Path, class_name: str, source_path: str | Path, index: int
) -> Path:
    stem = Path(source_path).stem
    return Path(output_root) / class_name / segmented_crop_filename(stem, index)


def crop_and_save(
    image: NDArray[np.uint8],
    boxes: list[BBox],
    output_root: str | Path,
    class_name: str,
    source_path: str | Path,
) -> int:
    written = 0
    for index, bbox in enumerate(boxes, start=1):
        dest = output_path(output_root, class_name, source_path, index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        patch = crop(image, bbox)
        if patch.size == 0:
            continue
        iio.imwrite(dest, patch)
        written += 1
    return written


def process_dataset(
    input_root: str | Path,
    output_root: str | Path,
    detector: Callable[[NDArray[np.uint8]], list[BBox]] = detect_name_labels,
    *,
    limit: int | None = None,
    offset: int = 0,
    progress_every: int = 0,
    workers: int = 1,
) -> DatasetReport:
    per_class: dict[str, ClassReport] = {}
    sources = find_source_images(input_root)[max(0, offset) :]
    if limit is not None:
        sources = sources[: max(0, limit)]
    total = len(sources)

    def process_source(source: Path) -> _ProcessedSource:
        class_name = source.parent.name
        image = load_image(source)
        boxes = detector(image)
        written = crop_and_save(image, boxes, output_root, class_name, source)
        return _ProcessedSource(class_name=class_name, crops=written)

    def record(index: int, result: _ProcessedSource) -> None:
        row = per_class.setdefault(
            result.class_name,
            ClassReport(class_name=result.class_name, frames=0, crops=0, empty_frames=0),
        )
        row.frames += 1
        row.crops += result.crops
        if result.crops == 0:
            row.empty_frames += 1
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(
                f"pdiseg: processed {index}/{total} images, crops={sum(r.crops for r in per_class.values())}",
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
            futures: list[Future[_ProcessedSource]] = [
                executor.submit(process_source, source) for source in sources
            ]
            for future in as_completed(futures):
                completed += 1
                record(completed, future.result())

    classes = [per_class[name] for name in sorted(per_class)]
    return DatasetReport(
        classes=classes,
        total_frames=sum(row.frames for row in classes),
        total_crops=sum(row.crops for row in classes),
        empty_frames=sum(row.empty_frames for row in classes),
    )


def _effective_workers(workers: int, total: int) -> int:
    if total <= 1:
        return 1
    return max(1, min(int(workers), total))


def run(
    input_root: str | Path,
    output_root: str | Path,
    detector: Callable[[NDArray[np.uint8]], list[BBox]] = detect_name_labels,
    *,
    limit: int | None = None,
    offset: int = 0,
    progress_every: int = 0,
    workers: int = 1,
) -> RunSummary:
    """Segment every image under ``input_root`` and write crops to ``output_root``."""
    report = process_dataset(
        input_root,
        output_root,
        detector=detector,
        limit=limit,
        offset=offset,
        progress_every=progress_every,
        workers=workers,
    )
    return RunSummary(images_processed=report.total_frames, crops_written=report.total_crops)


def dump_preprocessed(input_root: str | Path, output_dir: str | Path, limit: int = 5) -> list[Path]:
    from pdiseg.detection.preprocess import preprocess_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in find_source_images(input_root)[:limit]:
        dest = output_dir / f"{source.stem}_preprocessed.png"
        iio.imwrite(dest, preprocess_image(load_image(source)).work)
        written.append(dest)
    return written
