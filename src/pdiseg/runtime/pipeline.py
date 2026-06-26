"""Batch dataset runner: detect, crop, write ``result/`` tree."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox, crop
from pdiseg.detection.detector import detect_name_labels
from pdiseg.io.dataset import find_source_images, load_image


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


def output_path(
    output_root: str | Path, class_name: str, source_path: str | Path, index: int
) -> Path:
    stem = Path(source_path).stem
    return Path(output_root) / class_name / f"{stem}_segmentada_{index}.png"


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
) -> DatasetReport:
    per_class: dict[str, ClassReport] = {}
    for source in find_source_images(input_root):
        class_name = source.parent.name
        image = load_image(source)
        boxes = detector(image)
        crop_and_save(image, boxes, output_root, class_name, source)
        row = per_class.setdefault(
            class_name,
            ClassReport(class_name=class_name, frames=0, crops=0, empty_frames=0),
        )
        row.frames += 1
        row.crops += len(boxes)
        if not boxes:
            row.empty_frames += 1
    classes = [per_class[name] for name in sorted(per_class)]
    return DatasetReport(
        classes=classes,
        total_frames=sum(row.frames for row in classes),
        total_crops=sum(row.crops for row in classes),
        empty_frames=sum(row.empty_frames for row in classes),
    )


def run(
    input_root: str | Path,
    output_root: str | Path,
    detector: Callable[[NDArray[np.uint8]], list[BBox]] = detect_name_labels,
) -> RunSummary:
    """Segment every image under ``input_root`` and write crops to ``output_root``."""
    report = process_dataset(input_root, output_root, detector=detector)
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
