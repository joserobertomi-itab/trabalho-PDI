"""Walking-skeleton pipeline for poultry-packaging segmentation.

The public interface is intentionally small. ``run`` is the deep orchestrator;
``detect`` is the seam that later slices replace with the real two-stage detector
(see docs/adr/0001) without changing the I/O contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A bounding box in pixel coordinates: (x, y, width, height).
BBox = tuple[int, int, int, int]


@dataclass
class RunSummary:
    """Outcome of a full dataset walk."""

    images_processed: int
    crops_written: int


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_source_images(input_root) -> list[Path]:
    """Every image file under ``input_root/<Class>/``, sorted for determinism."""
    root = Path(input_root)
    images = [
        p
        for p in root.glob("*/*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return sorted(images)


def detect(image: np.ndarray) -> list[BBox]:
    """Trivial placeholder: the whole frame as one box (replaced in later slices)."""
    height, width = image.shape[:2]
    return [(0, 0, width, height)]


def crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w]


def output_path(output_root, class_name: str, source_path, index: int) -> Path:
    """``<output_root>/<class_name>/<source-stem>_segmentada_<index>.png``."""
    stem = Path(source_path).stem
    return Path(output_root) / class_name / f"{stem}_segmentada_{index}.png"


def run(input_root, output_root, detector=detect) -> RunSummary:
    """Walk the dataset, detect, and write one PNG crop per detection.

    ``detector`` is the seam later slices use to plug in the real two-stage
    detector (docs/adr/0001); it defaults to the trivial whole-frame placeholder.
    """
    import imageio.v3 as iio

    images_processed = 0
    crops_written = 0
    for source in find_source_images(input_root):
        class_name = source.parent.name
        image = iio.imread(source)
        for index, bbox in enumerate(detector(image), start=1):
            dest = output_path(output_root, class_name, source, index)
            dest.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(dest, crop(image, bbox))
            crops_written += 1
        images_processed += 1
    return RunSummary(images_processed=images_processed, crops_written=crops_written)
