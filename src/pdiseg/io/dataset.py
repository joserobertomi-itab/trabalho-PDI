"""Walk the dataset tree and load grayscale frames."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FPS_OVERLAY_REGION: BBox = (0, 0, 215, 48)
SEGMENTED_CROP_GLOB = "*_segmented_*.png"


def segmented_crop_filename(stem: str, index: int) -> str:
    """Return ``{stem}_segmented_{index}.png`` (English crop naming under ``result/``)."""
    return f"{stem}_segmented_{index}.png"


def find_source_images(input_root: str | Path) -> list[Path]:
    """Return sorted ``<class>/<file>`` image paths under ``input_root`` (one level deep)."""
    root = Path(input_root)
    images = [p for p in root.glob("*/*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images)


def load_image(path: str | Path) -> NDArray[np.uint8]:
    """Load a single frame as 2-D ``uint8`` grayscale."""
    image = iio.imread(path)
    if image.ndim > 2:
        image = image[..., 0]
    return image.astype(np.uint8)


def list_classes(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def count_images_per_class(dataset_root: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in find_source_images(dataset_root):
        name = path.parent.name
        counts[name] = counts.get(name, 0) + 1
    return counts
