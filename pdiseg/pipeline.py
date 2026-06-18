from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .imaging import BBox, crop


@dataclass
class RunSummary:
    images_processed: int
    crops_written: int


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_source_images(input_root: str | Path) -> list[Path]:

    root = Path(input_root)
    images = [p for p in root.glob("*/*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]
    return sorted(images)


FPS_OVERLAY_REGION: BBox = (0, 0, 215, 48)


def detect(image: NDArray[np.uint8]) -> list[BBox]:

    height, width = image.shape[:2]
    return [(0, 0, width, height)]


_CLUSTER_LOCAL_WINDOW = 25
_CLUSTER_TEXT_OFFSET = 15
_CLUSTER_CLOSE_STRUCTURE = np.ones((7, 15))
_CLUSTER_MIN_AREA = 800


def detect_clusters(image: NDArray[np.uint8]) -> list[BBox]:

    from scipy.ndimage import binary_closing, find_objects, label, uniform_filter

    img = image.astype(np.float32)
    local = uniform_filter(img, size=_CLUSTER_LOCAL_WINDOW)
    text = img > local + _CLUSTER_TEXT_OFFSET
    merged = binary_closing(text, structure=_CLUSTER_CLOSE_STRUCTURE)

    labels, _ = label(merged)
    boxes: list[BBox] = []
    for index, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        area = int((labels[slc] == index).sum())
        if area < _CLUSTER_MIN_AREA:
            continue
        ys, xs = slc
        boxes.append((xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start))
    return boxes


_LABEL_MIN_AREA = 3000
_LABEL_MAX_AREA = 150000
_LABEL_MAX_ELONGATION = 4.0


def keep_label_clusters(candidates: list[BBox]) -> list[BBox]:

    kept: list[BBox] = []
    for x, y, w, h in candidates:
        area = w * h
        elongation = max(w, h) / min(w, h)
        if _LABEL_MIN_AREA <= area <= _LABEL_MAX_AREA and elongation <= _LABEL_MAX_ELONGATION:
            kept.append((x, y, w, h))
    return kept


_REFINE_MIN_FRACTION = 0.05
_REFINE_MAX_FRACTION = 0.9


def refine_to_name_label(image: NDArray[np.uint8], cluster_bbox: BBox) -> BBox:

    from scipy.ndimage import find_objects, label
    from skimage.filters import threshold_otsu

    x, y, w, h = cluster_bbox
    region = image[y : y + h, x : x + w]
    if region.size == 0 or region.max() == region.min():
        return cluster_bbox

    dark = region <= threshold_otsu(region)  # type: ignore[no-untyped-call]
    labels, count = label(dark)
    if count == 0:
        return cluster_bbox

    best_slice = None
    best_area = 0
    for index, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        area = int((labels[slc] == index).sum())
        if area > best_area:
            best_area, best_slice = area, slc
    if best_slice is None:
        return cluster_bbox

    ys, xs = best_slice
    rw, rh = xs.stop - xs.start, ys.stop - ys.start
    fraction = (rw * rh) / (w * h)
    if not (_REFINE_MIN_FRACTION <= fraction <= _REFINE_MAX_FRACTION):
        return cluster_bbox
    return (x + xs.start, y + ys.start, rw, rh)


def preprocess(image: NDArray[np.uint8]) -> NDArray[np.uint8]:

    from scipy.ndimage import median_filter
    from skimage.exposure import equalize_hist
    from skimage.util import img_as_ubyte

    working = median_filter(image, size=3)
    working = cast(
        NDArray[np.uint8],
        img_as_ubyte(equalize_hist(working)),  # type: ignore[no-untyped-call]
    )

    x, y, w, h = FPS_OVERLAY_REGION
    working[y : y + h, x : x + w] = int(np.median(working))
    return working


def dump_preprocessed(input_root: str | Path, output_dir: str | Path, limit: int = 5) -> list[Path]:

    import imageio.v3 as iio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in find_source_images(input_root)[:limit]:
        dest = output_dir / f"{source.stem}_preprocessed.png"
        iio.imwrite(dest, preprocess(iio.imread(source)))
        written.append(dest)
    return written


def output_path(
    output_root: str | Path, class_name: str, source_path: str | Path, index: int
) -> Path:

    stem = Path(source_path).stem
    return Path(output_root) / class_name / f"{stem}_segmentada_{index}.png"


def detect_name_labels(image: NDArray[np.uint8]) -> list[BBox]:

    working = preprocess(image)
    clusters = keep_label_clusters(detect_clusters(working))
    return [refine_to_name_label(working, cluster) for cluster in clusters]


def run(
    input_root: str | Path,
    output_root: str | Path,
    detector: Callable[[NDArray[np.uint8]], list[BBox]] = detect_name_labels,
) -> RunSummary:

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
