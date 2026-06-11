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


# Fixed top-left region covering the burned-in "FPS: NN.NN" overlay (see CONTEXT.md).
# Calibrated against the base; (x, y, width, height) in pixels.
# NOTE (calibration #6): masking this region to 0 leaves a hard edge against bright
# neighbours, which Stage 1 can pick up as a spurious cluster at the top-left on
# low-content frames. Consider masking to a neutral value, or ignoring detections
# that hug this region, when calibrating false-positive rejection.
FPS_OVERLAY_REGION: BBox = (0, 0, 215, 48)


def detect(image: np.ndarray) -> list[BBox]:
    """Trivial placeholder: the whole frame as one box (replaced in later slices)."""
    height, width = image.shape[:2]
    return [(0, 0, width, height)]


# Stage-1 label-cluster detection parameters. Reasonable defaults; the final
# values are calibrated against the base in the calibration slice (issue #6).
_CLUSTER_LOCAL_WINDOW = 25  # local-mean window for the adaptive text threshold
_CLUSTER_TEXT_OFFSET = 15  # how much brighter than local background a stroke must be
_CLUSTER_CLOSE_STRUCTURE = np.ones((7, 15))  # merge text strokes into solid blocks
_CLUSTER_MIN_AREA = 800  # drop blobs smaller than this many pixels


def detect_clusters(image: np.ndarray) -> list[BBox]:
    """Locate label-cluster candidates via text density (docs/adr/0001, Stage 1).

    Highlight locally-bright text, close it into solid blocks, label connected
    components, and return each surviving block's bounding box. No false-positive
    geometry filtering yet — that is Stage 3 (issue #4).
    """
    from scipy.ndimage import binary_closing, find_objects, label, uniform_filter

    img = image.astype(np.float32)
    local = uniform_filter(img, size=_CLUSTER_LOCAL_WINDOW)
    text = img > local + _CLUSTER_TEXT_OFFSET
    merged = binary_closing(text, structure=_CLUSTER_CLOSE_STRUCTURE)

    # Watershed split (distance transform + watershed) is intentionally omitted:
    # on sample frames clusters come out separated, not merged, so the split adds
    # no value here. Revisit during calibration (issue #6) if merging appears.
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


# Stage-3 geometric false-positive rejection (issue #4). Calibrated against the
# base's Stage-1 candidates; final values are locked in the calibration slice (#6).
_LABEL_MIN_AREA = 3000  # drop tiny text fragments
_LABEL_MAX_AREA = 150000  # drop the SSA box / full-height background merges
_LABEL_MAX_ELONGATION = 4.0  # drop barcodes / thin edges (rotation-invariant)


def keep_label_clusters(candidates: list[BBox]) -> list[BBox]:
    """Reject false positives by geometry (docs/adr/0001, Stage 3).

    Keep only candidates whose area is label-sized and whose shape is not extremely
    elongated. Elongation is rotation-invariant, so labels at 0deg or 90deg both
    pass while barcodes and thin edges are dropped.
    """
    kept: list[BBox] = []
    for x, y, w, h in candidates:
        area = w * h
        elongation = max(w, h) / min(w, h)
        if _LABEL_MIN_AREA <= area <= _LABEL_MAX_AREA and elongation <= _LABEL_MAX_ELONGATION:
            kept.append((x, y, w, h))
    return kept


# Stage-2 refinement (issue #5). The dark name label should occupy a sane fraction
# of its cluster; outside this band the refinement is ambiguous and we fall back.
_REFINE_MIN_FRACTION = 0.05
_REFINE_MAX_FRACTION = 0.9


def refine_to_name_label(image: np.ndarray, cluster_bbox: BBox) -> BBox:
    """Shrink a cluster to its dark name-label box (docs/adr/0001, Stage 2).

    Inside the cluster the bright brand badge and the dark name label are the two
    dominant regions, so an Otsu threshold separates them; the largest dark
    component is the name label. Falls back to the cluster box when the split is
    ambiguous (uniform region, or a dark component that is implausibly small or
    fills almost the whole cluster).
    """
    from scipy.ndimage import find_objects, label
    from skimage.filters import threshold_otsu

    x, y, w, h = cluster_bbox
    region = image[y : y + h, x : x + w]
    if region.size == 0 or region.max() == region.min():
        return cluster_bbox

    dark = region <= threshold_otsu(region)
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


def preprocess(image: np.ndarray) -> np.ndarray:
    """Prepare a frame for detection: equalize contrast, then mask the FPS overlay."""
    from scipy.ndimage import median_filter
    from skimage.exposure import equalize_hist
    from skimage.util import img_as_ubyte

    working = median_filter(image, size=3)
    working = img_as_ubyte(equalize_hist(working))
    x, y, w, h = FPS_OVERLAY_REGION
    working[y : y + h, x : x + w] = 0
    return working


def dump_preprocessed(input_root, output_dir, limit: int = 5) -> list[Path]:
    """Write the preprocessed version of the first ``limit`` frames for inspection."""
    import imageio.v3 as iio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in find_source_images(input_root)[:limit]:
        dest = output_dir / f"{source.stem}_preprocessed.png"
        iio.imwrite(dest, preprocess(iio.imread(source)))
        written.append(dest)
    return written


def crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w]


def output_path(output_root, class_name: str, source_path, index: int) -> Path:
    """``<output_root>/<class_name>/<source-stem>_segmentada_<index>.png``."""
    stem = Path(source_path).stem
    return Path(output_root) / class_name / f"{stem}_segmentada_{index}.png"


def detect_name_labels(image: np.ndarray) -> list[BBox]:
    """The two-stage detector (docs/adr/0001): a raw frame to its name labels.

    Self-contained: preprocesses the frame, locates label clusters (Stage 1),
    rejects false positives by geometry (Stage 3), and refines each surviving
    cluster to its dark name label (Stage 2), with cluster fallback. The returned
    boxes index into the *original* frame, so callers crop from the source.

    The individual stages remain public so the calibration harness (issue #6) can
    inspect each stage's output; this is the composed, primary interface.
    """
    working = preprocess(image)
    clusters = keep_label_clusters(detect_clusters(working))
    return [refine_to_name_label(working, cluster) for cluster in clusters]


def run(input_root, output_root, detector=detect_name_labels) -> RunSummary:
    """Walk the dataset, detect name labels, and write one PNG crop per detection.

    ``detector`` receives each original frame and returns boxes into it; crops are
    taken from that original frame so the delivered crop stays faithful. The
    default is the real two-stage detector; tests inject a simpler detector to
    exercise orchestration on its own.
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
