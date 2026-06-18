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


def text_mask(
    image: NDArray[np.uint8],
    *,
    window: int = _CLUSTER_LOCAL_WINDOW,
    offset: int = _CLUSTER_TEXT_OFFSET,
) -> NDArray[np.bool_]:
    """Stage-1 primitive (ADR 0001): pixels brighter than their local mean by ``offset``.

    A box filter estimates the local background; a stroke reads as text where it sits
    ``offset`` grey levels above it. Captures letter edges, not solid interiors.
    """
    from scipy.ndimage import uniform_filter

    img = image.astype(np.float32)
    local = uniform_filter(img, size=window)
    return cast(NDArray[np.bool_], img > local + offset)


def close_mask(
    mask: NDArray[np.bool_], *, structure: NDArray[np.float64] = _CLUSTER_CLOSE_STRUCTURE
) -> NDArray[np.bool_]:
    """Stage-1 primitive (ADR 0001/0004): close text strokes into solid blocks.

    The structure stays small (``7x15``) on purpose: a wider closing collapses a whole
    crate of packages into one blob (ADR 0004).
    """
    from scipy.ndimage import binary_closing

    return cast(NDArray[np.bool_], binary_closing(mask, structure=structure))


def label_components(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    """Stage-1 primitive: label connected components (background 0, each blob a + int).

    Returned as a raster so the debug notebook can render the blobs the grader sees.
    """
    from scipy.ndimage import label

    labels, _ = label(mask)
    return cast(NDArray[np.int32], labels)


def boxes_from_components(
    labels: NDArray[np.int32], *, min_area: int = _CLUSTER_MIN_AREA
) -> list[BBox]:
    """Stage-1 primitive: bounding box of every component with at least ``min_area`` px.

    Area is the component's true pixel count (its strokes), not its bounding box.
    """
    from scipy.ndimage import find_objects

    boxes: list[BBox] = []
    for index, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        area = int((labels[slc] == index).sum())
        if area < min_area:
            continue
        ys, xs = slc
        boxes.append((xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start))
    return boxes


def detect_clusters(image: NDArray[np.uint8]) -> list[BBox]:
    """Stage 1, text-density variant (ADR 0001): a recall net composing the
    text-density primitives (ADR 0005).

    Kept for contrast/inspection only. On glare-heavy frames it merges the whole
    crate into one blob (text fires on ~20% of pixels) — see ADR 0006. The graded
    Stage 1 is moving to ``detect_dark_badges``.
    """
    return boxes_from_components(label_components(close_mask(text_mask(image))))


# Stage-1 dark-badge variant (ADR 0006). The name label is bright text on a *dark*
# rounded-rectangle badge; the plastic glare is bright-on-bright. So darkness, not
# text density, discriminates the label from the glare. The threshold is a
# percentile, which is invariant to the monotonic ``equalize_hist`` in preprocess —
# so it runs on the existing preprocessed frame without a second preprocessing path.
_DARK_PERCENTILE = 20  # provisional; lock against the ground-truth boxes
_DARK_OPEN_STRUCTURE = np.ones((3, 3))  # drop glare specks before solidifying
_DARK_CLOSE_STRUCTURE = np.ones((9, 9))  # close the badge (bright text leaves holes)


def dark_mask(
    image: NDArray[np.uint8], *, percentile: float = _DARK_PERCENTILE
) -> NDArray[np.bool_]:
    """Stage-1 primitive (ADR 0006): the darkest ``percentile``% of pixels.

    A percentile threshold (not a fixed grey level) adapts to each frame and is
    invariant to the monotonic histogram equalisation in ``preprocess``.
    """
    threshold = np.percentile(image, percentile)
    return image < threshold


def open_mask(
    mask: NDArray[np.bool_], *, structure: NDArray[np.float64] = _DARK_OPEN_STRUCTURE
) -> NDArray[np.bool_]:
    """Stage-1 primitive (ADR 0006): morphological opening to drop tiny dark specks."""
    from scipy.ndimage import binary_opening

    return cast(NDArray[np.bool_], binary_opening(mask, structure=structure))


def detect_dark_badges(
    image: NDArray[np.uint8], *, percentile: float = _DARK_PERCENTILE
) -> list[BBox]:
    """Stage 1, dark-badge variant (ADR 0006): a recall net keyed on dark badges.

    Threshold the darkest pixels, open away glare specks, close each badge solid,
    then bound the components. Over-detection is expected; rejection is Stage 3's
    job. Composes the Stage-1 primitives (ADR 0005) so the notebook renders the
    same code the grader runs.
    """
    mask = dark_mask(image, percentile=percentile)
    mask = open_mask(mask)
    mask = close_mask(mask, structure=_DARK_CLOSE_STRUCTURE)
    return boxes_from_components(label_components(mask))


# Stage-1 dark-RELIEF variant (ADR 0007, revising ADR 0006). The global percentile
# in ``dark_mask`` assumes the badge is among the darkest pixels of the *whole* frame;
# a grey, mid-tone label fails that assumption and is missed. But the badge is always
# dark *relative to its local neighbourhood*. A black top-hat (grey closing minus the
# image) lights up dark structures smaller than the structuring element on a brighter
# local background, regardless of absolute grey level — so the absolute level stops
# mattering and only local darkness does.
_TOPHAT_SIZE = 51  # structuring element side; must exceed the badge's larger dimension
_RELIEF_PERCENTILE = 20  # provisional; keep the top P% of relief (most dark-relative px)


def dark_relief(image: NDArray[np.uint8], *, size: int = _TOPHAT_SIZE) -> NDArray[np.uint8]:
    """Stage-1 primitive (ADR 0007): black top-hat — how much darker each pixel is
    than its local ``size``-wide neighbourhood.

    ``grey_closing`` fills dark structures up to the background level; subtracting the
    image leaves a positive bump exactly where a dark structure sat on a brighter
    surround, and ~0 on flat or bright regions. Unlike ``dark_mask`` this is keyed on
    *local* contrast, so a mid-grey badge lights up even when it is not globally dark.
    """
    from scipy.ndimage import grey_closing

    closed = grey_closing(image, size=(size, size))
    relief = closed.astype(np.int16) - image.astype(np.int16)
    return cast(NDArray[np.uint8], relief.clip(0, 255).astype(np.uint8))


def relief_mask(
    relief: NDArray[np.uint8], *, percentile: float = _RELIEF_PERCENTILE
) -> NDArray[np.bool_]:
    """Stage-1 primitive (ADR 0007): keep the top ``percentile``% of the relief.

    The threshold is taken at the ``100 - percentile`` quantile, so ``percentile=20``
    keeps the 20% most dark-relative pixels — mirroring ``dark_mask``'s "keep the most
    badge-like P%" semantics. Flat regions (relief 0) never fire: the test is strict
    ``>``, so a uniform frame yields an empty mask.
    """
    threshold = np.percentile(relief, 100 - percentile)
    return cast(NDArray[np.bool_], relief > threshold)


def detect_dark_relief_badges(
    image: NDArray[np.uint8],
    *,
    size: int = _TOPHAT_SIZE,
    percentile: float = _RELIEF_PERCENTILE,
) -> list[BBox]:
    """Stage 1, dark-relief variant (ADR 0007): a recall net keyed on *local* darkness.

    Black top-hat to surface locally-dark structures, threshold the strongest relief,
    open away specks, close each badge solid, then bound the components. Over-detection
    is expected; rejection is Stage 3's job. Composes the Stage-1 primitives (ADR 0005),
    reusing ``open_mask``/``close_mask``/``label_components``/``boxes_from_components``
    so the notebook renders the same code the grader runs.
    """
    mask = relief_mask(dark_relief(image, size=size), percentile=percentile)
    mask = open_mask(mask)
    mask = close_mask(mask, structure=_DARK_CLOSE_STRUCTURE)
    return boxes_from_components(label_components(mask))


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


def otsu_dark_mask(region: NDArray[np.uint8]) -> NDArray[np.bool_] | None:
    """Stage-2 primitive (ADR 0001): dark pixels of a cluster crop via Otsu.

    Inside a cluster the bright brand badge and the dark name label are the two
    dominant regions, so Otsu separates them. Returns ``None`` when the region is
    empty or uniform (no threshold to find).
    """
    from skimage.filters import threshold_otsu

    if region.size == 0 or region.max() == region.min():
        return None
    threshold = threshold_otsu(region)  # type: ignore[no-untyped-call]
    return cast(NDArray[np.bool_], region <= threshold)


def largest_component_box(mask: NDArray[np.bool_]) -> BBox | None:
    """Stage-2 primitive: bbox of the largest connected component, in mask coords.

    Returns ``None`` when the mask has no components. Largest is by true pixel count.
    """
    from scipy.ndimage import find_objects, label

    labels, count = label(mask)
    if count == 0:
        return None
    best_slice = None
    best_area = 0
    for index, slc in enumerate(find_objects(labels), start=1):
        if slc is None:
            continue
        area = int((labels[slc] == index).sum())
        if area > best_area:
            best_area, best_slice = area, slc
    if best_slice is None:
        return None
    ys, xs = best_slice
    return (xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)


def refine_to_name_label(image: NDArray[np.uint8], cluster_bbox: BBox) -> BBox:
    """Stage 2 (ADR 0001): shrink a cluster to its dark name label, composing the
    Stage-2 primitives (ADR 0005).

    Falls back to the whole cluster (still a valid segmentation, ADR 0001) when the
    split is ambiguous: no dark region, or a dark component implausibly small (<5%)
    or filling almost the whole cluster (>90%).
    """
    x, y, w, h = cluster_bbox
    region = image[y : y + h, x : x + w]
    mask = otsu_dark_mask(region)
    if mask is None:
        return cluster_bbox
    local = largest_component_box(mask)
    if local is None:
        return cluster_bbox

    lx, ly, rw, rh = local
    fraction = (rw * rh) / (w * h)
    if not (_REFINE_MIN_FRACTION <= fraction <= _REFINE_MAX_FRACTION):
        return cluster_bbox
    return (x + lx, y + ly, rw, rh)


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
