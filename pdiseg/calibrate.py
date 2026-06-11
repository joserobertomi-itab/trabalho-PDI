"""Calibration harness (issue #6): run the pipeline over the base, expose the
per-stage detection breakdown, render overlays, and aggregate per-class stats so
a human can lock the fixed pixel-size thresholds (Stages 1-3, docs/adr/0001).

The harness reads folder names only to label and group its *reports*; the
detection algorithm never sees them (acceptance criterion of issue #6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pipeline import (
    BBox,
    detect_clusters,
    find_source_images,
    keep_label_clusters,
    preprocess,
    refine_to_name_label,
)


@dataclass
class FrameInspection:
    """Per-stage detection breakdown for one frame (calibration view).

    ``candidates`` are the raw Stage-1 clusters, ``kept`` are the Stage-3
    geometric survivors, and ``labels`` are the Stage-2 refined name labels (the
    production output). The funnel only narrows: candidates >= kept == labels.
    """

    candidates: list[BBox]
    kept: list[BBox]
    labels: list[BBox]


def inspect_frame(image: np.ndarray) -> FrameInspection:
    """Run the two-stage detector on a raw frame, keeping each stage's output.

    Mirrors ``detect_name_labels`` (same internal preprocessing, boxes index the
    original frame) but returns the intermediate Stage-1/Stage-3 boxes too, so the
    calibration overlays can show where false positives are rejected.
    """
    working = preprocess(image)
    candidates = detect_clusters(working)
    kept = keep_label_clusters(candidates)
    labels = [refine_to_name_label(working, cluster) for cluster in kept]
    return FrameInspection(candidates=candidates, kept=kept, labels=labels)


# Overlay colors per stage (RGB). Rejected candidates are drawn first so kept and
# final boxes paint over them where they overlap.
_REJECTED_COLOR = (220, 50, 50)  # red: Stage-1 candidates dropped by Stage 3
_KEPT_COLOR = (240, 200, 40)  # yellow: Stage-3 survivors (cluster framing)
_LABEL_COLOR = (40, 200, 80)  # green: Stage-2 refined name labels (the output)


def _draw_box(rgb: np.ndarray, bbox: BBox, color: tuple[int, int, int]) -> None:
    from skimage.draw import rectangle_perimeter

    x, y, w, h = bbox
    rr, cc = rectangle_perimeter(
        (y, x), extent=(h, w), shape=rgb.shape[:2], clip=True
    )
    rgb[rr, cc] = color


def render_overlay(image: np.ndarray, inspection: FrameInspection) -> np.ndarray:
    """Draw an inspection's per-stage boxes on an RGB copy of the frame.

    Red = Stage-1 candidates rejected by geometry, yellow = Stage-3 kept clusters,
    green = Stage-2 refined name labels (the delivered output). A human reads the
    overlay to judge detection vs false positives and lock the thresholds.
    """
    from skimage.color import gray2rgb
    from skimage.util import img_as_ubyte

    rgb = gray2rgb(img_as_ubyte(image)).copy()
    kept_set = set(inspection.kept)
    for bbox in inspection.candidates:
        if bbox not in kept_set:
            _draw_box(rgb, bbox, _REJECTED_COLOR)
    for bbox in inspection.kept:
        _draw_box(rgb, bbox, _KEPT_COLOR)
    for bbox in inspection.labels:
        _draw_box(rgb, bbox, _LABEL_COLOR)
    return rgb


@dataclass
class ClassStats:
    """Per-class aggregate of the detection funnel over the calibrated base.

    ``candidates``/``kept``/``labels`` sum each stage's box count across every
    frame in the class, so a human can spot classes that over- or under-detect.
    """

    class_name: str
    frames: int
    candidates: int
    kept: int
    labels: int


def calibrate(input_root, output_dir, per_class_limit: int = 3) -> list[ClassStats]:
    """Run the pipeline over the base, writing overlays and per-class stats.

    For each class folder under ``input_root`` every frame is inspected and its
    per-stage box counts are summed into a ``ClassStats``. Up to
    ``per_class_limit`` overlay PNGs per class are written under
    ``output_dir/<Class>/<stem>_overlay.png`` for human review. Folder names are
    used only to group and label reports, never as algorithm input.
    """
    import imageio.v3 as iio

    output_dir = Path(output_dir)
    totals: dict[str, dict[str, int]] = {}
    written_per_class: dict[str, int] = {}

    for source in find_source_images(input_root):
        class_name = source.parent.name
        image = iio.imread(source)
        inspection = inspect_frame(image)

        acc = totals.setdefault(
            class_name, {"frames": 0, "candidates": 0, "kept": 0, "labels": 0}
        )
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
    return stats


def _write_stats_csv(path: Path, stats: list[ClassStats]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "frames", "candidates", "kept", "labels"])
        for s in stats:
            writer.writerow([s.class_name, s.frames, s.candidates, s.kept, s.labels])
