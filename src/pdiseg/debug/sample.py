"""Run the production pipeline on a small per-class sample for notebook/CLI debug."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox
from pdiseg.debug.viz import FrameDebugSnapshot, analyze_frame, save_debug_bundle
from pdiseg.detection.config import DetectionConfig
from pdiseg.io.dataset import IMAGE_SUFFIXES, load_image
from pdiseg.runtime.pipeline import ClassReport, DatasetReport, crop_and_save


@dataclass(frozen=True)
class DebugFrameResult:
    class_name: str
    source: Path
    labels: list[BBox]
    bundle_dir: Path
    snapshot: FrameDebugSnapshot | None = None


@dataclass(frozen=True)
class DebugSampleReport:
    dataset_report: DatasetReport
    frames: list[DebugFrameResult]
    result_root: Path
    bundle_root: Path


@dataclass(frozen=True)
class DebugFrameView:
    """Notebook-friendly row: frame metadata + loaded image + pipeline snapshot."""

    item: DebugFrameResult
    image: NDArray[np.uint8]
    snapshot: FrameDebugSnapshot


def resolve_frame_snapshot(
    item: DebugFrameResult,
    image: NDArray[np.uint8] | None = None,
    config: DetectionConfig | None = None,
) -> FrameDebugSnapshot:
    """Return stored snapshot or re-analyze (handles stale notebook kernels)."""
    snapshot = getattr(item, "snapshot", None)
    if isinstance(snapshot, FrameDebugSnapshot):
        return snapshot
    frame = image if image is not None else load_image(item.source)
    return analyze_frame(frame, config)


def build_sample_views(
    report: DebugSampleReport,
    config: DetectionConfig | None = None,
) -> list[DebugFrameView]:
    """Build views for every frame in a debug sample report."""
    views: list[DebugFrameView] = []
    for item in report.frames:
        image = load_image(item.source)
        snapshot = resolve_frame_snapshot(item, image, config)
        views.append(DebugFrameView(item=item, image=image, snapshot=snapshot))
    return views


def select_sample_images(input_root: str | Path, per_class: int = 1) -> list[Path]:
    """Return up to ``per_class`` sorted images from each class folder under ``input_root``."""
    root = Path(input_root)
    if per_class <= 0:
        return []
    selected: list[Path] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        images = sorted(
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        selected.extend(images[:per_class])
    return selected


def run_debug_sample(
    input_root: str | Path,
    result_root: str | Path,
    *,
    bundle_root: str | Path | None = None,
    per_class: int = 1,
    config: DetectionConfig | None = None,
) -> DebugSampleReport:
    """Run detect + crop + debug bundles on a small sample (default: one frame per class)."""
    input_root = Path(input_root)
    result_root = Path(result_root)
    bundle_root = Path(bundle_root) if bundle_root is not None else result_root.parent / "bundles"

    per_class_stats: dict[str, ClassReport] = {}
    frames: list[DebugFrameResult] = []

    for source in select_sample_images(input_root, per_class):
        class_name = source.parent.name
        image = load_image(source)
        snapshot = analyze_frame(image, config)
        labels = snapshot.detection.labels
        crop_and_save(image, labels, result_root, class_name, source)
        title = f"{class_name} / {source.name}"
        bundle_dir = save_debug_bundle(
            image,
            bundle_root / class_name,
            source.stem,
            snapshot=snapshot,
            title=title,
        )

        row = per_class_stats.setdefault(
            class_name,
            ClassReport(class_name=class_name, frames=0, crops=0, empty_frames=0),
        )
        row.frames += 1
        row.crops += len(labels)
        if not labels:
            row.empty_frames += 1

        frames.append(
            DebugFrameResult(
                class_name=class_name,
                source=source,
                labels=labels,
                bundle_dir=bundle_dir,
                snapshot=snapshot,
            )
        )

    classes = [per_class_stats[name] for name in sorted(per_class_stats)]
    dataset_report = DatasetReport(
        classes=classes,
        total_frames=sum(row.frames for row in classes),
        total_crops=sum(row.crops for row in classes),
        empty_frames=sum(row.empty_frames for row in classes),
    )
    return DebugSampleReport(
        dataset_report=dataset_report,
        frames=frames,
        result_root=result_root,
        bundle_root=bundle_root,
    )
