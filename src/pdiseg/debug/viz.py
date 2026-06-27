"""Notebook helpers for mask/box visualization and debug bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox, FrameInspection, crop, render_overlay
from pdiseg.detection.config import DetectionConfig
from pdiseg.detection.detector import DetectionResult, inspect_detection
from pdiseg.detection.masks import CandidateMasks, build_candidate_masks, opened_background
from pdiseg.detection.preprocess import PreprocessResult, preprocess_image
from pdiseg.detection.scoring import ScoredCandidate

if TYPE_CHECKING:
    import matplotlib.figure


@dataclass(frozen=True)
class FrameDebugSnapshot:
    """Full per-frame pipeline state for notebook plots and saved bundles."""

    detection: DetectionResult
    prep: PreprocessResult
    masks: CandidateMasks


def to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    from typing import cast

    from skimage.color import gray2rgb
    from skimage.util import img_as_ubyte

    if image.ndim == 3:
        return image
    return cast(
        NDArray[np.uint8],
        gray2rgb(img_as_ubyte(image)),  # type: ignore[no-untyped-call]
    )


def draw_boxes(
    image: NDArray[np.uint8], boxes: list[BBox], color: tuple[int, int, int]
) -> NDArray[np.uint8]:
    from pdiseg.core.imaging import _draw_box

    rgb = to_rgb(image).copy()
    for box in boxes:
        _draw_box(rgb, box, color)
    return rgb


def visualize_masks(masks: CandidateMasks) -> dict[str, NDArray[np.uint8]]:
    panels: dict[str, NDArray[np.uint8]] = {
        "text_density": masks.text_density.astype(np.uint8) * 255,
        "dark_luma": masks.dark_luma.astype(np.uint8) * 255,
        "black_hat": masks.black_hat.astype(np.uint8) * 255,
        "glare": masks.glare.astype(np.uint8) * 255,
        "combined": masks.combined.astype(np.uint8) * 255,
    }
    if masks.edge_density is not None:
        panels["edge_density"] = masks.edge_density.astype(np.uint8) * 255
    if masks.dog_text is not None:
        panels["dog_text"] = masks.dog_text.astype(np.uint8) * 255
    return panels


def visualize_opened_background(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    from pdiseg.detection.config import DetectionConfig

    return opened_background(gray, DetectionConfig())


def debug_frame(
    image: NDArray[np.uint8],
    config: DetectionConfig | None = None,
) -> tuple[DetectionResult, PreprocessResult, CandidateMasks]:
    from pdiseg.detection.config import DetectionConfig

    cfg = config or DetectionConfig()
    prep = preprocess_image(image, cfg)
    masks = build_candidate_masks(prep.work, cfg, gray=prep.gray)
    detection = inspect_detection(image, cfg)
    return detection, prep, masks


def analyze_frame(
    image: NDArray[np.uint8],
    config: DetectionConfig | None = None,
) -> FrameDebugSnapshot:
    detection, prep, masks = debug_frame(image, config)
    return FrameDebugSnapshot(detection=detection, prep=prep, masks=masks)


def _flat_axes(count: int, ncols: int = 6) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    import matplotlib.pyplot as plt

    nrows = max(1, math.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows))
    flat = np.atleast_1d(axes).ravel()
    return fig, flat


def plot_all_preprocessed(
    items: list[tuple[str, FrameDebugSnapshot]],
    *,
    ncols: int = 6,
    suptitle: str = "Preprocessed — all sample frames (CLAHE + FPS mask)",
) -> matplotlib.figure.Figure:

    import matplotlib.pyplot as plt

    fig, axes = _flat_axes(len(items), ncols=ncols)
    for ax, (title, snap) in zip(axes, items, strict=False):
        ax.imshow(snap.prep.work, cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    return fig


def plot_all_preprocess_stages(
    items: list[tuple[str, FrameDebugSnapshot]],
    *,
    ncols: int = 6,
) -> list[matplotlib.figure.Figure]:
    """Grayscale, CLAHE work image, and opened background for every sample frame."""
    import matplotlib.pyplot as plt

    stages: list[tuple[str, str]] = [
        ("1. grayscale input", "gray"),
        ("2. preprocessed (work)", "work"),
        ("3. opened background", "opened"),
    ]
    figures: list[matplotlib.figure.Figure] = []
    for stage_title, key in stages:
        fig, axes = _flat_axes(len(items), ncols=ncols)
        for ax, (title, snap) in zip(axes, items, strict=False):
            if key == "gray":
                panel = snap.prep.gray
            elif key == "work":
                panel = snap.prep.work
            else:
                panel = visualize_opened_background(snap.prep.gray)
            ax.imshow(panel, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        for ax in axes[len(items) :]:
            ax.axis("off")
        fig.suptitle(f"Preprocess — {stage_title}", fontsize=12)
        plt.tight_layout()
        figures.append(fig)
    return figures


def plot_all_mask_layers(
    items: list[tuple[str, FrameDebugSnapshot]],
    *,
    ncols: int = 6,
) -> list[matplotlib.figure.Figure]:
    """One grid figure per mask layer (text_density, dark_luma, black_hat, …)."""
    import matplotlib.pyplot as plt

    if not items:
        return []
    layer_names = list(visualize_masks(items[0][1].masks).keys())
    figures: list[matplotlib.figure.Figure] = []
    for layer in layer_names:
        fig, axes = _flat_axes(len(items), ncols=ncols)
        for ax, (title, snap) in zip(axes, items, strict=False):
            panels = visualize_masks(snap.masks)
            ax.imshow(panels[layer], cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        for ax in axes[len(items) :]:
            ax.axis("off")
        fig.suptitle(f"Mask layer: {layer}", fontsize=12)
        plt.tight_layout()
        figures.append(fig)
    return figures


def plot_frame_masks(
    snapshot: FrameDebugSnapshot,
    *,
    title: str = "",
    figsize: tuple[float, float] = (14, 3.5),
) -> matplotlib.figure.Figure:
    """All mask layers side-by-side for a single frame."""
    import matplotlib.pyplot as plt

    panels = visualize_masks(snapshot.masks)
    count = len(panels)
    fig, axes = plt.subplots(1, count, figsize=figsize)
    for ax, (name, panel) in zip(np.atleast_1d(axes), panels.items(), strict=False):
        ax.imshow(panel, cmap="gray")
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


DetectionStage = Literal["candidates", "kept", "labels", "overlay"]


def plot_all_detection_stage(
    images: list[NDArray[np.uint8]],
    items: list[tuple[str, FrameDebugSnapshot]],
    stage: DetectionStage = "candidates",
    *,
    ncols: int = 6,
) -> matplotlib.figure.Figure:
    """Grid of one detection stage across all sample frames."""
    import matplotlib.pyplot as plt

    stage_titles = {
        "candidates": ("6. raw candidates (red)", (220, 50, 50), "candidates"),
        "kept": ("7. kept after scoring (yellow)", (240, 200, 40), "kept"),
        "labels": ("8. final labels (green)", (40, 200, 80), "labels"),
        "overlay": ("Post-process overlay (R/Y/G)", None, None),
    }
    caption, color, attr = stage_titles[stage]
    fig, axes = _flat_axes(len(items), ncols=ncols)
    for ax, image, (title, snap) in zip(axes, images, items, strict=False):
        detection = snap.detection
        if stage == "overlay":
            panel = render_overlay(
                image,
                FrameInspection(
                    candidates=detection.candidates,
                    kept=detection.kept,
                    labels=detection.labels,
                ),
            )
            count = len(detection.labels)
        else:
            assert attr is not None and color is not None
            boxes = getattr(detection, attr)
            panel = draw_boxes(image, boxes, color)
            count = len(boxes)
        ax.imshow(panel)
        ax.set_title(f"{title}\nn={count}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(caption, fontsize=12)
    plt.tight_layout()
    return fig


def plot_all_overlays(
    images: list[NDArray[np.uint8]],
    items: list[tuple[str, FrameDebugSnapshot]],
    *,
    ncols: int = 6,
    suptitle: str = "Post-process overlays — all sample frames (R/Y/G boxes)",
) -> matplotlib.figure.Figure:

    import matplotlib.pyplot as plt

    fig, axes = _flat_axes(len(items), ncols=ncols)
    for ax, image, (title, snap) in zip(axes, images, items, strict=False):
        overlay = render_overlay(
            image,
            FrameInspection(
                candidates=snap.detection.candidates,
                kept=snap.detection.kept,
                labels=snap.detection.labels,
            ),
        )
        ax.imshow(overlay)
        ax.set_title(f"{title}\nL={len(snap.detection.labels)}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    return fig


def plot_all_masks(
    items: list[tuple[str, FrameDebugSnapshot]],
    mask_name: str = "combined",
    *,
    ncols: int = 6,
) -> matplotlib.figure.Figure:

    import matplotlib.pyplot as plt

    fig, axes = _flat_axes(len(items), ncols=ncols)
    for ax, (title, snap) in zip(axes, items, strict=False):
        panels = visualize_masks(snap.masks)
        ax.imshow(panels.get(mask_name, panels["combined"]), cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[len(items) :]:
        ax.axis("off")
    fig.suptitle(f"Mask layer: {mask_name}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_frame_pipeline(
    image: NDArray[np.uint8],
    snapshot: FrameDebugSnapshot,
    *,
    title: str = "",
    figsize: tuple[float, float] = (15, 9),
) -> matplotlib.figure.Figure:

    import matplotlib.pyplot as plt

    detection = snapshot.detection
    prep = snapshot.prep
    masks = snapshot.masks
    mask_views = visualize_masks(masks)

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes[0, 0].imshow(to_rgb(image))
    axes[0, 0].set_title("1. original")
    axes[0, 1].imshow(prep.gray, cmap="gray")
    axes[0, 1].set_title("2. grayscale input")
    axes[0, 2].imshow(prep.work, cmap="gray")
    axes[0, 2].set_title("3. preprocessed (work)")
    axes[0, 3].imshow(visualize_opened_background(prep.gray), cmap="gray")
    axes[0, 3].set_title("4. opened background")

    axes[1, 0].imshow(mask_views["combined"], cmap="gray")
    axes[1, 0].set_title(f"5. combined mask ({len(detection.candidates)} cand.)")
    axes[1, 1].imshow(draw_boxes(image, detection.candidates, (220, 50, 50)))
    axes[1, 1].set_title("6. raw candidates")
    axes[1, 2].imshow(draw_boxes(image, detection.kept, (240, 200, 40)))
    axes[1, 2].set_title(f"7. kept ({len(detection.kept)})")
    axes[1, 3].imshow(draw_boxes(image, detection.labels, (40, 200, 80)))
    axes[1, 3].set_title(f"8. final labels ({len(detection.labels)})")

    for ax in axes.ravel():
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def plot_frame_full_debug(
    image: NDArray[np.uint8],
    snapshot: FrameDebugSnapshot,
    *,
    title: str = "",
) -> matplotlib.figure.Figure:
    """Extended single-frame panel: preprocess, every mask, and all detection stages."""
    import matplotlib.pyplot as plt

    detection = snapshot.detection
    prep = snapshot.prep
    mask_views = visualize_masks(snapshot.masks)
    mask_names = list(mask_views.keys())
    n_masks = len(mask_names)
    ncols = max(4, n_masks)
    fig, axes = plt.subplots(3, ncols, figsize=(2.6 * ncols, 8.5))
    row0 = axes[0]
    row1 = axes[1]
    row2 = axes[2]

    row0[0].imshow(to_rgb(image))
    row0[0].set_title("original")
    row0[1].imshow(prep.gray, cmap="gray")
    row0[1].set_title("grayscale")
    row0[2].imshow(prep.work, cmap="gray")
    row0[2].set_title("work (CLAHE)")
    row0[3].imshow(visualize_opened_background(prep.gray), cmap="gray")
    row0[3].set_title("opened background")
    for ax in row0[4:]:
        ax.axis("off")

    for ax, name in zip(row1, mask_names, strict=False):
        ax.imshow(mask_views[name], cmap="gray")
        ax.set_title(name.replace("_", " "))
    for ax in row1[len(mask_names) :]:
        ax.axis("off")

    stages: list[tuple[str, NDArray[np.uint8]]] = [
        (f"candidates ({len(detection.candidates)})", draw_boxes(image, detection.candidates, (220, 50, 50))),
        (f"kept ({len(detection.kept)})", draw_boxes(image, detection.kept, (240, 200, 40))),
        (f"labels ({len(detection.labels)})", draw_boxes(image, detection.labels, (40, 200, 80))),
        (
            "overlay R/Y/G",
            render_overlay(
                image,
                FrameInspection(
                    candidates=detection.candidates,
                    kept=detection.kept,
                    labels=detection.labels,
                ),
            ),
        ),
    ]
    for ax, (label, panel) in zip(row2, stages, strict=False):
        ax.imshow(panel)
        ax.set_title(label)
        ax.axis("off")
    for ax in row2[len(stages) :]:
        ax.axis("off")

    for ax in row0[:4]:
        ax.axis("off")
    for ax in row1[: len(mask_names)]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def plot_all_frame_pipelines(
    images: list[NDArray[np.uint8]],
    items: list[tuple[str, FrameDebugSnapshot]],
) -> list[matplotlib.figure.Figure]:
    figures: list[matplotlib.figure.Figure] = []
    for image, (title, snap) in zip(images, items, strict=True):
        figures.append(plot_frame_pipeline(image, snap, title=title))
    return figures


def plot_all_crops(
    images: list[NDArray[np.uint8]],
    labels_per_frame: list[list[BBox]],
    titles: list[str],
    *,
    ncols: int = 6,
) -> matplotlib.figure.Figure | None:

    import matplotlib.pyplot as plt

    panels: list[tuple[str, NDArray[np.uint8]]] = []
    for image, labels, frame_title in zip(images, labels_per_frame, titles, strict=True):
        if not labels:
            panels.append((f"{frame_title}\n(no crop)", np.full((48, 48), 180, dtype=np.uint8)))
            continue
        for index, box in enumerate(labels, start=1):
            panels.append((f"{frame_title}\n#{index}", crop(image, box)))

    if not panels:
        return None

    fig, axes = _flat_axes(len(panels), ncols=ncols)
    for ax, (title, patch) in zip(axes, panels, strict=False):
        ax.imshow(patch, cmap="gray")
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle("Saved crops — all sample frames", fontsize=12)
    plt.tight_layout()
    return fig


def list_bundle_artifacts(bundle_dir: Path, stem: str) -> list[Path]:
    """Sorted PNG paths written by :func:`save_debug_bundle` for one frame."""
    return sorted(bundle_dir.glob(f"{stem}_*.png"))


def plot_bundle_gallery(
    bundle_dir: Path,
    stem: str,
    *,
    ncols: int = 4,
    figsize_scale: float = 3.2,
) -> matplotlib.figure.Figure | None:
    """Grid of every PNG artifact saved on disk for one frame."""
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    paths = list_bundle_artifacts(bundle_dir, stem)
    if not paths:
        return None

    fig, axes = _flat_axes(len(paths), ncols=ncols)
    for ax, path in zip(axes, paths, strict=False):
        ax.imshow(iio.imread(path))
        ax.set_title(path.stem.removeprefix(f"{stem}_"), fontsize=8)
        ax.axis("off")
    for ax in axes[len(paths) :]:
        ax.axis("off")
    fig.set_size_inches(figsize_scale * ncols, figsize_scale * max(1, math.ceil(len(paths) / ncols)))
    fig.suptitle(f"On-disk bundle: {stem}", fontsize=12)
    plt.tight_layout()
    return fig


def save_pipeline_panel(
    image: NDArray[np.uint8],
    snapshot: FrameDebugSnapshot,
    path: str | Path,
    *,
    title: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_frame_pipeline(image, snapshot, title=title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_debug_bundle(
    image: NDArray[np.uint8],
    output_dir: str | Path,
    stem: str,
    *,
    snapshot: FrameDebugSnapshot | None = None,
    title: str = "",
) -> Path:
    import imageio.v3 as iio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if snapshot is None:
        detection, prep, masks = debug_frame(image)
        snapshot = FrameDebugSnapshot(detection=detection, prep=prep, masks=masks)
    detection = snapshot.detection
    prep = snapshot.prep
    masks = snapshot.masks

    iio.imwrite(output_dir / f"{stem}_source.png", image)
    iio.imwrite(output_dir / f"{stem}_gray.png", prep.gray)
    iio.imwrite(output_dir / f"{stem}_preprocess.png", prep.work)
    iio.imwrite(output_dir / f"{stem}_opened_background.png", visualize_opened_background(prep.gray))
    iio.imwrite(
        output_dir / f"{stem}_overlay.png",
        render_overlay(
            image,
            FrameInspection(
                candidates=detection.candidates,
                kept=detection.kept,
                labels=detection.labels,
            ),
        ),
    )
    iio.imwrite(
        output_dir / f"{stem}_candidates.png",
        draw_boxes(image, detection.candidates, (220, 50, 50)),
    )
    iio.imwrite(
        output_dir / f"{stem}_kept.png",
        draw_boxes(image, detection.kept, (240, 200, 40)),
    )
    iio.imwrite(
        output_dir / f"{stem}_labels.png",
        draw_boxes(image, detection.labels, (40, 200, 80)),
    )
    for name, mask_img in visualize_masks(masks).items():
        iio.imwrite(output_dir / f"{stem}_mask_{name}.png", mask_img)
    for index, box in enumerate(detection.labels, start=1):
        iio.imwrite(output_dir / f"{stem}_crop_{index}.png", crop(image, box))
    save_pipeline_panel(image, snapshot, output_dir / f"{stem}_pipeline.png", title=title)
    return output_dir


def scored_table(
    scored: list[ScoredCandidate],
) -> list[dict[str, float | int | tuple[int, int, int, int]]]:
    rows: list[dict[str, float | int | tuple[int, int, int, int]]] = []
    for rank, item in enumerate(sorted(scored, key=lambda row: row.score, reverse=True), start=1):
        row: dict[str, float | int | tuple[int, int, int, int]] = {
            "rank": rank,
            "score": item.score,
            "box": item.box,
        }
        row.update(item.features)
        rows.append(row)
    return rows


def feature_summary(scored: list[ScoredCandidate]) -> list[str]:
    """Human-readable lines for notebook display (TASK-03/04/05 panels)."""
    keys = (
        "background_level",
        "bright_on_dark",
        "extent",
        "bimodal_score",
        "body_overlap",
    )
    lines: list[str] = []
    for item in sorted(scored, key=lambda row: row.score, reverse=True)[:5]:
        parts = [f"score={item.score:.3f}"]
        for key in keys:
            if key in item.features:
                parts.append(f"{key}={item.features[key]:.3f}")
        lines.append(" | ".join(parts))
    return lines
