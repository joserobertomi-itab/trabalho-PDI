"""Notebook helpers for mask/box visualization and debug bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pdiseg.core.imaging import BBox, FrameInspection, crop, render_overlay
from pdiseg.detection.detector import DetectionResult, inspect_detection
from pdiseg.detection.masks import CandidateMasks, build_candidate_masks, opened_background
from pdiseg.detection.preprocess import PreprocessResult, preprocess_image
from pdiseg.detection.scoring import ScoredCandidate


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
    return panels


def visualize_opened_background(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    from pdiseg.detection.config import DetectionConfig

    return opened_background(gray, DetectionConfig())


def debug_frame(
    image: NDArray[np.uint8],
) -> tuple[DetectionResult, PreprocessResult, CandidateMasks]:
    from pdiseg.detection.config import DetectionConfig

    prep = preprocess_image(image)
    masks = build_candidate_masks(prep.work, DetectionConfig(), gray=prep.gray)
    detection = inspect_detection(image)
    return detection, prep, masks


def save_debug_bundle(
    image: NDArray[np.uint8],
    output_dir: str | Path,
    stem: str,
) -> Path:
    import imageio.v3 as iio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detection, prep, masks = debug_frame(image)
    iio.imwrite(output_dir / f"{stem}_source.png", image)
    iio.imwrite(output_dir / f"{stem}_work.png", prep.work)
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
    for name, mask_img in visualize_masks(masks).items():
        iio.imwrite(output_dir / f"{stem}_mask_{name}.png", mask_img)
    for index, box in enumerate(detection.labels, start=1):
        iio.imwrite(output_dir / f"{stem}_crop_{index}.png", crop(image, box))
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
