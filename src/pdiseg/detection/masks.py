"""Candidate masks: text density, dark luma, black-hat, glare, edge density, DoG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    gaussian_filter,
    median_filter,
    sobel,
    uniform_filter,
)
from skimage.morphology import black_tophat, disk, footprint_rectangle, opening
from skimage.segmentation import clear_border

from .config import DetectionConfig


@dataclass(frozen=True)
class CandidateMasks:
    text_density: NDArray[np.bool_]
    dark_luma: NDArray[np.bool_]
    black_hat: NDArray[np.bool_]
    glare: NDArray[np.bool_]
    combined: NDArray[np.bool_]
    edge_density: NDArray[np.bool_] | None = None


def glare_mask(image: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.bool_]:
    threshold = float(np.percentile(image, config.glare_percentile))
    return image >= threshold


def opened_background(gray: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.uint8]:
    radius = max(3, config.opened_background_size // 2)
    opened = opening(gray, footprint=disk(radius))  # type: ignore[no-untyped-call]
    return np.asarray(opened, dtype=np.uint8)


def edge_density_mask(gray: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.bool_]:
    """Sobel edge-density text mask (notebook cell 26, scikit-image stack)."""
    smoothed = median_filter(gray, size=3).astype(np.float32)
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    magnitude = np.hypot(gx, gy)
    edge = magnitude > config.edge_mag_threshold

    window = max(7, int(gray.shape[1] * config.edge_density_window_frac))
    if window % 2 == 0:
        window += 1
    density = uniform_filter(edge.astype(np.float32), size=window)
    text = density > config.edge_density_min

    close_r = max(3, config.edge_close_size // 2)
    open_r = max(3, config.edge_open_size // 2)
    text = binary_closing(text, structure=disk(close_r))  # type: ignore[no-untyped-call]
    text = binary_opening(text, structure=disk(open_r))  # type: ignore[no-untyped-call]
    return np.asarray(text, dtype=np.bool_)


def dark_body_mask(gray: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.bool_]:
    """Adaptive dark solid-body gate (reference_segment_label dark_body_mask, scikit-image stack)."""
    from skimage.filters import threshold_local

    height, width = gray.shape
    block = max(15, (min(height, width) // config.body_block_div) | 1)
    thresh_map = threshold_local(gray, block_size=block, offset=-config.body_C)  # type: ignore[no-untyped-call]
    body = gray < thresh_map
    body = binary_closing(body, structure=disk(10))  # type: ignore[no-untyped-call]
    body = binary_opening(body, structure=disk(4))  # type: ignore[no-untyped-call]
    return np.asarray(body, dtype=np.bool_)


def dog_text_mask(gray: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.bool_]:
    """DoG-style bright text on locally dark background (reference_segment_label L209-223)."""
    height, width = gray.shape
    values = gray.astype(np.float32)
    sigma = min(config.dog_sigma, min(height, width) / 10.0)
    background = gaussian_filter(values, sigma=max(3.0, sigma))
    dark_bg = background < np.percentile(background, config.dog_bg_percentile)
    text = ((values - background) > config.dog_contrast_threshold) & dark_bg
    bold_r = max(1, config.dog_bold_size // 2)
    bold = binary_opening(text, structure=disk(bold_r))  # type: ignore[no-untyped-call]
    dil_r = max(3, config.dog_dilate_size // 2)
    grouped = binary_dilation(bold, structure=disk(dil_r))  # type: ignore[no-untyped-call]
    grouped = binary_closing(grouped, structure=disk(dil_r))  # type: ignore[no-untyped-call]
    return np.asarray(grouped, dtype=np.bool_)


def _apply_clear_border(
    mask: NDArray[np.bool_], height: int, width: int, config: DetectionConfig
) -> NDArray[np.bool_]:
    buffer = max(1, int(min(height, width) * config.clear_border_buffer_frac))
    cleared = clear_border(mask, buffer_size=buffer)  # type: ignore[no-untyped-call]
    return np.asarray(cleared, dtype=np.bool_)


def build_candidate_masks(
    work: NDArray[np.uint8],
    config: DetectionConfig,
    *,
    gray: NDArray[np.uint8] | None = None,
) -> CandidateMasks:
    height, width = work.shape
    local = uniform_filter(work.astype(np.float32), size=config.text_local_window)
    text_density = work.astype(np.float32) > local + config.text_offset

    dark_threshold = float(np.percentile(work, config.dark_percentile))
    dark_luma = work <= dark_threshold

    selem = footprint_rectangle((max(3, height // 80), max(3, width // 120)))  # type: ignore[no-untyped-call]
    black_hat_img = black_tophat(work, selem)
    bh_threshold = float(np.percentile(black_hat_img, 72))
    black_hat = black_hat_img >= bh_threshold

    glare = glare_mask(work, config)
    combined = (text_density | dark_luma | black_hat) & ~glare

    close_h = max(3, width // 180)
    close_v = max(3, height // 120)
    combined = binary_closing(combined, structure=footprint_rectangle((close_v, close_h)))  # type: ignore[no-untyped-call]
    combined = _apply_clear_border(combined, height, width, config)

    edge: NDArray[np.bool_] | None = None
    if gray is not None:
        edge = edge_density_mask(gray, config)

    return CandidateMasks(
        text_density=text_density,
        dark_luma=dark_luma,
        black_hat=black_hat,
        glare=glare,
        combined=combined,
        edge_density=edge,
    )


def text_density_mask(work: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.bool_]:
    local = uniform_filter(work.astype(np.float32), size=config.text_local_window)
    merged = binary_closing(
        work.astype(np.float32) > local + config.text_offset,
        structure=footprint_rectangle((7, 15)),  # type: ignore[no-untyped-call]
    )
    return np.asarray(merged, dtype=np.bool_)
