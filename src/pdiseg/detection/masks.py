"""Candidate masks: text density, dark luma, black-hat, glare."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_closing, uniform_filter
from skimage.morphology import black_tophat, footprint_rectangle

from .config import DetectionConfig


@dataclass(frozen=True)
class CandidateMasks:
    text_density: NDArray[np.bool_]
    dark_luma: NDArray[np.bool_]
    black_hat: NDArray[np.bool_]
    glare: NDArray[np.bool_]
    combined: NDArray[np.bool_]


def glare_mask(image: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.bool_]:
    threshold = float(np.percentile(image, config.glare_percentile))
    return image >= threshold


def build_candidate_masks(work: NDArray[np.uint8], config: DetectionConfig) -> CandidateMasks:
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

    return CandidateMasks(
        text_density=text_density,
        dark_luma=dark_luma,
        black_hat=black_hat,
        glare=glare,
        combined=combined,
    )


def text_density_mask(work: NDArray[np.uint8], config: DetectionConfig) -> NDArray[np.bool_]:
    local = uniform_filter(work.astype(np.float32), size=config.text_local_window)
    merged = binary_closing(
        work.astype(np.float32) > local + config.text_offset,
        structure=footprint_rectangle((7, 15)),  # type: ignore[no-untyped-call]
    )
    return np.asarray(merged, dtype=np.bool_)
