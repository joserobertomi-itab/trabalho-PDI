"""Median denoise, CLAHE, and FPS overlay masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter, uniform_filter

from pdiseg.io.dataset import FPS_OVERLAY_REGION

from .config import DetectionConfig


@dataclass(frozen=True)
class PreprocessResult:
    gray: NDArray[np.uint8]
    clahe: NDArray[np.uint8]
    work: NDArray[np.uint8]


def preprocess_image(
    image: NDArray[np.uint8], config: DetectionConfig | None = None
) -> PreprocessResult:
    _ = config
    from skimage.exposure import equalize_adapthist
    from skimage.util import img_as_ubyte

    gray = image.astype(np.uint8)
    if gray.ndim > 2:
        gray = gray[..., 0]
    denoised = median_filter(gray, size=3)
    clahe = cast(
        NDArray[np.uint8],
        img_as_ubyte(equalize_adapthist(denoised, clip_limit=0.02)),  # type: ignore[no-untyped-call]
    )
    work = clahe.copy()
    x, y, w, h = FPS_OVERLAY_REGION
    work[y : y + h, x : x + w] = int(np.median(work))
    return PreprocessResult(gray=gray, clahe=clahe, work=work)


def preprocess(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return preprocess_image(image).work


def background_estimate(image: NDArray[np.uint8], sigma: float = 25.0) -> NDArray[np.float32]:
    return np.asarray(
        uniform_filter(image.astype(np.float32), size=max(3, int(sigma))),
        dtype=np.float32,
    )


def shadow_corrected(image: NDArray[np.uint8], sigma: float = 25.0) -> NDArray[np.uint8]:
    bg = background_estimate(image, sigma=sigma)
    corrected = image.astype(np.float32) - bg + float(np.median(image))
    return np.clip(corrected, 0, 255).astype(np.uint8)  # type: ignore[no-any-return]
