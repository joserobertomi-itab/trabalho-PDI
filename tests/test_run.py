"""End-to-end behavior of the walking skeleton, exercised through the public API."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

import pdiseg


def _write_gray(path: Path, value: int = 120, size=(16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full(size, value, dtype=np.uint8)
    iio.imwrite(path, img)


def test_run_mirrors_classes_and_writes_a_valid_crop_per_source(tmp_path):
    # Synthetic dataset: two class folders, a few grayscale frames.
    dataset = tmp_path / "dataset"
    sources = {
        "Peito_Congelado": ["img001.jpg", "img002.jpg"],
        "Moela": ["img003.jpg"],
    }
    for klass, files in sources.items():
        for name in files:
            _write_gray(dataset / klass / name)

    out = tmp_path / "resultado"
    pdiseg.run(dataset, out)

    for klass, files in sources.items():
        for name in files:
            stem = Path(name).stem
            produced = out / klass / f"{stem}_segmentada_1.png"
            assert produced.exists(), f"missing output for {klass}/{name}"
            # Crop must be a valid, readable PNG with image content.
            data = iio.imread(produced)
            assert data.ndim >= 2 and data.size > 0
