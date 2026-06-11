"""End-to-end behavior of the walking skeleton, exercised through the public API."""

import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import pdiseg

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_multiple_detections_are_numbered_from_one(tmp_path):
    dataset = tmp_path / "dataset"
    _write_gray(dataset / "Moela" / "img003.jpg", size=(20, 20))

    def two_boxes(image):
        return [(0, 0, 10, 20), (10, 0, 10, 20)]

    out = tmp_path / "resultado"
    pdiseg.run(dataset, out, detector=two_boxes)

    assert (out / "Moela" / "img003_segmentada_1.png").exists()
    assert (out / "Moela" / "img003_segmentada_2.png").exists()
    # Numbering is 1-based — there is no _0.
    assert not (out / "Moela" / "img003_segmentada_0.png").exists()


def test_find_source_images_discovers_images_and_ignores_other_files(tmp_path):
    dataset = tmp_path / "dataset"
    _write_gray(dataset / "ClassA" / "a.jpg")
    _write_gray(dataset / "ClassA" / "b.png")
    _write_gray(dataset / "ClassB" / "c.jpeg")
    (dataset / "ClassA" / "notes.txt").write_text("not an image")
    _write_gray(dataset / "loose.jpg")  # at root, not inside a class folder

    found = pdiseg.find_source_images(dataset)

    assert {p.name for p in found} == {"a.jpg", "b.png", "c.jpeg"}


def test_find_source_images_ignores_zone_identifier_sidecars(tmp_path):
    # The real dataset ships a Windows "<name>.jpgZone.Identifier" sidecar next to
    # every image; those must never be treated as source images.
    dataset = tmp_path / "dataset"
    _write_gray(dataset / "ClassA" / "img001.jpg")
    (dataset / "ClassA" / "img001.jpgZone.Identifier").write_text("[ZoneTransfer]")

    found = pdiseg.find_source_images(dataset)

    assert [p.name for p in found] == ["img001.jpg"]


def test_module_entry_point_runs_over_given_dirs(tmp_path):
    dataset = tmp_path / "dataset"
    _write_gray(dataset / "Peito_Congelado" / "img001.jpg")
    out = tmp_path / "resultado"

    result = subprocess.run(
        [sys.executable, "-m", "pdiseg", str(dataset), str(out)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "Peito_Congelado" / "img001_segmentada_1.png").exists()
