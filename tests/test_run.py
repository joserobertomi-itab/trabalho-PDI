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


def test_run_detects_on_preprocessed_but_crops_from_original(tmp_path):
    dataset = tmp_path / "dataset"
    src = dataset / "ClassA" / "img001.png"  # PNG so the crop compares exactly
    src.parent.mkdir(parents=True)
    original = np.full((10, 10), 200, dtype=np.uint8)
    iio.imwrite(src, original)

    seen = {}

    def marking_preprocessor(image):
        return np.zeros_like(image)  # erase everything

    def recording_detector(image):
        seen["max"] = int(image.max())  # what the detector actually received
        h, w = image.shape[:2]
        return [(0, 0, w, h)]

    out = tmp_path / "resultado"
    pdiseg.run(
        dataset,
        out,
        detector=recording_detector,
        preprocessor=marking_preprocessor,
    )

    # The detector saw the PREPROCESSED image (all zeros)...
    assert seen["max"] == 0
    # ...but the written crop came from the ORIGINAL frame (all 200).
    written = iio.imread(out / "ClassA" / "img001_segmentada_1.png")
    assert int(written.min()) == 200 and int(written.max()) == 200


def test_preprocess_masks_the_fps_overlay_region():
    img = np.full((100, 300), 50, dtype=np.uint8)
    x, y, w, h = pdiseg.FPS_OVERLAY_REGION
    img[y : y + h, x : x + w] = 255  # simulate the bright FPS text

    out = pdiseg.preprocess(img)

    region = out[y : y + h, x : x + w]
    # The overlay region is wiped to a single constant value (masked out).
    assert region.min() == region.max() == 0
    assert out.shape == img.shape


def test_preprocess_equalizes_histogram_to_spread_contrast():
    # Low-contrast ramp confined to a narrow intensity band.
    ramp = np.linspace(100, 130, 256).astype(np.uint8)
    img = np.tile(ramp, (300, 1))  # shape (300, 256)

    out = pdiseg.preprocess(img)

    # Compare a slice to the right of the masked FPS region (unaffected by masking).
    _, _, w, _ = pdiseg.FPS_OVERLAY_REGION
    in_range = int(img[:, w:].max()) - int(img[:, w:].min())
    out_range = int(out[:, w:].max()) - int(out[:, w:].min())
    assert out_range > in_range


def test_preprocess_removes_salt_noise_with_median():
    # Varied (gradient) background so equalization is well-behaved, plus one impulse.
    col = np.linspace(0, 255, 300).astype(np.uint8)
    img = np.repeat(col[:, None], 300, axis=1)  # shape (300, 300), value varies by row
    img[150, 250] = 255  # lone salt impulse, outside the FPS region

    out = pdiseg.preprocess(img)

    # Median (run before equalization) replaces the impulse with its local value,
    # so it ends up identical to its same-row neighbour. Without median it would not.
    assert out[150, 250] == out[150, 240]


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


def test_dump_preprocessed_writes_limited_preview_images(tmp_path):
    dataset = tmp_path / "dataset"
    for i in range(4):
        _write_gray(dataset / "ClassA" / f"img{i}.jpg")

    preview = tmp_path / "preview"
    written = pdiseg.dump_preprocessed(dataset, preview, limit=3)

    assert len(written) == 3
    for path in written:
        assert Path(path).exists()
        data = iio.imread(path)
        assert data.ndim >= 2 and data.size > 0


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
