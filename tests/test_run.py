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

    dataset = tmp_path / "dataset"
    sources = {
        "Peito_Congelado": ["img001.jpg", "img002.jpg"],
        "Moela": ["img003.jpg"],
    }
    for klass, files in sources.items():
        for name in files:
            _write_gray(dataset / klass / name)

    out = tmp_path / "result"
    pdiseg.run(dataset, out, detector=pdiseg.detect)

    for klass, files in sources.items():
        for name in files:
            stem = Path(name).stem
            produced = out / klass / f"{stem}_segmented_1.png"
            assert produced.exists(), f"missing output for {klass}/{name}"

            data = iio.imread(produced)
            assert data.ndim >= 2 and data.size > 0


def test_multiple_detections_are_numbered_from_one(tmp_path):
    dataset = tmp_path / "dataset"
    _write_gray(dataset / "Moela" / "img003.jpg", size=(20, 20))

    def two_boxes(image):
        return [(0, 0, 10, 20), (10, 0, 10, 20)]

    out = tmp_path / "result"
    pdiseg.run(dataset, out, detector=two_boxes)

    assert (out / "Moela" / "img003_segmented_1.png").exists()
    assert (out / "Moela" / "img003_segmented_2.png").exists()

    assert not (out / "Moela" / "img003_segmented_0.png").exists()


def test_run_crops_from_the_original_frame(tmp_path):
    dataset = tmp_path / "dataset"
    src = dataset / "ClassA" / "img001.png"
    src.parent.mkdir(parents=True)
    original = np.full((10, 10), 200, dtype=np.uint8)
    iio.imwrite(src, original)

    def box_detector(image):
        return [(2, 2, 5, 5)]

    out = tmp_path / "result"
    pdiseg.run(dataset, out, detector=box_detector)

    written = iio.imread(out / "ClassA" / "img001_segmented_1.png")

    assert written.shape == (5, 5)
    assert int(written.min()) == 200 and int(written.max()) == 200


def _draw_text_block(img, x0, y0, x1, y1, bg=30, ink=220):

    img[y0:y1, x0:x1] = bg
    for x in range(x0, x1, 8):
        img[y0:y1, x : x + 3] = ink


def _gradient_frame(h, w):

    return np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))


def test_detect_clusters_finds_a_single_text_block(tmp_path=None):
    img = np.full((200, 400), 30, dtype=np.uint8)
    _draw_text_block(img, 150, 80, 300, 140)

    clusters = pdiseg.detect_clusters(img)

    assert len(clusters) == 1
    x, y, w, h = clusters[0]
    cx, cy = x + w / 2, y + h / 2
    assert 150 <= cx <= 300 and 80 <= cy <= 140
    assert w >= 100 and h >= 30


def test_detect_clusters_returns_nothing_on_a_uniform_image():
    img = np.full((200, 400), 120, dtype=np.uint8)
    assert pdiseg.detect_clusters(img) == []


def test_detect_clusters_separates_two_distant_blocks():
    img = np.full((200, 500), 30, dtype=np.uint8)
    _draw_text_block(img, 40, 80, 180, 140)
    _draw_text_block(img, 320, 80, 460, 140)

    assert len(pdiseg.detect_clusters(img)) == 2


def test_detect_clusters_ignores_specks_below_min_area():
    img = np.full((200, 400), 30, dtype=np.uint8)
    _draw_text_block(img, 10, 10, 30, 30)

    assert pdiseg.detect_clusters(img) == []


def test_refine_to_name_label_isolates_the_dark_region():
    img = np.full((200, 500), 120, dtype=np.uint8)

    img[50:130, 100:180] = 210
    img[50:130, 180:260] = 40
    cluster = (100, 50, 160, 80)

    rx, ry, rw, rh = pdiseg.refine_to_name_label(img, cluster)

    assert rx >= 100 and ry >= 50 and rx + rw <= 260 and ry + rh <= 130
    assert rx + rw / 2 > 100 + 160 / 2
    assert rw < 160


def test_refine_to_name_label_falls_back_when_no_dark_region():
    img = np.full((200, 500), 200, dtype=np.uint8)
    cluster = (100, 50, 160, 80)

    assert pdiseg.refine_to_name_label(img, cluster) == cluster


def test_refine_to_name_label_prefers_textured_dark_badge_over_plain_shadow():
    img = np.full((220, 520), 120, dtype=np.uint8)
    img[50:150, 80:210] = 28
    img[60:140, 230:360] = 32
    for x in range(238, 350, 10):
        img[70:130, x : x + 4] = 220
    cluster = (80, 50, 280, 100)

    rx, _, rw, _ = pdiseg.refine_to_name_label(img, cluster)

    assert rx >= 220
    assert rx + rw <= 365


def test_refine_to_name_label_handles_stale_autoreload_helper(monkeypatch):
    from pdiseg.detection import postprocess

    img = np.full((220, 520), 120, dtype=np.uint8)
    img[60:140, 230:360] = 32
    for x in range(238, 350, 10):
        img[70:130, x : x + 4] = 220

    monkeypatch.delattr(postprocess, "_projection_split_boxes")

    box = pdiseg.refine_to_name_label(img, (210, 50, 170, 100))

    assert box[2] > 0 and box[3] > 0


def test_postprocess_rejects_text_on_light_brand_badge():
    img = np.full((240, 420), 125, dtype=np.uint8)
    img[80:160, 120:300] = 215
    for x in range(130, 290, 12):
        img[90:150, x : x + 4] = 25

    labels, _, kept = pdiseg.postprocess_boxes(
        img, img, [(120, 80, 180, 80)], pdiseg.DetectionConfig()
    )

    assert labels == []
    assert kept == []


def test_postprocess_rejects_plain_dark_clutter():
    img = np.full((240, 420), 125, dtype=np.uint8)
    img[80:160, 120:300] = 25

    labels, _, kept = pdiseg.postprocess_boxes(
        img, img, [(120, 80, 180, 80)], pdiseg.DetectionConfig()
    )

    assert labels == []
    assert kept == []


def test_postprocess_keeps_dark_badge_with_bright_glyphs():
    img = np.full((240, 420), 125, dtype=np.uint8)
    img[80:160, 120:300] = 25
    for x in range(132, 288, 12):
        img[92:148, x : x + 4] = 225

    labels, _, kept = pdiseg.postprocess_boxes(
        img, img, [(120, 80, 180, 80)], pdiseg.DetectionConfig()
    )

    assert len(kept) == 1
    assert len(labels) == 1


def test_product_badge_without_brand_context_still_emits_padded_cluster():
    img = np.full((240, 420), 125, dtype=np.uint8)
    img[90:150, 140:300] = 25
    for x in range(152, 288, 12):
        img[102:138, x : x + 4] = 225

    labels, _, kept = pdiseg.postprocess_boxes(
        img, img, [(140, 90, 160, 60)], pdiseg.DetectionConfig()
    )

    assert len(kept) == 1
    assert len(labels) == 1
    x, y, w, h = labels[0]
    assert x <= 140 and y <= 90
    assert x + w >= 300 and y + h >= 150


def test_postprocess_expands_product_anchor_to_adjacent_brand_context():
    img = np.full((240, 420), 125, dtype=np.uint8)
    img[60:110, 110:190] = 210
    for x in range(120, 184, 12):
        img[70:100, x : x + 3] = 30
    img[105:165, 150:310] = 25
    for x in range(162, 300, 12):
        img[118:152, x : x + 4] = 225

    labels, _, kept = pdiseg.postprocess_boxes(
        img,
        img,
        [(105, 58, 210, 110), (150, 105, 160, 60), (110, 60, 80, 50)],
        pdiseg.DetectionConfig(),
    )

    assert len(kept) == 1
    assert len(labels) == 1
    x, y, w, h = labels[0]
    assert x <= 110 and y <= 60
    assert x + w >= 310 and y + h >= 165


def test_postprocess_does_not_merge_unrelated_text_into_cluster():
    img = np.full((240, 420), 125, dtype=np.uint8)
    img[30:80, 30:110] = 210
    for x in range(40, 100, 12):
        img[40:70, x : x + 3] = 30
    img[100:160, 220:340] = 25
    for x in range(232, 328, 12):
        img[112:148, x : x + 4] = 225

    labels, _, kept = pdiseg.postprocess_boxes(
        img, img, [(30, 30, 80, 50), (220, 100, 120, 60)], pdiseg.DetectionConfig()
    )

    assert len(kept) == 1
    assert len(labels) == 1
    x, _, w, _ = labels[0]
    assert x > 120
    assert x + w >= 340


def test_postprocess_defaults_to_one_primary_cluster():
    img = np.full((240, 520), 125, dtype=np.uint8)
    for left in (80, 300):
        img[90:150, left : left + 140] = 25
        for x in range(left + 12, left + 128, 12):
            img[102:138, x : x + 4] = 225

    labels, _, kept = pdiseg.postprocess_boxes(
        img,
        img,
        [(80, 90, 140, 60), (300, 90, 140, 60)],
        pdiseg.DetectionConfig(),
    )

    assert len(kept) == 1
    assert len(labels) == 1


def test_group_label_fragments_merges_overlapping_words():
    features = {
        "bright_on_dark": 0.12,
        "background_level": 70.0,
        "edge_density": 0.45,
        "extent": 0.58,
        "area_frac": 0.02,
        "aspect": 1.6,
        "frame_area": 120_000.0,
    }
    fragments = [
        pdiseg.ScoredCandidate((100, 80, 80, 50), 0.80, features),
        pdiseg.ScoredCandidate((170, 82, 80, 50), 0.78, features),
    ]

    grouped = pdiseg.group_label_fragments(
        fragments, pdiseg.DetectionConfig(), frame_size=(300, 400)
    )

    assert grouped == [(100, 80, 150, 52)]


def test_group_label_fragments_keeps_separate_package_labels():
    features = {
        "bright_on_dark": 0.12,
        "background_level": 70.0,
        "edge_density": 0.45,
        "extent": 0.58,
        "area_frac": 0.02,
        "aspect": 1.6,
        "frame_area": 120_000.0,
    }
    fragments = [
        pdiseg.ScoredCandidate((80, 80, 70, 50), 0.80, features),
        pdiseg.ScoredCandidate((280, 80, 70, 50), 0.78, features),
    ]

    grouped = pdiseg.group_label_fragments(
        fragments, pdiseg.DetectionConfig(), frame_size=(300, 400)
    )

    assert len(grouped) == 2


def test_keep_label_clusters_keeps_a_label_sized_box():
    label = (200, 150, 150, 100)
    assert pdiseg.keep_label_clusters([label]) == [label]


def test_keep_label_clusters_rejects_oversized_box():
    ssa_box = (0, 0, 950, 700)
    assert pdiseg.keep_label_clusters([ssa_box]) == []


def test_keep_label_clusters_rejects_elongated_box():
    barcode = (100, 100, 400, 30)
    assert pdiseg.keep_label_clusters([barcode]) == []


def test_keep_label_clusters_rejects_tiny_box():
    speck = (10, 10, 30, 30)
    assert pdiseg.keep_label_clusters([speck]) == []


def test_keep_label_clusters_filters_a_mixed_list():
    label = (200, 150, 150, 100)
    candidates = [
        label,
        (0, 0, 950, 700),
        (100, 100, 400, 30),
        (10, 10, 30, 30),
    ]
    assert pdiseg.keep_label_clusters(candidates) == [label]


def test_detect_name_labels_finds_a_label_on_a_raw_frame(tmp_path=None):

    frame = _gradient_frame(300, 500)
    _draw_text_block(frame, 200, 100, 360, 180)

    boxes = pdiseg.detect_name_labels(frame)

    assert len(boxes) >= 1
    for x, y, w, h in boxes:
        assert x >= 0 and y >= 0 and x + w <= 500 and y + h <= 300


def test_preprocess_masks_the_fps_overlay_region():
    img = np.full((100, 300), 50, dtype=np.uint8)
    x, y, w, h = pdiseg.FPS_OVERLAY_REGION
    img[y : y + h, x : x + w] = 255

    out = pdiseg.preprocess(img)

    region = out[y : y + h, x : x + w]

    assert region.min() == region.max()
    assert out.shape == img.shape


def test_preprocess_equalizes_histogram_to_spread_contrast():

    ramp = np.linspace(100, 130, 256).astype(np.uint8)
    img = np.tile(ramp, (300, 1))

    out = pdiseg.preprocess(img)

    _, _, w, _ = pdiseg.FPS_OVERLAY_REGION
    in_range = int(img[:, w:].max()) - int(img[:, w:].min())
    out_range = int(out[:, w:].max()) - int(out[:, w:].min())
    assert out_range > in_range


def test_preprocess_removes_salt_noise_with_median():

    col = np.linspace(0, 255, 300).astype(np.uint8)
    img = np.repeat(col[:, None], 300, axis=1)
    img[150, 250] = 255

    out = pdiseg.preprocess(img)

    assert out[150, 250] == out[150, 240]


def test_find_source_images_discovers_images_and_ignores_other_files(tmp_path):
    dataset = tmp_path / "dataset"
    _write_gray(dataset / "ClassA" / "a.jpg")
    _write_gray(dataset / "ClassA" / "b.png")
    _write_gray(dataset / "ClassB" / "c.jpeg")
    (dataset / "ClassA" / "notes.txt").write_text("not an image")
    _write_gray(dataset / "loose.jpg")

    found = pdiseg.find_source_images(dataset)

    assert {p.name for p in found} == {"a.jpg", "b.png", "c.jpeg"}


def test_find_source_images_ignores_zone_identifier_sidecars(tmp_path):

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
    src = dataset / "Peito_Congelado" / "img001.png"
    src.parent.mkdir(parents=True)
    frame = _gradient_frame(300, 500)
    _draw_text_block(frame, 200, 100, 360, 180)
    iio.imwrite(src, frame)
    out = tmp_path / "result"

    result = subprocess.run(
        [sys.executable, "-m", "pdiseg", str(dataset), str(out)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "Peito_Congelado" / "img001_segmented_1.png").exists()
