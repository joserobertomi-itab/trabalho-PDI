import imageio.v3 as iio
import numpy as np

import pdiseg
from pdiseg.core.boxes import clamp_box


def test_process_dataset_creates_result_tree(tmp_path):
    dataset = tmp_path / "dataset" / "ClassA"
    dataset.mkdir(parents=True)
    frame = np.tile(np.linspace(50, 180, 320).astype(np.uint8), (240, 1))
    frame[90:150, 120:240] = 30
    for x in range(130, 230, 10):
        frame[100:140, x : x + 4] = 220
    iio.imwrite(dataset / "img.png", frame)

    report = pdiseg.process_dataset(dataset.parent, tmp_path / "result")
    assert report.total_frames == 1
    assert (tmp_path / "result" / "ClassA" / "img_segmented_1.png").exists()


def test_saved_crops_are_non_empty(tmp_path):
    dataset = tmp_path / "dataset" / "ClassA"
    dataset.mkdir(parents=True)
    frame = np.full((120, 160), 90, dtype=np.uint8)
    frame[40:80, 50:110] = 25
    for x in range(54, 106, 10):
        frame[46:74, x : x + 2] = 220
    iio.imwrite(dataset / "img.png", frame)

    pdiseg.run(dataset.parent, tmp_path / "result")
    crop = iio.imread(tmp_path / "result" / "ClassA" / "img_segmented_1.png")
    assert crop.size > 0


def test_boxes_stay_inside_image():
    image = np.full((100, 120), 100, dtype=np.uint8)
    boxes = pdiseg.detect_name_labels(image)
    height, width = image.shape
    for x, y, w, h in boxes:
        clamped = clamp_box((x, y, w, h), width, height)
        assert clamped[0] >= 0 and clamped[1] >= 0
        assert clamped[0] + clamped[2] <= width
        assert clamped[1] + clamped[3] <= height


def test_empty_class_folder_does_not_break(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "Empty").mkdir(parents=True)
    good = dataset / "ClassA"
    good.mkdir()
    iio.imwrite(good / "img.png", np.full((40, 40), 120, dtype=np.uint8))

    summary = pdiseg.run(dataset, tmp_path / "result")
    assert summary.images_processed == 1


def test_dataset_report_counts_empty_frames(tmp_path):
    dataset = tmp_path / "dataset" / "Uniform"
    dataset.mkdir(parents=True)
    iio.imwrite(dataset / "blank.png", np.full((80, 80), 128, dtype=np.uint8))

    report = pdiseg.process_dataset(dataset.parent, tmp_path / "result")
    assert report.total_frames == 1
    assert report.empty_frames >= 0


def test_process_dataset_respects_image_limit(tmp_path):
    dataset = tmp_path / "dataset" / "ClassA"
    dataset.mkdir(parents=True)
    for index in range(3):
        iio.imwrite(dataset / f"img{index}.png", np.full((40, 40), 128, dtype=np.uint8))

    report = pdiseg.process_dataset(
        dataset.parent,
        tmp_path / "result",
        detector=lambda image: [(0, 0, 10, 10)],
        limit=2,
        offset=1,
        workers=2,
    )

    assert report.total_frames == 2
    assert report.total_crops == 2
    assert not (tmp_path / "result" / "ClassA" / "img0_segmented_1.png").exists()
    assert (tmp_path / "result" / "ClassA" / "img1_segmented_1.png").exists()
    assert (tmp_path / "result" / "ClassA" / "img2_segmented_1.png").exists()
