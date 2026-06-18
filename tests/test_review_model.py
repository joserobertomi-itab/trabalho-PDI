from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pdiseg.review.model import get_frame, list_classes, list_frames, load_bundle


def _draw_text_block(img, x0, y0, x1, y1, bg=30, ink=220):
    img[y0:y1, x0:x1] = bg
    for x in range(x0, x1, 8):
        img[y0:y1, x : x + 3] = ink


def _gradient_frame(h, w):
    return np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))


def _write_frame(path: Path, with_block: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _gradient_frame(300, 500)
    if with_block:
        _draw_text_block(frame, 200, 100, 360, 180)
    iio.imwrite(path, frame)


def _write_boxes(path: Path, rel_path: str, labels: list[list[int]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        rel_path: {
            "candidates": labels,
            "kept": labels,
            "labels": labels,
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_bundle_lists_classes_and_frames(tmp_path):
    dataset = tmp_path / "dataset"
    _write_frame(dataset / "ClassA" / "f1.png", with_block=True)
    _write_frame(dataset / "ClassA" / "f2.png", with_block=False)
    calibration = tmp_path / "calibration"
    _write_boxes(calibration / "boxes.json", "ClassA/f1.png", [[210, 110, 140, 60]])
    (calibration / "stats.csv").write_text(
        "class_name,frames,candidates,kept,labels\nClassA,2,2,1,1\n",
        encoding="utf-8",
    )

    bundle = load_bundle(dataset, calibration)
    classes = list_classes(bundle)
    assert [row.class_name for row in classes] == ["ClassA"]
    assert classes[0].frames == 2

    frames = list_frames(bundle, "ClassA")
    assert {frame.stem for frame in frames} == {"f1", "f2"}


def test_list_frames_supports_min_labels_and_only_rejected_filters(tmp_path):
    dataset = tmp_path / "dataset"
    _write_frame(dataset / "ClassA" / "f1.png")
    _write_frame(dataset / "ClassA" / "f2.png")
    calibration = tmp_path / "calibration"
    import json

    calibration.mkdir()
    (calibration / "boxes.json").write_text(
        json.dumps(
            {
                "ClassA/f1.png": {
                    "candidates": [[1, 1, 10, 10], [50, 50, 10, 10]],
                    "kept": [[1, 1, 10, 10]],
                    "labels": [[1, 1, 10, 10]],
                },
                "ClassA/f2.png": {
                    "candidates": [[1, 1, 10, 10]],
                    "kept": [[1, 1, 10, 10]],
                    "labels": [[1, 1, 10, 10], [20, 20, 10, 10]],
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_bundle(dataset, calibration)
    assert len(list_frames(bundle, "ClassA", min_labels=2)) == 1
    assert list_frames(bundle, "ClassA", only_rejected=True)[0].stem == "f1"


def test_get_frame_finds_crops_from_result_when_boxes_missing(tmp_path):
    dataset = tmp_path / "dataset"
    _write_frame(dataset / "ClassA" / "f1.png")
    result_root = tmp_path / "result"
    class_result = result_root / "ClassA"
    class_result.mkdir(parents=True)
    iio.imwrite(class_result / "f1_segmentada_1.png", np.full((8, 8), 200, dtype=np.uint8))

    bundle = load_bundle(dataset, tmp_path / "calibration", result_root)
    frame = get_frame(bundle, "ClassA", "f1")
    assert frame is not None
    assert len(frame.crop_paths) == 1
    assert frame.boxes is None
