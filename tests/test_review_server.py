from pathlib import Path

import imageio.v3 as iio
import numpy as np
from fastapi.testclient import TestClient

from pdiseg.review.model import load_bundle
from pdiseg.review.server import create_app


def _write_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"
    class_dir = dataset / "ClassA"
    class_dir.mkdir(parents=True)
    frame = np.full((80, 120), 90, dtype=np.uint8)
    iio.imwrite(class_dir / "f1.png", frame)

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "boxes.json").write_text(
        """
        {
          "ClassA/f1.png": {
            "candidates": [[10, 10, 40, 30]],
            "kept": [[10, 10, 40, 30]],
            "labels": [[12, 12, 20, 18]]
          }
        }
        """,
        encoding="utf-8",
    )
    (calibration / "stats.csv").write_text(
        "class_name,frames,candidates,kept,labels\nClassA,1,1,1,1\n",
        encoding="utf-8",
    )
    class_result = tmp_path / "result" / "ClassA"
    class_result.mkdir(parents=True)
    iio.imwrite(class_result / "f1_segmentada_1.png", frame[10:50, 10:50])
    return dataset, calibration, tmp_path / "result"


def test_review_api_and_media_endpoints(tmp_path):
    dataset, calibration, result = _write_bundle(tmp_path)
    app = create_app(load_bundle(dataset, calibration, result))
    client = TestClient(app)

    index = client.get("/")
    assert index.status_code == 200
    assert "Review Viewer" in index.text

    classes = client.get("/api/classes").json()
    assert classes["classes"][0]["name"] == "ClassA"
    assert classes["totals"]["frames"] == 1

    frames = client.get("/api/frames", params={"class_name": "ClassA"}).json()
    assert frames["frames"][0]["stem"] == "f1"

    source = client.get("/media/source/ClassA/f1")
    assert source.status_code == 200
    assert source.headers["content-type"] == "image/png"

    overlay = client.get("/media/overlay/ClassA/f1")
    assert overlay.status_code == 200

    crop = client.get("/media/crop/ClassA/f1/1")
    assert crop.status_code == 200


def test_review_returns_404_for_unknown_frame(tmp_path):
    dataset, calibration, result = _write_bundle(tmp_path)
    app = create_app(load_bundle(dataset, calibration, result))
    client = TestClient(app)

    missing = client.get("/api/frame/ClassA/missing")
    assert missing.status_code == 404

    overlay = client.get("/media/overlay/ClassA/missing")
    assert overlay.status_code == 404


def test_review_renders_crop_from_boxes_when_result_crop_missing(tmp_path):
    dataset = tmp_path / "dataset"
    class_dir = dataset / "ClassA"
    class_dir.mkdir(parents=True)
    frame = np.full((60, 80), 100, dtype=np.uint8)
    iio.imwrite(class_dir / "f1.png", frame)

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "boxes.json").write_text(
        '{"ClassA/f1.png": {"candidates": [[5,5,20,20]], "kept": [[5,5,20,20]], "labels": [[6,6,10,10]]}}',
        encoding="utf-8",
    )

    app = create_app(load_bundle(dataset, calibration, None))
    client = TestClient(app)
    crop = client.get("/media/crop/ClassA/f1/1")
    assert crop.status_code == 200

    missing_overlay_dir = tmp_path / "no_boxes"
    missing_overlay_dir.mkdir()
    (missing_overlay_dir / "stats.csv").write_text(
        "class_name,frames,candidates,kept,labels\n", encoding="utf-8"
    )
    app2 = create_app(load_bundle(dataset, missing_overlay_dir, None))
    client2 = TestClient(app2)
    assert client2.get("/media/overlay/ClassA/f1").status_code == 404
