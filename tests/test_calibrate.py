"""Calibration harness behavior, exercised through the public API (issue #6)."""

import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import pdiseg

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _draw_text_block(img, x0, y0, x1, y1, bg=30, ink=220):
    """Paint a region of bright vertical 'text' strokes on a dark background."""
    img[y0:y1, x0:x1] = bg
    for x in range(x0, x1, 8):
        img[y0:y1, x : x + 3] = ink


def _gradient_frame(h, w):
    """Full-range background so equalize_hist is ~identity (a uniform background
    would collapse after equalization, hiding the drawn block)."""
    return np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))


def test_inspect_frame_breaks_a_raw_frame_into_per_stage_boxes():
    # A raw (un-preprocessed) frame with one label-cluster-sized text block,
    # clear of the FPS overlay region.
    frame = _gradient_frame(300, 500)
    _draw_text_block(frame, 200, 100, 360, 180)

    inspection = pdiseg.inspect_frame(frame)

    # Stage 1 finds candidates; Stage 3 keeps a subset; Stage 2 yields one box
    # per kept cluster. The funnel only narrows.
    assert len(inspection.candidates) >= 1
    assert len(inspection.kept) <= len(inspection.candidates)
    assert len(inspection.labels) == len(inspection.kept)
    # Every box stays inside the frame.
    for x, y, w, h in inspection.labels:
        assert x >= 0 and y >= 0 and x + w <= 500 and y + h <= 300


def test_inspect_frame_labels_match_the_production_detector():
    # The calibration view must not drift from what `run` actually crops, or the
    # overlays would mislead the human. inspect_frame.labels == detect_name_labels.
    frame = _gradient_frame(300, 500)
    _draw_text_block(frame, 200, 100, 360, 180)

    assert pdiseg.inspect_frame(frame).labels == pdiseg.detect_name_labels(frame)


def test_render_overlay_draws_colored_boxes_on_an_rgb_copy():
    frame = np.full((120, 160), 30, dtype=np.uint8)
    inspection = pdiseg.FrameInspection(
        candidates=[(20, 20, 60, 40)], kept=[(20, 20, 60, 40)], labels=[(30, 30, 30, 20)]
    )

    overlay = pdiseg.render_overlay(frame, inspection)

    # An RGB image the size of the frame.
    assert overlay.shape == (120, 160, 3)
    # The overlay carries color the grayscale source could not: at least one pixel
    # has unequal channels (a drawn box edge), so it is not a plain gray copy.
    channels_differ = (overlay[..., 0] != overlay[..., 1]) | (overlay[..., 1] != overlay[..., 2])
    assert channels_differ.any()


def _write_frame_with_block(path: Path, with_block: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _gradient_frame(300, 500)
    if with_block:
        _draw_text_block(frame, 200, 100, 360, 180)  # detectable cluster
    iio.imwrite(path, frame)


def test_calibrate_writes_overlays_and_returns_per_class_stats(tmp_path):
    dataset = tmp_path / "dataset"
    # Two classes; class A has two detectable frames, class B has one blank frame.
    _write_frame_with_block(dataset / "ClassA" / "f1.png", with_block=True)
    _write_frame_with_block(dataset / "ClassA" / "f2.png", with_block=True)
    _write_frame_with_block(dataset / "ClassB" / "f3.png", with_block=False)

    out = tmp_path / "calibration"
    stats = pdiseg.calibrate(dataset, out, per_class_limit=5)

    # One ClassStats per class, sorted by class name, counting every frame.
    by_class = {s.class_name: s for s in stats}
    assert set(by_class) == {"ClassA", "ClassB"}
    assert by_class["ClassA"].frames == 2
    assert by_class["ClassB"].frames == 1
    # Class A detects labels; the funnel relation holds on aggregates.
    a = by_class["ClassA"]
    assert a.candidates >= a.kept == a.labels >= 1

    # Overlay PNGs are written under a per-class folder, one per processed frame.
    assert (out / "ClassA" / "f1_overlay.png").exists()
    assert (out / "ClassA" / "f2_overlay.png").exists()
    assert (out / "ClassB" / "f3_overlay.png").exists()


def test_calibrate_respects_per_class_overlay_limit(tmp_path):
    dataset = tmp_path / "dataset"
    for i in range(4):
        _write_frame_with_block(dataset / "ClassA" / f"f{i}.png", with_block=True)

    out = tmp_path / "calibration"
    stats = pdiseg.calibrate(dataset, out, per_class_limit=2)

    # Every frame is counted in the stats...
    assert stats[0].frames == 4
    # ...but only the first per_class_limit overlays are written.
    overlays = list((out / "ClassA").glob("*_overlay.png"))
    assert len(overlays) == 2


def test_calibrate_writes_boxes_json_with_per_stage_boxes(tmp_path):
    dataset = tmp_path / "dataset"
    _write_frame_with_block(dataset / "ClassA" / "f1.png", with_block=True)

    out = tmp_path / "calibration"
    pdiseg.calibrate(dataset, out)

    import json

    payload = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    assert "ClassA/f1.png" in payload
    entry = payload["ClassA/f1.png"]
    assert set(entry) == {"candidates", "kept", "labels"}
    assert len(entry["labels"]) == len(entry["kept"])
    assert len(entry["candidates"]) >= len(entry["kept"])


def test_calibrate_writes_a_csv_stats_summary(tmp_path):
    dataset = tmp_path / "dataset"
    _write_frame_with_block(dataset / "ClassA" / "f1.png", with_block=True)
    _write_frame_with_block(dataset / "ClassB" / "f2.png", with_block=False)

    out = tmp_path / "calibration"
    pdiseg.calibrate(dataset, out)

    summary = out / "stats.csv"
    assert summary.exists()
    text = summary.read_text()
    # Header plus one row per class, with the funnel columns.
    assert "class_name" in text and "candidates" in text and "labels" in text
    assert "ClassA" in text and "ClassB" in text


def test_calibrate_entry_point_runs_over_given_dirs(tmp_path):
    dataset = tmp_path / "dataset"
    _write_frame_with_block(dataset / "ClassA" / "f1.png", with_block=True)
    out = tmp_path / "calibration"

    result = subprocess.run(
        [sys.executable, "-m", "pdiseg.calibrate_cli", str(dataset), str(out)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    # No runpy re-import warning leaks to stderr.
    assert "RuntimeWarning" not in result.stderr
    assert (out / "stats.csv").exists()
    assert (out / "boxes.json").exists()
    assert (out / "ClassA" / "f1_overlay.png").exists()
    assert "TOTAL" in result.stdout
