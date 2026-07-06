import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from pdiseg.cli.recognize import main as recognize_main
from pdiseg.recognition.batch import (
    aggregate_by_image,
    find_segment_crops,
    predict_images,
    score_dataset,
    summarize,
    sweep_thresholds,
)
from pdiseg.recognition.classify import (
    UNKNOWN,
    classify_features,
    classify_segment,
    load_templates,
)
from pdiseg.recognition.config import RecognitionConfig
from pdiseg.recognition.features import extract_features
from pdiseg.recognition.matching import match_fraction

CFG = RecognitionConfig()


def _pattern(seed: int, size: int = 160, cell: int = 8) -> np.ndarray:
    """Blocky random pattern with plenty of corners for SIFT."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size // cell, size // cell), dtype=np.uint8)
    return np.kron(base, np.ones((cell, cell), dtype=np.uint8))


def test_extract_features_returns_none_on_flat_image():
    flat = np.full((60, 60), 128, dtype=np.uint8)
    assert extract_features(flat, CFG) is None


def test_extract_features_rejects_unknown_descriptor():
    with pytest.raises(ValueError):
        extract_features(_pattern(1), RecognitionConfig(descriptor="surf"))


def test_match_fraction_separates_true_and_false_pairs():
    frame_a, frame_b = _pattern(1), _pattern(2)
    template = extract_features(frame_a[20:120, 20:120], CFG)
    desc_a = extract_features(frame_a, CFG)
    desc_b = extract_features(frame_b, CFG)
    assert template is not None and desc_a is not None and desc_b is not None
    assert match_fraction(template, desc_a, CFG) > 0.5
    assert match_fraction(template, desc_b, CFG) < 0.1


def _write_templates(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    iio.imwrite(root / "ClassA.png", _pattern(1)[20:120, 20:120])
    iio.imwrite(root / "ClassB.png", _pattern(2)[30:130, 10:110])


def test_classify_segment_picks_the_right_template_or_unknown(tmp_path):
    _write_templates(tmp_path / "templates")
    templates = load_templates(tmp_path / "templates", CFG)

    prediction = classify_segment(_pattern(1), templates, CFG)
    assert prediction.label == "ClassA"
    assert prediction.score > CFG.min_match_frac

    stranger = classify_segment(_pattern(99), templates, CFG)
    assert stranger.label == UNKNOWN

    assert classify_features(None, templates, CFG).label == UNKNOWN


def test_load_templates_rejects_empty_dir_and_flat_templates(tmp_path):
    with pytest.raises(ValueError):
        load_templates(tmp_path, CFG)
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    iio.imwrite(flat_dir / "ClassX.png", np.full((60, 60), 128, dtype=np.uint8))
    with pytest.raises(ValueError):
        load_templates(flat_dir, CFG)


def _write_segments(root: Path) -> None:
    """Two classes, both crop suffix spellings, plus a non-segment file to skip."""
    (root / "ClassA").mkdir(parents=True)
    (root / "ClassB").mkdir(parents=True)
    iio.imwrite(root / "ClassA" / "img1_segmented_1.png", _pattern(1))
    iio.imwrite(root / "ClassA" / "img1_segmented_2.png", np.full((60, 60), 128, dtype=np.uint8))
    iio.imwrite(root / "ClassB" / "img2_segmentada_1.png", _pattern(2))
    iio.imwrite(root / "ClassB" / "notes.png", _pattern(3))


def test_find_segment_crops_matches_both_suffixes_only(tmp_path):
    _write_segments(tmp_path)
    names = [p.name for p in find_segment_crops(tmp_path)]
    assert names == ["img1_segmented_1.png", "img1_segmented_2.png", "img2_segmentada_1.png"]


def test_batch_pipeline_aggregates_predicts_and_sweeps(tmp_path):
    _write_templates(tmp_path / "templates")
    templates = load_templates(tmp_path / "templates", CFG)
    _write_segments(tmp_path / "result")

    scores = score_dataset(tmp_path / "result", templates, CFG, workers=2)
    aggregated = aggregate_by_image(scores)
    assert [(a.class_name, a.source_stem) for a in aggregated] == [
        ("ClassA", "img1"),
        ("ClassB", "img2"),
    ]

    predictions = predict_images(aggregated, CFG)
    assert [p.predicted for p in predictions] == ["ClassA", "ClassB"]
    assert all(p.correct for p in predictions)
    assert summarize(predictions).false_positives == 0

    rows = sweep_thresholds(aggregated, CFG, thresholds=[0.0, 1.0])
    assert rows[0][1].unknown == 0
    assert rows[1][1].unknown == 2  # nothing clears a 100% match requirement


def test_cli_writes_predictions_and_sweep_csv(tmp_path, capsys):
    _write_templates(tmp_path / "templates")
    _write_segments(tmp_path / "result")
    sweep_csv = tmp_path / "sweep.csv"

    recognize_main(
        [
            str(tmp_path / "result"),
            str(tmp_path / "templates"),
            "--sweep-csv",
            str(sweep_csv),
            "--progress-every",
            "0",
        ]
    )

    out = capsys.readouterr().out
    assert "accuracy 100.0%" in out

    with (tmp_path / "result" / "recognition.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["predicted"] for row in rows} == {"ClassA", "ClassB"}
    assert all(row["correct"] == "1" for row in rows)
    assert sweep_csv.exists()
