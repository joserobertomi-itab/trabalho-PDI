"""CLI entry: recognize products in T1 segment crops via local-feature matching."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdiseg.recognition.batch import (
    aggregate_by_image,
    predict_images,
    score_dataset,
    summarize,
    sweep_thresholds,
    write_predictions_csv,
    write_sweep_csv,
)
from pdiseg.recognition.classify import load_templates
from pdiseg.recognition.config import RecognitionConfig
from pdiseg.runtime.env import env_int


def main(argv: list[str] | None = None) -> None:
    defaults = RecognitionConfig()
    parser = argparse.ArgumentParser(
        prog="pdiseg-recognize",
        description="Match T1 segment crops against per-class templates (SIFT/ORB).",
    )
    parser.add_argument(
        "segments_root",
        nargs="?",
        default="result",
        help="Root containing <Class>/<stem>_segment*_<N>.png crops from T1.",
    )
    parser.add_argument(
        "templates_root",
        nargs="?",
        default="templates",
        help="Directory with one <Class>.png template per class.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Predictions CSV path (default: <segments_root>/recognition.csv).",
    )
    parser.add_argument(
        "--sweep-csv",
        default=None,
        help="Also write a min-match-frac threshold sweep CSV to this path.",
    )
    parser.add_argument("--descriptor", choices=["sift", "orb"], default=defaults.descriptor)
    parser.add_argument("--max-ratio", type=float, default=defaults.max_ratio)
    parser.add_argument("--min-match-frac", type=float, default=defaults.min_match_frac)
    parser.add_argument("--limit", type=int, default=None, help="Score at most N segments.")
    parser.add_argument(
        "--workers",
        type=int,
        default=env_int("PDISEG_WORKERS", 1),
        help="Segments scored concurrently (default: PDISEG_WORKERS or 1).",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)

    config = RecognitionConfig(
        descriptor=args.descriptor,
        max_ratio=args.max_ratio,
        min_match_frac=args.min_match_frac,
    )
    templates = load_templates(args.templates_root, config)
    segment_scores = score_dataset(
        args.segments_root,
        templates,
        config,
        limit=args.limit,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    aggregated = aggregate_by_image(segment_scores)
    predictions = predict_images(aggregated, config)

    csv_path = Path(args.csv) if args.csv else Path(args.segments_root) / "recognition.csv"
    write_predictions_csv(csv_path, predictions)
    if args.sweep_csv:
        write_sweep_csv(args.sweep_csv, sweep_thresholds(aggregated, config))

    s = summarize(predictions)
    print(
        f"Recognized {s.images} images: accuracy {s.accuracy:.1%}, "
        f"{s.unknown} unknown, {s.false_positives} false positives "
        f"(min_match_frac={config.min_match_frac}) -> {csv_path}"
    )


if __name__ == "__main__":
    main()
