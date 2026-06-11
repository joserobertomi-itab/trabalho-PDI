"""Command-line entry point: ``python -m pdiseg [INPUT_ROOT] [OUTPUT_ROOT]``."""

from __future__ import annotations

import argparse

from .pipeline import (
    detect_clusters,
    keep_label_clusters,
    refine_to_name_label,
    run,
)


def _detect_labels(image):
    """Locate clusters (Stage 1), reject by geometry (Stage 3), refine to the
    dark name label (Stage 2), each with cluster fallback."""
    clusters = keep_label_clusters(detect_clusters(image))
    return [refine_to_name_label(image, c) for c in clusters]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pdiseg",
        description="Segment poultry-packaging name labels across a dataset.",
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default="data/Train_and_Validation",
        help="Dataset root containing <Class>/ image folders.",
    )
    parser.add_argument(
        "output_root",
        nargs="?",
        default="resultado",
        help="Where to write <Class>/<source>_segmentada_<N>.png crops.",
    )
    args = parser.parse_args(argv)

    summary = run(args.input_root, args.output_root, detector=_detect_labels)
    print(
        f"Processed {summary.images_processed} images, "
        f"wrote {summary.crops_written} crops to {args.output_root}"
    )


if __name__ == "__main__":
    main()
