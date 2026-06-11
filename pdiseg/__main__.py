"""Command-line entry point: ``python -m pdiseg [INPUT_ROOT] [OUTPUT_ROOT]``."""

from __future__ import annotations

import argparse

from .pipeline import detect_clusters, keep_label_clusters, run


def _detect_labels(image):
    """Stage 1 (locate clusters) followed by Stage 3 (geometric FP rejection)."""
    return keep_label_clusters(detect_clusters(image))


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
