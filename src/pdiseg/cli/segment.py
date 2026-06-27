"""CLI entry: segment a dataset directory tree."""

from __future__ import annotations

import argparse

from pdiseg.core.acceleration import log_acceleration_once
from pdiseg.runtime.env import env_int
from pdiseg.runtime.pipeline import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pdiseg",
        description="Segment poultry packaging labels in a dataset folder tree.",
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
        default="result",
        help="Where to write <Class>/<source>_segmented_<N>.png crops.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images, useful for safe batch runs.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N sorted images before applying --limit.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N images to stderr; use 0 to disable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=env_int("PDISEG_WORKERS", 1),
        help="Number of images to process concurrently (default: PDISEG_WORKERS or 1).",
    )
    args = parser.parse_args(argv)
    log_acceleration_once()

    summary = run(
        args.input_root,
        args.output_root,
        limit=args.limit,
        offset=args.offset,
        progress_every=args.progress_every,
        workers=args.workers,
    )
    print(
        f"Processed {summary.images_processed} images, "
        f"wrote {summary.crops_written} crops to {args.output_root}"
    )


if __name__ == "__main__":
    main()
