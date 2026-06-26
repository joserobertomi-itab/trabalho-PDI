"""CLI entry: segment a dataset directory tree."""

from __future__ import annotations

import argparse

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
        help="Where to write <Class>/<source>_segmentada_<N>.png crops.",
    )
    args = parser.parse_args(argv)

    summary = run(args.input_root, args.output_root)
    print(
        f"Processed {summary.images_processed} images, "
        f"wrote {summary.crops_written} crops to {args.output_root}"
    )


if __name__ == "__main__":
    main()
