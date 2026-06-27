"""CLI entry: run full pipeline on a small debug sample."""

from __future__ import annotations

import argparse

from pdiseg.debug.sample import run_debug_sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pdiseg-debug",
        description=(
            "Run the production segmentation pipeline on a small sample "
            "(default: one image per class) and write debug_result/ artifacts."
        ),
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default="data/Train_and_Validation",
        help="Dataset root containing <Class>/ image folders.",
    )
    parser.add_argument(
        "result_root",
        nargs="?",
        default="debug_result/result",
        help="Where to write segmented crops (mirrors production result/ layout).",
    )
    parser.add_argument(
        "--bundle-root",
        default="debug_result/bundles",
        help="Per-frame mask/overlay bundles (default: debug_result/bundles).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="How many images to process per class folder (default: 1).",
    )
    args = parser.parse_args(argv)

    report = run_debug_sample(
        args.input_root,
        args.result_root,
        bundle_root=args.bundle_root,
        per_class=args.per_class,
    )
    ds = report.dataset_report
    print(
        f"Debug sample: {ds.total_frames} frames, {ds.total_crops} crops, "
        f"{ds.empty_frames} empty"
    )
    print(f"Crops written to {report.result_root}/")
    print(f"Bundles written to {report.bundle_root}/")
    if ds.classes:
        print(f"{'class':52} {'frames':>6} {'crops':>6} {'empty':>6}")
        for row in ds.classes:
            print(f"{row.class_name:52} {row.frames:6} {row.crops:6} {row.empty_frames:6}")


if __name__ == "__main__":
    main()
