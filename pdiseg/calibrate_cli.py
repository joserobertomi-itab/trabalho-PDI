"""Command-line entry point for the calibration harness (issue #6):
``python -m pdiseg.calibrate_cli [INPUT_ROOT] [OUTPUT_DIR]``.

Kept in its own module (not imported by ``pdiseg/__init__``) so running it with
``-m`` does not re-import an already-loaded submodule (runpy RuntimeWarning).
"""

from __future__ import annotations

import argparse

from .calibrate import calibrate


def main(argv: list[str] | None = None) -> None:
    """Run the calibration harness over the base, write overlays + stats.csv under
    OUTPUT_DIR, and print the per-class funnel plus totals."""
    parser = argparse.ArgumentParser(
        prog="pdiseg.calibrate_cli",
        description="Run the calibration harness: overlays + per-class stats (issue #6).",
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default="data/Train_and_Validation",
        help="Dataset root containing <Class>/ image folders.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="calibration",
        help="Where to write <Class>/<stem>_overlay.png and stats.csv.",
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=3,
        help="How many sample overlays to write per class (default: 3).",
    )
    args = parser.parse_args(argv)

    stats = calibrate(
        args.input_root, args.output_dir, per_class_limit=args.per_class_limit
    )

    print(f"{'class_name':52} {'frames':>6} {'cand':>6} {'kept':>6} {'labels':>6}")
    for s in stats:
        print(
            f"{s.class_name:52} {s.frames:6} {s.candidates:6} {s.kept:6} {s.labels:6}"
        )
    total = lambda field: sum(getattr(s, field) for s in stats)
    print(
        f"{'TOTAL':52} {total('frames'):6} {total('candidates'):6} "
        f"{total('kept'):6} {total('labels'):6}"
    )
    frames = total("frames")
    if frames:
        print(f"per-frame labels avg = {total('labels') / frames:.2f}")
    print(f"overlays + stats.csv written to {args.output_dir}/")


if __name__ == "__main__":
    main()
