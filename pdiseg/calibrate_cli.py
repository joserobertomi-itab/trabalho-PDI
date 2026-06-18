from __future__ import annotations

import argparse
from typing import Literal

from .calibrate import calibrate


def main(argv: list[str] | None = None) -> None:

    parser = argparse.ArgumentParser(
        prog="pdiseg.calibrate_cli",
        description="Write overlays, boxes.json and stats.csv for a dataset.",
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

    stats = calibrate(args.input_root, args.output_dir, per_class_limit=args.per_class_limit)

    print(f"{'class_name':52} {'frames':>6} {'cand':>6} {'kept':>6} {'labels':>6}")
    for s in stats:
        print(f"{s.class_name:52} {s.frames:6} {s.candidates:6} {s.kept:6} {s.labels:6}")

    def total(field: Literal["frames", "candidates", "kept", "labels"]) -> int:
        return sum(getattr(row, field) for row in stats)

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
