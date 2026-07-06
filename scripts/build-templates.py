"""Bootstrap templates/<Class>.png from the T1 detector's label-cluster crop.

Picks, per class, the first dataset frame whose detected label crop yields a
feature-rich SIFT template. These bootstrap crops stand in for the manual
template curation required by the T2 brief and can be replaced by hand-made
crops at any time (the recognizer only reads the directory).

Usage: uv run python scripts/build-templates.py [dataset_root] [templates_root]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio

from pdiseg.core.imaging import crop
from pdiseg.detection.detector import detect_name_labels
from pdiseg.detection.postprocess import refine_to_name_label
from pdiseg.io.dataset import IMAGE_SUFFIXES, load_image
from pdiseg.recognition.config import RecognitionConfig
from pdiseg.recognition.features import extract_descriptors

MIN_TEMPLATE_KEYPOINTS = 60


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", default="data/Train_and_Validation")
    parser.add_argument("templates_root", nargs="?", default="templates")
    parser.add_argument("--max-tries", type=int, default=15, help="Frames tried per class.")
    args = parser.parse_args(argv)

    config = RecognitionConfig()
    templates_root = Path(args.templates_root)
    templates_root.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(p for p in Path(args.dataset_root).iterdir() if p.is_dir()):
        target = templates_root / f"{class_dir.name}.png"
        if target.exists():
            print(f"keep    {target}")
            continue
        frames = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        best: tuple[int, Path, object] | None = None
        for frame_path in frames[: args.max_tries]:
            image = load_image(frame_path)
            labels = detect_name_labels(image)
            if not labels:
                continue
            # Small discriminative region (T2 brief): the product-name anchor,
            # not the whole label cluster (brand/table text matches every class).
            template = crop(image, refine_to_name_label(image, labels[0]))
            descriptors = extract_descriptors(template, config)
            if descriptors is None or len(descriptors) < MIN_TEMPLATE_KEYPOINTS:
                continue
            if best is None or len(descriptors) > best[0]:
                best = (len(descriptors), frame_path, template)
        if best is None:
            print(f"FAILED  {class_dir.name}: no usable template found", file=sys.stderr)
            continue
        keypoints, frame_path, template = best
        iio.imwrite(target, template)
        print(f"wrote   {target}  ({keypoints} keypoints, from {frame_path.name})")


if __name__ == "__main__":
    main()
