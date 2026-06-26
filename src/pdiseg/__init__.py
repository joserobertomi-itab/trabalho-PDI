"""Poultry packaging name-label segmentation (classical PDI, IFG).

Public API re-exports stable symbols for notebooks, tests, and scripts.
Internal layout is split under :mod:`pdiseg.core`, :mod:`pdiseg.detection`,
:mod:`pdiseg.runtime`, etc. See ``docs/src/ARCHITECTURE.md``.
"""

from pdiseg.calibration.service import ClassStats, calibrate
from pdiseg.core.imaging import BBox, FrameInspection, crop, render_overlay
from pdiseg.debug.viz import (
    debug_frame,
    draw_boxes,
    save_debug_bundle,
    scored_table,
    to_rgb,
    visualize_masks,
)
from pdiseg.detection.candidates import detect_clusters, find_candidate_boxes
from pdiseg.detection.config import DetectionConfig
from pdiseg.detection.detector import (
    DetectionResult,
    detect,
    detect_name_labels,
    inspect_detection,
    inspect_frame,
)
from pdiseg.detection.masks import build_candidate_masks
from pdiseg.detection.postprocess import (
    keep_label_clusters,
    postprocess_boxes,
    refine_to_name_label,
)
from pdiseg.detection.preprocess import preprocess, preprocess_image
from pdiseg.detection.scoring import ScoredCandidate, score_candidate
from pdiseg.io.dataset import (
    FPS_OVERLAY_REGION,
    count_images_per_class,
    find_source_images,
    list_classes,
    load_image,
)
from pdiseg.runtime.pipeline import (
    ClassReport,
    DatasetReport,
    RunSummary,
    crop_and_save,
    dump_preprocessed,
    output_path,
    process_dataset,
    run,
)

__all__ = [
    "FPS_OVERLAY_REGION",
    "BBox",
    "ClassReport",
    "ClassStats",
    "DatasetReport",
    "DetectionConfig",
    "DetectionResult",
    "FrameInspection",
    "RunSummary",
    "ScoredCandidate",
    "build_candidate_masks",
    "calibrate",
    "count_images_per_class",
    "crop",
    "crop_and_save",
    "debug_frame",
    "detect",
    "detect_clusters",
    "detect_name_labels",
    "draw_boxes",
    "dump_preprocessed",
    "find_candidate_boxes",
    "find_source_images",
    "inspect_detection",
    "inspect_frame",
    "keep_label_clusters",
    "list_classes",
    "load_image",
    "output_path",
    "postprocess_boxes",
    "preprocess",
    "preprocess_image",
    "process_dataset",
    "refine_to_name_label",
    "render_overlay",
    "run",
    "save_debug_bundle",
    "score_candidate",
    "scored_table",
    "to_rgb",
    "visualize_masks",
]
