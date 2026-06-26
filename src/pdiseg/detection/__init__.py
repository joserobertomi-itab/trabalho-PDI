"""Classical CV pipeline: preprocess → masks → candidates → score → postprocess."""

from pdiseg.detection.config import DetectionConfig
from pdiseg.detection.detector import detect_name_labels, inspect_detection

__all__ = ["DetectionConfig", "detect_name_labels", "inspect_detection"]
