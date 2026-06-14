"""Read-only review viewer for segmentation artifacts (outside the graded pipeline)."""

from .model import ReviewBundle, load_bundle

__all__ = ["ReviewBundle", "load_bundle"]
