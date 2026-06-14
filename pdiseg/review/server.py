"""Thin HTTP layer for the read-only review viewer."""

from __future__ import annotations

import io
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from numpy.typing import NDArray

from pdiseg.imaging import FrameInspection, crop, render_overlay
from pdiseg.review.model import (
    FrameReview,
    ReviewBundle,
    get_frame,
    label_box,
    list_classes,
    list_frames,
    totals,
)

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(bundle: ReviewBundle) -> FastAPI:
    app = FastAPI(title="PDI Seg Review Viewer", docs_url=None, redoc_url=None)
    app.state.bundle = bundle

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/classes")
    def api_classes() -> dict[str, object]:
        classes = list_classes(bundle)
        total = totals(bundle)
        return {
            "classes": [
                {
                    "name": row.class_name,
                    "frames": row.frames,
                    "candidates": row.candidates,
                    "kept": row.kept,
                    "labels": row.labels,
                }
                for row in classes
            ],
            "totals": {
                "frames": total.frames,
                "candidates": total.candidates,
                "kept": total.kept,
                "labels": total.labels,
            },
        }

    @app.get("/api/frames")
    def api_frames(
        class_name: str = Query(...),
        min_labels: int = Query(0, ge=0),
        only_rejected: bool = Query(False),
    ) -> dict[str, object]:
        frames = list_frames(
            bundle,
            class_name,
            min_labels=min_labels,
            only_rejected=only_rejected,
        )
        return {"frames": [_frame_payload(frame) for frame in frames]}

    @app.get("/api/frame/{class_name}/{stem}")
    def api_frame(class_name: str, stem: str) -> dict[str, object]:
        frame = get_frame(bundle, class_name, stem)
        if frame is None:
            raise HTTPException(status_code=404, detail="frame not found")
        return _frame_payload(frame, include_crops=True)

    @app.get("/media/source/{class_name}/{stem}")
    def media_source(class_name: str, stem: str) -> Response:
        image = _read_source(bundle, class_name, stem)
        return _png_response(image)

    @app.get("/media/overlay/{class_name}/{stem}")
    def media_overlay(class_name: str, stem: str) -> Response:
        frame = _require_frame(bundle, class_name, stem)
        image = _read_source(bundle, class_name, stem)
        if frame.boxes is None:
            raise HTTPException(status_code=404, detail="box metadata missing for frame")
        overlay = render_overlay(image, frame.boxes)
        return _png_response(overlay)

    @app.get("/media/crop/{class_name}/{stem}/{index}")
    def media_crop(class_name: str, stem: str, index: int) -> Response:
        if index < 1:
            raise HTTPException(status_code=400, detail="crop index is 1-based")
        frame = _require_frame(bundle, class_name, stem)
        if index <= len(frame.crop_paths):
            crop_image = iio.imread(frame.crop_paths[index - 1])
            return _png_response(crop_image)
        bbox = label_box(frame, index)
        if bbox is None:
            raise HTTPException(status_code=404, detail="crop not available")
        source = _read_source(bundle, class_name, stem)
        return _png_response(crop(source, bbox))

    return app


def _require_frame(bundle: ReviewBundle, class_name: str, stem: str) -> FrameReview:
    frame = get_frame(bundle, class_name, stem)
    if frame is None:
        raise HTTPException(status_code=404, detail="frame not found")
    return frame


def _read_source(bundle: ReviewBundle, class_name: str, stem: str) -> NDArray[np.uint8]:
    frame = _require_frame(bundle, class_name, stem)
    path = bundle.dataset_root / frame.rel_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="source image missing")
    image = iio.imread(path)
    if image.ndim > 2:
        image = image[..., 0]
    return image.astype(np.uint8)


def _frame_payload(frame: FrameReview, *, include_crops: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "class_name": frame.class_name,
        "stem": frame.stem,
        "rel_path": frame.rel_path,
        "source_exists": frame.source_exists,
        "has_boxes": frame.boxes is not None,
        "candidate_count": frame.candidate_count,
        "kept_count": frame.kept_count,
        "label_count": frame.label_count,
        "rejected_count": frame.rejected_count,
        "crop_count": len(frame.crop_paths),
        "source_url": f"/media/source/{frame.class_name}/{frame.stem}",
        "overlay_url": (f"/media/overlay/{frame.class_name}/{frame.stem}" if frame.boxes else None),
    }
    if include_crops:
        count = max(frame.label_count, len(frame.crop_paths))
        payload["crops"] = [
            {
                "index": index,
                "url": f"/media/crop/{frame.class_name}/{frame.stem}/{index}",
                "from_disk": index <= len(frame.crop_paths),
            }
            for index in range(1, count + 1)
        ]
        if frame.boxes:
            payload["boxes"] = _boxes_payload(frame.boxes)
    return payload


def _boxes_payload(inspection: FrameInspection) -> dict[str, object]:
    kept_set = set(inspection.kept)
    return {
        "candidates": [list(box) for box in inspection.candidates],
        "kept": [list(box) for box in inspection.kept],
        "labels": [list(box) for box in inspection.labels],
        "rejected": [list(box) for box in inspection.candidates if box not in kept_set],
    }


def _png_response(image: NDArray[np.uint8]) -> StreamingResponse:
    buffer = io.BytesIO()
    iio.imwrite(buffer, image, extension=".png")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
