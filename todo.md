# T2 roadmap — product recognition via local feature matching

Brief: `enunciadot2.pdf`. Match each T1 segment against one manually-cropped
template per tray class using local descriptors (SIFT/ORB); decision = % of
valid matches; minimize false positives. No ML/AI. Deliverables: code,
templates, PDF report.

No new dependencies: `skimage.feature` already ships `SIFT`, `ORB`,
`BRIEF`/`corner_fast`, and `match_descriptors` (Lowe ratio via `max_ratio`,
plus `cross_check`). FLANN/OpenCV only if brute-force matching proves too
slow — unlikely at this dataset size.

## 0. Groundwork

- [x] ADR `docs/adr/0006-local-feature-recognition.md`: descriptor choice (skimage SIFT vs ORB), ratio test + cross-check, match-percentage threshold rationale, why no OpenCV.

## 1. Inputs

- [x] `templates/<Class>.png` — one manual crop per tray class (discriminative packaging region). Committed.
- [x] Curated segments: manually fix T1 output so every image has one correct segment (brief allows plain manual crops). Layout: `result/<Class>/<stem>_segmented_1.png` (reuse T1 convention) or a `segments/` dir if corrections diverge from `result/`.

## 2. Recognition package — `src/pdiseg/recognition/`

Sibling of `detection/`, same conventions (frozen config dataclass, absolute imports of `pdiseg.core/io`).

- [x] `config.py` — `RecognitionConfig`: descriptor kind, detector params, Lowe `max_ratio`, `cross_check`, min valid-match %.
- [x] `features.py` — keypoints + descriptors via `skimage.feature.SIFT`/`ORB` (grayscale in, arrays out; handle "no keypoints found" case).
- [x] `matching.py` — `skimage.feature.match_descriptors(max_ratio=…, cross_check=True)` → valid-match % per (segment, template).
- [x] `classify.py` — score segment against all templates; best % above threshold wins, else "unknown" (false-positive guard).

## 3. CLI + wiring

- [x] `src/pdiseg/cli/recognize.py` — batch: walk segments, load templates, print/write per-image predicted class + score (CSV/JSON under `result/`).
- [x] Register `pdiseg-recognize` in `pyproject.toml [project.scripts]`; re-export public API in `pdiseg/__init__.py`.
- [x] `Makefile` target `recognize` (help-annotated); compose service under `tools` profile if needed.

## 4. Calibration

- [x] Sweep the valid-match % threshold (and ratio) over the dataset; emit accuracy / confusion / false-positive stats (CSV like `calibration/stats.csv`). Small script or flag on the CLI — no new UI.

## 5. Quality gate

- [x] `tests/test_recognition.py` — matching math, threshold gating, unknown case (synthetic images, no dataset dependency).
- [x] `make check` green (ruff, strict mypy, pytest ≥80% coverage).

## 6. Report (PDF)

- [x] Method, templates shown, calibration curves, accuracy + false-positive analysis, limitations.
