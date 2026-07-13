# UPDATES — T2 product recognition

What this branch (`feat/t2-ny-SIFT`) adds on top of `main`. Everything here implements
**T2** (`enunciadot2.pdf`): identify the product class of each T1 segment by matching
hand-engineered **local descriptors** against one curated template per class. No ML/AI;
the decision rests on a single calibration knob (the percentage of valid matches), and
false positives are treated as worse than misses.

Diff summary: **28 files, +1272 / −4**, 9 commits (`4032ffe..b1a4a3f`).

---

## 1. New recognition package — `src/pdiseg/recognition/`

Sibling of `detection/`, same conventions (frozen config dataclass, absolute imports).

| Module | Responsibility |
|--------|----------------|
| `config.py` | `RecognitionConfig` — the tunable knobs (descriptor, Lowe `max_ratio`, `cross_check`, `min_match_frac`, RANSAC params). One shared config, no per-class tuning. |
| `features.py` | `extract_features` / `extract_descriptors` — keypoints + descriptors via `skimage.feature.SIFT` (or `ORB`). Returns `None` for flat/degenerate crops (< `min_keypoints`), which callers read as "no evidence", never a match. |
| `matching.py` | `match_fraction` — `match_descriptors(cross_check, max_ratio)` (Lowe ratio + cross-check) **then RANSAC affine verification**. Valid matches must agree on one transform, so keypoints on words shared by every label (brand, "CONGELADO", the weight-table font) don't form a consensus and are discarded. Score = inliers / template keypoints. |
| `classify.py` | `load_templates`, `classify_segment`, `classify_features` — best-scoring template wins **only if** it clears `min_match_frac`; otherwise `unknown`. |
| `batch.py` | Walk T1 segment crops, score each against every template, aggregate per source image (max score across its segments), gate to a prediction, summarize (accuracy / unknown / false positives), and sweep thresholds over cached scores. CSV writers for predictions and the sweep. Optional thread pool like `calibrate`. |

### Decision logic (precision-first)

1. Extract SIFT keypoints/descriptors for the segment crop.
2. For each template: match descriptors (ratio + cross-check), require ≥ `ransac_min_matches`, run RANSAC affine, count inliers.
3. Score = inliers / template keypoints; best template wins **iff** score ≥ `min_match_frac`, else `unknown`.
4. One source image can have several T1 segments → image-level score is the max per template across its segments.

---

## 2. CLI + wiring

- **`src/pdiseg/cli/recognize.py`** → console script `pdiseg-recognize` (registered in `pyproject.toml`).
  ```sh
  uv run pdiseg-recognize [SEGMENTS_ROOT=result] [TEMPLATES_ROOT=templates] \
      [--descriptor sift|orb] [--max-ratio F] [--min-match-frac F] \
      [--limit N] [--workers N] [--sweep-csv PATH]
  ```
  Writes `result/recognition.csv` (per-image prediction + score + correctness) and, with
  `--sweep-csv`, a threshold sweep. Prints accuracy / unknown / false-positive summary.
- **`src/pdiseg/__init__.py`** — re-exports the public recognition API (`RecognitionConfig`, `Features`, `Template`, `Prediction`, `extract_features`, `match_fraction`, `classify_segment`, `load_templates`, `UNKNOWN`, …).
- **`Makefile`** — new targets `templates`, `recognize`, `report`; also fixed the coverage path (`--cov=src/pdiseg`).
- **`pyproject.toml`** — registers the `pdiseg-recognize` entry point.

---

## 3. Templates — `templates/<Class>.png`

Seven committed templates, one per class, bootstrapped by **`scripts/build-templates.py`**
from the T1 detector's label-cluster crop (first frame per class yielding a feature-rich
SIFT template, ≥ 60 keypoints). These stand in for manual curation and can be replaced by
hand-made crops at any time — the recognizer only reads the directory.

> **Known gap:** only 7 of the 18 dataset classes currently have a template committed.
> Segments of the other 11 classes can therefore only be `unknown` or false positives,
> which caps achievable accuracy (see §5). Adding the missing templates is the main lever.

---

## 4. Tests

- **`tests/test_recognition.py`** (148 lines) — matching math, threshold gating, the
  `unknown` case, and image-level aggregation, all on synthetic images (no dataset
  dependency). `make check` (ruff + strict mypy + pytest ≥ 80% coverage) stays green.

---

## 5. Calibration results

The threshold sweep (`calibration/recognition_sweep.csv`, 900 images) shows the
precision/recall trade-off on the single knob. Selected rows:

| `min_match_frac` | accuracy | unknown | false positives | FP rate |
|------------------|---------:|--------:|----------------:|--------:|
| 0.00 | 31.7% | 0 | 615 | 68.3% |
| 0.01 | 31.7% | 232 | 383 | 42.6% |
| **0.05** (default) | **27.6%** | 566 | 86 | **9.6%** |
| 0.10 | 14.0% | 770 | 4 | 0.4% |
| 0.13 | 6.8% | 839 | 0 | 0.0% |

The default `min_match_frac = 0.05` is the precision-first pick: it keeps false positives
under 10% while retaining most of the recall available given the 7-template coverage. Push
the knob higher to drive false positives to zero at the cost of recall — consistent with
the brief's "misses beat false positives" requirement. Accuracy is bounded above by the
missing-template gap in §3, not by the matcher.

---

## 6. Reports & docs

- **`docs/adr/0006-local-feature-recognition.md`** — decision record: why skimage SIFT (no
  OpenCV), ratio + cross-check + RANSAC, the match-fraction threshold, and `unknown` as a
  first-class output.
- **`docs/report/t2_report.pdf`** — full technical report (method, calibration sweep,
  confusion, limitations). Built by `scripts/build-t2-report.py`.
- **`docs/report/t2_simplified.pdf`** — 3-part deliverable (templates · parameters ·
  results), Brazilian Portuguese. Built by `scripts/build-t2-simplified.py`.
- **`todo.md`** — T2 roadmap (all items checked).
- **`enunciadot2.pdf`** — the T2 assignment brief.

---

## 7. How to reproduce

```sh
make templates      # bootstrap templates/ from the T1 detector
make run            # generate T1 segments under result/
make recognize      # → result/recognition.csv + calibration/recognition_sweep.csv
make report         # rebuild docs/report/*.pdf
make check          # ruff + mypy + pytest
```
