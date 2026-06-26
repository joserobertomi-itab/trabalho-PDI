---
name: pdiseg-pipeline
description: >-
  Modify the classical CV segmentation pipeline in pdiseg (masks, candidates,
  scoring, postprocess, DetectionConfig). Use when changing detection behavior,
  tuning thresholds, reducing false positives, or refactoring pipeline modules.
---

# pdiseg pipeline

## Read first

- `.agents/workflows/pipeline-change.md`
- `CONTEXT.md`, relevant `docs/adr/`
- `src/pdiseg/detection/config.py` for thresholds
- `docs/src/ARCHITECTURE.md` for module map

## Module chain

```
load_image → preprocess_image → find_candidate_boxes(work, text_source=gray)
  → score_candidates → postprocess_boxes → detect_name_labels
```

## Tuning rules

- Prefer `DetectionConfig` fields over literals.
- Text density: always from **gray**, not CLAHE `work`.
- Scoring: global dark threshold; include `text_density` feature.
- Cap crops: `max_labels_per_frame` (default 2) + score gap between 1st/2nd.
- Fallback must respect `score_threshold_fallback`; no junk crops below ~0.30.

## Verify

```sh
uv run pytest tests/test_run.py tests/test_detector_validation.py -q
make check
```

## Metrics

If full run: compare total crops and empty frames vs `docs/PIPELINE_IMPROVEMENTS.md`.
