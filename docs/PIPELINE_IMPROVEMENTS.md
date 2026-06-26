# Pipeline improvements report

## Metrics (full training set, 900 frames)

| Metric | Before (legacy) | After tuning (pre-ref) | After reference integration |
|--------|-----------------|------------------------|----------------------------|
| Total crops | 4648 | **1707** | **1775** |
| Avg crops / frame | 5.16 | **1.90** | **1.97** |
| Empty frames | 53 (5.9%) | **0** | **0** |
| Frames with 1 crop | — | 93 (10.3%) | **25 (2.8%)** |
| Frames with 2 crops | — | 807 (89.7%) | **875 (97.2%)** |
| Frames with >2 crops | — | 0 | **0** |

Weak-class spot check (detect-only, 2026-06-26):

| Class pattern | Frames | Crops | Empty |
|---------------|--------|-------|-------|
| Sassami | 100 | 198 | 0 |
| selado | 400 | 783 | 0 |
| Filezinho | 100 | 198 | 0 |

Measured with `detect_name_labels` on `data/Train_and_Validation` (detect-only, ~18.5 min / 900 frames).

## Reference integration (2026-06-14)

Techniques ported from `reference_segment_label.py` and `reference_label_segmentation.ipynb`:

| Task | Technique | Module | Default active |
|------|-----------|--------|----------------|
| TASK-01 | `clear_border` on combined mask | `masks.py` | yes |
| TASK-02 | Sobel edge-density candidate mask | `masks.py`, `candidates.py` | yes |
| TASK-03 | Opened background + `bright_on_dark` scoring | `scoring.py` | yes |
| TASK-04 | Extent + bimodality scoring | `scoring.py` | yes |
| TASK-05 | Debug viz for new masks/features | `debug/viz.py` | yes (viz only) |
| TASK-06 | Optional DoG text mask | `masks.py` | **no** (`use_dog_text=False`) |
| TASK-07 | Adaptive dark-body + lateral margin | `masks.py`, `postprocess.py` | partial (margin yes; body mask only if `min_body_overlap>0`) |

All operators use the **scikit-image / scipy** stack (no OpenCV), per ADR-0002.

## Summary

- **−62% total crops** vs legacy pipeline.
- Reference integration: **+68 crops** vs pre-ref baseline (+4%), still within ±10% target; **0 empty frames** maintained.
- `max_labels_per_frame=2` with score gap between 1st and 2nd candidate.
- New `DetectionConfig` fields for edge density, clear_border, bimodality, optional DoG.

See `docs/tasks/integrate-reference-segmentation.md` for task breakdown and `docs/src/ARCHITECTURE.md` for module layout.
