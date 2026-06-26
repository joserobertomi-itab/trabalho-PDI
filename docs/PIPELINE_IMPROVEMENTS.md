# Pipeline improvements report

## Metrics (full training set, 900 frames)

| Metric | Before (legacy) | After tuning (pre-ref) | After reference integration |
|--------|-----------------|------------------------|----------------------------|
| Total crops | 4648 | **1707** | **~1780** (90-frame sample extrapolated) |
| Avg crops / frame | 5.16 | **1.90** | **~1.98** |
| Empty frames | 53 (5.9%) | **0** | **0** (90-frame sample) |
| Frames with 1 crop | — | 93 (10.3%) | — |
| Frames with 2 crops | — | 807 (89.7%) | — |

Sample validation (90 frames, detect-only): 178 crops, avg 1.98/frame, 0 empty.

Measured with `detect_name_labels` on `data/Train_and_Validation`.

## Reference integration (2026-06-14)

Techniques ported from `segmenta_rotulo(1).py` and `Segmentacao_Rotulos_PDI.ipynb`:

| Task | Technique | Module |
|------|-----------|--------|
| TASK-01 | `clear_border` on combined mask | `masks.py` |
| TASK-02 | Sobel edge-density candidate mask | `masks.py`, `candidates.py` |
| TASK-03 | Opened background + `bright_on_dark` scoring | `scoring.py` |
| TASK-04 | Extent + bimodality scoring | `scoring.py` |
| TASK-05 | Debug viz for new masks/features | `debug/viz.py` |
| TASK-06 | Optional DoG text mask (`use_dog_text=False`) | `masks.py` |
| TASK-07 | Adaptive dark-body + lateral margin filter | `masks.py`, `postprocess.py` |

All operators use the **scikit-image / scipy** stack (no OpenCV), per ADR-0002.

## Summary

- **−63% total crops** vs legacy pipeline (pre-ref tuning).
- Reference integration keeps **0 empty frames** on sample; crop count stable (~±5%).
- `max_labels_per_frame=2` with score gap between 1st and 2nd candidate.
- New `DetectionConfig` fields for edge density, clear_border, bimodality, optional DoG.

See `docs/tasks/integrate-reference-segmentation.md` for task breakdown and `docs/src/ARCHITECTURE.md` for module layout.
