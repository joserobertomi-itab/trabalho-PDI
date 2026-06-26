# Pipeline improvements report

## Metrics (full training set, 900 frames)

| Metric | Before (legacy) | After (final config) |
|--------|-----------------|----------------------|
| Total crops | 4648 | **1707** |
| Avg crops / frame | 5.16 | **1.90** |
| Empty frames | 53 (5.9%) | **0** |
| Frames with 1 crop | — | **93 (10.3%)** |
| Frames with 2 crops | — | **807 (89.7%)** |

Measured with `detect_name_labels` on `data/Train_and_Validation` (detect-only pass, no disk write).

## Summary

- **−63% total crops** vs legacy pipeline.
- **No empty frames** on the training set (trade-off: some secondary crops may still be false positives).
- Cap `max_labels_per_frame=2` with score gap between 1st and 2nd candidate.

See README section 10 for module layout and debug notebook.
