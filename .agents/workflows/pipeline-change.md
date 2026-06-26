# Workflow: Pipeline change

Use when modifying detection, masks, scoring, or postprocess.

## Steps

1. **Baseline** — note current metrics if `result/` exists (crops/frame, empty count).
2. **Read** — `src/pdiseg/detection/detector.py` entry, trace masks → candidates → scoring → postprocess.
3. **Implement** — smallest change; tune `DetectionConfig` not magic literals in loops.
4. **Unit tests** — `uv run pytest tests/test_run.py tests/test_detector_validation.py -q`.
5. **Synthetic sanity** — gradient + text block tests must still pass.
6. **Sample real** — 5–10 frames from weak classes (Sassami, selado) via notebook or `inspect_detection`.
7. **Full run** (if requested) — `make run` (~10+ min on 900 frames).
8. **Document** — update `docs/PIPELINE_IMPROVEMENTS.md` if metrics change materially.

## Anti-patterns

- Raising `max_labels_per_frame` to fix empty frames without fixing scores.
- Per-class thresholds keyed on folder name.
- Using CLAHE image for text-density (use `prep.gray`).
