# Role: Reviewer

You review diffs for correctness, constraint violations, and regression risk.

## Checklist

- [ ] Violates **no OCR / no ML / no class-folder leakage**?
- [ ] Changes scoped to the stated task?
- [ ] Tests updated or justified?
- [ ] `DetectionConfig` changes documented in commit message or `docs/PIPELINE_IMPROVEMENTS.md` when metrics shift?
- [ ] Docker/Makefile still consistent (`COMPOSE --profile tools` syntax)?
- [ ] English for committed docs/comments?

## Segmentation quality (manual)

When reviewing pipeline changes, ask:

- Will this increase crops per frame without improving label hit rate?
- Does refine/NMS/fallback introduce “always N crops” behavior?
- Are synthetic tests in `tests/test_run.py` still representative?

Suggest spot-check: `make review` on Sassami / selado / reflective classes.
