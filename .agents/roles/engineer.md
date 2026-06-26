# Role: Engineer

You implement changes under `src/pdiseg/` and supporting scripts/tests/docs.

## Before coding

1. `CONTEXT.md` + relevant `docs/adr/*.md`.
2. `.agents/constraints.md`.
3. Read the module you will touch (`pipeline.py` orchestrates; logic lives in `detector`, `masks`, `scoring`, `postprocess`, etc.).

## While coding

- Keep pipeline **modular** — one concern per file.
- Thresholds in `DetectionConfig` must be **relative** (fraction of frame, percentiles) with a one-line rationale in PR/commit if non-obvious.
- Prefer extending `score_candidate` / masks over new ad-hoc filters.
- Do not add dependencies that violate ADR-0002 without a new ADR.

## Done means

- `make check` passes (or explain what could not run).
- If pipeline behavior changed: note impact on crops/empty frames; suggest `make calibrate` + `make review` sample.
- No secrets in repo.
