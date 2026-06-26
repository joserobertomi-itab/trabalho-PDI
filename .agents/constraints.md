# Project constraints (agents must follow)

## Academic / detector

- **Classical PDI only** — Part 1 techniques: thresholding, morphology, filters, histograms, connected components, etc.
- **No OCR** — do not read product names from pixels.
- **No trained ML** — no CNNs, no external segmentation APIs, no class-name hints from folder names.
- **Crop from original frame** — outputs must come from the source image geometry (see ADR-0003).

## Product goal

- Target: **name label** (dark badge with product name), not brand badge alone, not glare, not random plastic.
- Balance: correct label, few false positives, avoid empty frames when a label is visible.

## Engineering

- **Python 3.12+**, `uv`, package `pdiseg`.
- **Tests required** for behavior changes — `make check` before claiming done.
- **Minimal diffs** — match existing module style; no drive-by refactors.
- **Committed artifacts in English**; chat with user in Brazilian Portuguese.

## Process (explicit non-goals)

- **No spec-driven development** — do not require a spec/PRD file before coding. Issues and domain docs are enough.
- Do not invent `specs/`, `prd/`, or approval gates unless the user explicitly asks.

## Docker / delivery

- Default data path: `data/Train_and_Validation`.
- Output: `result/<class>/<stem>_segmentada_N.png`.
- Compose profile `tools` for `calibrate` and `review` — flag goes **before** subcommand: `docker compose --profile tools up calibrate`.
