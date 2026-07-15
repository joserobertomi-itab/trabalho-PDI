---
name: pdiseg-calibrate-review
description: >-
  Run calibration overlays, boxes.json, and the web review viewer for pdiseg.
  Use when generating calibration artifacts, inspecting overlays, or using
  pdiseg-review to compare source, boxes, and crops.
---

# Calibrate and review

## Local

```sh
make calibrate              # LIMIT=3 overlays per class
make calibrate LIMIT=9999   # all frames in boxes.json
make review                 # http://127.0.0.1:8765/
```

Outputs: `calibration/<class>/*_overlay.png`, `calibration/boxes.json`, `calibration/stats.csv`.

## Docker

```sh
make docker-calibrate
make docker-review PORT=8765
```

Compose: `docker compose --profile tools run --rm --no-deps calibrate`

Review UI: `docker compose --profile review up review` (does not re-run detection).

## Review contract

See `docs/review-viewer-contract.md`. Review does **not** re-run the detector.

## When calibrating after pipeline change

Regenerate overlays on weak classes (Sassami, selado) first; spot-check green boxes on name labels.
