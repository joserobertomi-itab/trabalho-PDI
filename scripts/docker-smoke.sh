#!/usr/bin/env bash
# End-to-end smoke test for the Docker Compose delivery path.
# Creates a tiny synthetic dataset/, runs the pipeline container, asserts resultado/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/.docker-smoke/dataset"
OUT_DIR="$ROOT/.docker-smoke/resultado"
CALIB_DIR="$ROOT/.docker-smoke/calibration"

rm -rf "$ROOT/.docker-smoke"
mkdir -p "$DATA_DIR/SmokeClass" "$OUT_DIR" "$CALIB_DIR"
chmod -R a+rwx "$ROOT/.docker-smoke"

uv run python - <<'PY'
import numpy as np
import imageio.v3 as iio
from pathlib import Path

out = Path(".docker-smoke/dataset/SmokeClass")
out.mkdir(parents=True, exist_ok=True)
h, w = 300, 500
frame = np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))
frame[100:180, 200:360] = 30
for x in range(200, 360, 8):
    frame[100:180, x : x + 3] = 220
iio.imwrite(out / "frame.png", frame)
PY

export DATA="$DATA_DIR"
export OUT="$OUT_DIR"
export CALIB="$CALIB_DIR"

docker compose build pipeline

docker compose up --no-build pipeline
test -f "$OUT_DIR/SmokeClass/frame_segmentada_1.png"

docker compose --profile tools run --rm calibrate
test -f "$CALIB_DIR/boxes.json"
test -f "$CALIB_DIR/stats.csv"

echo "Docker Compose smoke test passed (pipeline + calibrate)."
