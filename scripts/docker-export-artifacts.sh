#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-pdiseg-result}"
CALIB="${CALIB:-pdiseg-calibration}"
EXPORT_RESULT="${EXPORT_RESULT:-./result}"
EXPORT_CALIB="${EXPORT_CALIB:-./calibration}"

is_named_volume() {
    local name="$1"
    [[ "$name" != ./* && "$name" != /* ]]
}

copy_volume() {
    local volume="$1"
    local host_dir="$2"
    if ! is_named_volume "$volume"; then
        echo "skip $volume (bind mount at $volume)"
        return 0
    fi
    if ! docker volume inspect "$volume" >/dev/null 2>&1; then
        echo "skip $volume (not created yet)"
        return 0
    fi
    mkdir -p "$host_dir"
    docker run --rm \
        -v "${volume}:/from:ro" \
        -v "$(realpath "$host_dir"):/to" \
        alpine:3.20 cp -a /from/. /to/
    echo "exported $volume -> $host_dir"
}

copy_volume "$OUT" "$EXPORT_RESULT"
copy_volume "$CALIB" "$EXPORT_CALIB"
