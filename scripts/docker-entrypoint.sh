#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-pdiseg}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"

prepare_writable_mount() {
    local path="$1"
    if ! mountpoint -q "$path"; then
        return 0
    fi
    mkdir -p "$path"
    if ! touch "${path}/.pdiseg_write_test" 2>/dev/null; then
        return 0
    fi
    rm -f "${path}/.pdiseg_write_test"
    chown -R "${APP_UID}:${APP_GID}" "$path"
}

if [[ "$(id -u)" -eq 0 ]]; then
    if [[ "${PDISEG_PREPARE_MOUNTS:-0}" == "1" ]]; then
        for dir in /data/output /data/calibration; do
            prepare_writable_mount "$dir"
        done
    fi
    exec gosu "${APP_USER}" "$@"
fi

exec "$@"
