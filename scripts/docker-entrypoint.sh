#!/usr/bin/env bash
# Ensure writable bind mounts are owned by the app user, then drop privileges.
set -euo pipefail

APP_USER="${APP_USER:-pdiseg}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"

prepare_writable_mount() {
    local path="$1"
    if mountpoint -q "$path"; then
        mkdir -p "$path"
        chown -R "${APP_UID}:${APP_GID}" "$path"
    fi
}

if [[ "$(id -u)" -eq 0 ]]; then
    for dir in /data/output /data/calibration; do
        prepare_writable_mount "$dir"
    done
    exec gosu "${APP_USER}" "$@"
fi

exec "$@"
