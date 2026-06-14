#!/usr/bin/env bash
# Ensure writable bind mounts are owned by the app user, then drop privileges
# with lowered CPU/IO priority so batch runs stay gentle on the host.
set -euo pipefail

APP_USER="${APP_USER:-pdiseg}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"
NICE_LEVEL="${NICE_LEVEL:-10}"
IONICE_CLASS="${IONICE_CLASS:-3}"
IONICE_LEVEL="${IONICE_LEVEL:-7}"

prepare_writable_mount() {
    local path="$1"
    if mountpoint -q "$path"; then
        mkdir -p "$path"
        chown -R "${APP_UID}:${APP_GID}" "$path"
    fi
}

run_as_app_user() {
    # ionice/nice keep the batch job from starving the desktop; ignore if unavailable.
    if command -v ionice >/dev/null 2>&1; then
        exec gosu "${APP_USER}" ionice -c"${IONICE_CLASS}" -n"${IONICE_LEVEL}" nice -n "${NICE_LEVEL}" "$@"
    fi
    exec gosu "${APP_USER}" nice -n "${NICE_LEVEL}" "$@"
}

if [[ "$(id -u)" -eq 0 ]]; then
    for dir in /data/output /data/calibration; do
        prepare_writable_mount "$dir"
    done
    run_as_app_user "$@"
fi

if command -v ionice >/dev/null 2>&1; then
    exec ionice -c"${IONICE_CLASS}" -n"${IONICE_LEVEL}" nice -n "${NICE_LEVEL}" "$@"
fi
exec nice -n "${NICE_LEVEL}" "$@"
