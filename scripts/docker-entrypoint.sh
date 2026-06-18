#!/usr/bin/env bash
set -euo pipefail

APP_USER="${APP_USER:-pdiseg}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"
NICE_LEVEL="${NICE_LEVEL:-10}"
IONICE_CLASS="${IONICE_CLASS:-3}"
IONICE_LEVEL="${IONICE_LEVEL:-7}"

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

run_as_app_user() {
    if command -v ionice >/dev/null 2>&1; then
        exec gosu "${APP_USER}" ionice -c"${IONICE_CLASS}" -n"${IONICE_LEVEL}" nice -n "${NICE_LEVEL}" "$@"
    fi
    exec gosu "${APP_USER}" nice -n "${NICE_LEVEL}" "$@"
}

run_with_priority() {
    if command -v ionice >/dev/null 2>&1; then
        exec ionice -c"${IONICE_CLASS}" -n"${IONICE_LEVEL}" nice -n "${NICE_LEVEL}" "$@"
    fi
    exec nice -n "${NICE_LEVEL}" "$@"
}

if [[ "$(id -u)" -eq 0 ]]; then
    if [[ "${PDISEG_PREPARE_MOUNTS:-0}" == "1" ]]; then
        for dir in /data/output /data/calibration; do
            prepare_writable_mount "$dir"
        done
    fi
    run_as_app_user "$@"
fi

run_with_priority "$@"
