"""Optional runtime acceleration backend selection."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

BackendName = Literal["cpu", "gpu"]
BackendMode = Literal["auto", "cpu", "gpu"]
GpuModules = tuple[Any, Any]

_DISABLED_GPU_REASON: str | None = None
_LOGGED = False


@dataclass(frozen=True)
class AccelerationInfo:
    backend: BackendName
    requested: BackendMode
    reason: str
    device_count: int = 0
    device_name: str | None = None


def requested_backend() -> BackendMode:
    raw = os.environ.get("PDISEG_BACKEND", "auto").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"cpu", "0", "false", "off", "none"}:
        return "cpu"
    if raw in {"gpu", "cuda", "1", "true", "on"}:
        return "gpu"
    return "auto"


def gpu_modules() -> GpuModules | None:
    """Return ``(cupy, cupyx.scipy.ndimage)`` when CUDA acceleration is usable."""
    if _DISABLED_GPU_REASON is not None:
        return None
    mode = requested_backend()
    if mode == "cpu":
        return None
    modules, reason, _, _ = _load_gpu_modules()
    if modules is None:
        if mode == "gpu":
            raise RuntimeError(f"PDISEG_BACKEND=gpu requested, but GPU is unavailable: {reason}")
        return None
    return modules


def acceleration_info() -> AccelerationInfo:
    mode = requested_backend()
    if mode == "cpu":
        return AccelerationInfo(backend="cpu", requested=mode, reason="CPU backend requested")
    if _DISABLED_GPU_REASON is not None:
        return AccelerationInfo(
            backend="cpu",
            requested=mode,
            reason=f"GPU disabled after runtime failure: {_DISABLED_GPU_REASON}",
        )
    modules, reason, device_count, device_name = _load_gpu_modules()
    if modules is not None:
        return AccelerationInfo(
            backend="gpu",
            requested=mode,
            reason="CuPy CUDA backend available",
            device_count=device_count,
            device_name=device_name,
        )
    if mode == "gpu":
        raise RuntimeError(f"PDISEG_BACKEND=gpu requested, but GPU is unavailable: {reason}")
    return AccelerationInfo(backend="cpu", requested=mode, reason=reason)


def disable_gpu(reason: str) -> None:
    """Disable GPU for the current process after an operation-level failure."""
    global _DISABLED_GPU_REASON
    _DISABLED_GPU_REASON = reason


def log_acceleration_once() -> None:
    """Print backend selection when explicitly requested by the environment."""
    global _LOGGED
    if _LOGGED or os.environ.get("PDISEG_BACKEND_LOG", "0") not in {"1", "true", "TRUE"}:
        return
    _LOGGED = True
    info = acceleration_info()
    device = f", device={info.device_name}" if info.device_name else ""
    print(
        f"pdiseg backend: {info.backend} (requested={info.requested}, {info.reason}{device})",
        file=sys.stderr,
        flush=True,
    )


@lru_cache(maxsize=1)
def _load_gpu_modules() -> tuple[GpuModules | None, str, int, str | None]:
    try:
        import cupy as cp  # type: ignore[import-not-found]
        from cupyx.scipy import ndimage as cndimage  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        return None, f"CuPy is not installed ({exc})", 0, None

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on host driver
        return None, f"CUDA device query failed ({exc})", 0, None
    if device_count <= 0:
        return None, "No CUDA devices reported by CuPy", 0, None

    device_name: str | None = None
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name")
        if isinstance(raw_name, bytes):
            device_name = raw_name.decode("utf-8", errors="replace")
        elif raw_name is not None:
            device_name = str(raw_name)
    except Exception:
        device_name = None

    return (cp, cndimage), "CuPy CUDA backend available", device_count, device_name
