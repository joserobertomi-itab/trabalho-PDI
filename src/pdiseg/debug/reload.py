"""Reload pipeline modules in dependency order (Jupyter / autoreload safety)."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_PIPELINE_MODULES: tuple[str, ...] = (
    "pdiseg.detection.config",
    "pdiseg.detection.preprocess",
    "pdiseg.detection.masks",
    "pdiseg.detection.candidates",
    "pdiseg.detection.scoring",
    "pdiseg.detection.postprocess",
    "pdiseg.detection.detector",
    "pdiseg.runtime.pipeline",
    "pdiseg.debug.viz",
    "pdiseg.debug.sample",
)


def reload_pipeline_modules(*, reload_package: bool = True) -> list[ModuleType]:
    """Reload detection + debug modules so dataclass/API changes take effect in notebooks."""
    loaded: list[ModuleType] = []
    for name in _PIPELINE_MODULES:
        module = sys.modules.get(name)
        module = importlib.import_module(name) if module is None else importlib.reload(module)
        loaded.append(module)
    if reload_package and "pdiseg" in sys.modules:
        loaded.append(importlib.reload(sys.modules["pdiseg"]))
    return loaded


def assert_pipeline_schema() -> None:
    """Fail fast when the notebook kernel still has stale dataclass definitions."""
    from pdiseg.detection.masks import CandidateMasks

    required = ("edge_density", "dog_text")
    missing = [field for field in required if field not in CandidateMasks.__dataclass_fields__]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Stale pdiseg in kernel (CandidateMasks missing: {names}). "
            "Run reload_pipeline_modules() or Restart Kernel, then run setup again."
        )
