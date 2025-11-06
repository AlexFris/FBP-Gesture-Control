"""Compatibility helpers for loading pykinect2 on newer Python builds."""

from __future__ import annotations

from importlib import import_module, invalidate_caches, util
from pathlib import Path
from types import ModuleType
from typing import Tuple


def ensure_pykinect_import(exc: Exception) -> Tuple[ModuleType, ModuleType]:
    """Resolve pykinect2 import issues triggered by ``KinectSensor``."""

    if isinstance(exc, ImportError):  # pragma: no cover - runtime guidance only
        raise ImportError(
            "pykinect2 is required to use the KinectSensor helper. Install the Kinect "
            "for Windows SDK v2 and the pykinect2 package."
        ) from exc

    if not isinstance(exc, AssertionError):
        raise exc

    try:
        return load_pykinect_modules()
    except Exception as patch_exc:  # pragma: no cover - best effort fallback
        raise RuntimeError(
            "pykinect2 failed an internal structure size check while importing. "
            "An automatic patch that relaxes the STATSTG assertion was attempted but "
            "did not complete successfully. Install a wheel compiled for your Python "
            "interpreter (community Python 3.8 builds are known to work)."
        ) from patch_exc


def _rewrite_statstg_assertion(source_path: Path) -> None:
    """Allow both 72- and 80-byte STATSTG layouts in PyKinectV2."""

    sentinel = "assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)"
    replacement = "assert sizeof(tagSTATSTG) in (72, 80), sizeof(tagSTATSTG)"

    original = source_path.read_text(encoding="utf-8")
    if replacement in original:
        return
    if sentinel not in original:
        raise RuntimeError(
            "Unexpected pykinect2.PyKinectV2 contents; cannot apply STATSTG patch."
        )

    source_path.write_text(original.replace(sentinel, replacement), encoding="utf-8")


def _find_source_path() -> Path:
    spec = util.find_spec("pykinect2.PyKinectV2")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to locate pykinect2.PyKinectV2 source file to patch.")

    source_path = Path(spec.origin)
    if source_path.suffix == ".pyc":
        source_path = source_path.with_suffix(".py")
    if not source_path.exists():
        raise RuntimeError("Unable to locate pykinect2.PyKinectV2 source file to patch.")
    return source_path


def load_pykinect_modules() -> Tuple[ModuleType, ModuleType]:
    """Patch PyKinectV2 on disk then import both runtime modules."""

    source_path = _find_source_path()
    _rewrite_statstg_assertion(source_path)

    invalidate_caches()
    # Ensure we reload fresh versions after patching on disk.
    import sys

    sys.modules.pop("pykinect2.PyKinectV2", None)
    sys.modules.pop("pykinect2.PyKinectRuntime", None)

    runtime = import_module("pykinect2.PyKinectRuntime")
    v2 = import_module("pykinect2.PyKinectV2")
    return runtime, v2
