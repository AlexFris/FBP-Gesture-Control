"""Utilities for accessing Kinect v2 camera streams.

This module provides a small convenience wrapper around the
``pykinect2`` package that ships with the Kinect for Windows SDK.  The
goal is to make it easy to consume the Kinect streams inside other
parts of the project (for instance the gesture recogniser) without
needing to interact with the fairly low level API directly.

Example
-------

```python
from KinectSensor import KinectSensor

sensor = KinectSensor(enable_color=True, enable_infrared=True, enable_depth=True)
with sensor:
    while True:
        frames = sensor.poll_frames()
        color = frames.color
        infrared = frames.infrared
        depth = frames.depth
        # ... use the frames here ...
```

The module only depends on :mod:`numpy` and :mod:`pykinect2` which are
both available on Windows when the Kinect SDK is installed.  All frame
arrays are returned as ``numpy.ndarray`` instances in their raw
resolution (1920x1080 for colour and 512x424 for infrared/depth).  The
colour frame is returned in OpenCV friendly BGR order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

try:
    from pykinect2 import PyKinectRuntime, PyKinectV2
except AssertionError as exc:  # pragma: no cover - triggered by incompatible builds only
    from importlib import import_module, util
    import sys
    import types

    def _load_pykinect_with_statstg_patch() -> tuple[Any, Any]:
        """Reload ``pykinect2`` while tolerating the STATSTG size difference."""

        # Ensure the package itself is importable; this does not trigger the
        # failing assertion (it only occurs when PyKinectV2 is executed).
        import_module("pykinect2")

        spec_v2 = util.find_spec("pykinect2.PyKinectV2")
        if spec_v2 is None or spec_v2.origin is None:
            raise RuntimeError("Unable to locate pykinect2.PyKinectV2 source file.")

        source_v2 = Path(spec_v2.origin).read_text(encoding="utf-8")
        sentinel = "assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)"
        if sentinel not in source_v2:
            raise RuntimeError(
                "Unexpected pykinect2.PyKinectV2 contents; cannot apply STATSTG patch."
            )

        patched_source = source_v2.replace(
            sentinel,
            "assert sizeof(tagSTATSTG) in (72, 80), sizeof(tagSTATSTG)",
        )

        module_v2 = types.ModuleType("pykinect2.PyKinectV2")
        module_v2.__file__ = spec_v2.origin
        module_v2.__package__ = "pykinect2"
        exec(compile(patched_source, spec_v2.origin, "exec"), module_v2.__dict__)
        sys.modules[module_v2.__name__] = module_v2

        spec_runtime = util.find_spec("pykinect2.PyKinectRuntime")
        if spec_runtime is None or spec_runtime.origin is None or spec_runtime.loader is None:
            raise RuntimeError("Unable to locate pykinect2.PyKinectRuntime source file.")

        module_runtime = util.module_from_spec(spec_runtime)
        spec_runtime.loader.exec_module(module_runtime)
        sys.modules[module_runtime.__name__] = module_runtime

        return module_runtime, module_v2

    try:
        PyKinectRuntime, PyKinectV2 = _load_pykinect_with_statstg_patch()
    except Exception as patch_exc:  # pragma: no cover - best effort fallback
        raise RuntimeError(
            "pykinect2 failed an internal structure size check while importing. "
            "This happens when the installed wheel was built for a different "
            "Python version/architecture. A best-effort patch that tolerates "
            "72- or 80-byte STATSTG layouts also failed; install a wheel built "
            "for your interpreter (community Python 3.8 builds are known to work)."
        ) from patch_exc
except ImportError as exc:  # pragma: no cover - informative error for runtime only
    raise ImportError(
        "pykinect2 is required to use the KinectSensor helper. "
        "Install the Kinect for Windows SDK v2 and the pykinect2 package."
    ) from exc


@dataclass(frozen=True)
class FrameSet:
    """Container returned by :meth:`KinectSensor.poll_frames`.

    Attributes are ``None`` when the corresponding stream has not
    produced a new frame during the poll call.  Depth and infrared
    values are returned as unsigned 16-bit arrays to preserve the raw
    measurement data.  The colour frame is returned as an unsigned
    8-bit BGR image so it can be displayed directly with OpenCV.
    """

    color: Optional[np.ndarray] = None
    infrared: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    body_index: Optional[np.ndarray] = None
    body_frame: Optional[Any] = None


class KinectSensor:
    """High level wrapper for Kinect v2 data streams.

    Parameters
    ----------
    enable_color:
        Enable streaming of the colour (RGB) camera.
    enable_infrared:
        Enable streaming of the infrared camera (short exposure).
    enable_depth:
        Enable streaming of the time-of-flight depth camera.
    enable_body_index:
        Enable streaming of the body index map.
    enable_body:
        Enable skeleton/body tracking data.

    Notes
    -----
    * At least one stream must be enabled when instantiating the sensor.
    * The class can be used as a context manager which guarantees that
      the underlying sensor handle is closed.
    """

    COLOR_HEIGHT = PyKinectV2.color_frame_desc.Height
    COLOR_WIDTH = PyKinectV2.color_frame_desc.Width
    INFRARED_HEIGHT = PyKinectV2.infrared_frame_desc.Height
    INFRARED_WIDTH = PyKinectV2.infrared_frame_desc.Width
    DEPTH_HEIGHT = PyKinectV2.depth_frame_desc.Height
    DEPTH_WIDTH = PyKinectV2.depth_frame_desc.Width

    def __init__(
        self,
        *,
        enable_color: bool = True,
        enable_infrared: bool = False,
        enable_depth: bool = False,
        enable_body_index: bool = False,
        enable_body: bool = False,
    ) -> None:
        self._enable_color = enable_color
        self._enable_infrared = enable_infrared
        self._enable_depth = enable_depth
        self._enable_body_index = enable_body_index
        self._enable_body = enable_body

        sources = 0
        if enable_color:
            sources |= PyKinectV2.FrameSourceTypes_Color
        if enable_infrared:
            sources |= PyKinectV2.FrameSourceTypes_Infrared
        if enable_depth:
            sources |= PyKinectV2.FrameSourceTypes_Depth
        if enable_body_index:
            sources |= PyKinectV2.FrameSourceTypes_BodyIndex
        if enable_body:
            sources |= PyKinectV2.FrameSourceTypes_Body

        if sources == 0:
            raise ValueError("At least one Kinect frame source must be enabled.")

        self._runtime: Optional[PyKinectRuntime.PyKinectRuntime]
        self._runtime = PyKinectRuntime.PyKinectRuntime(sources)

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "KinectSensor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def runtime(self) -> PyKinectRuntime.PyKinectRuntime:
        """Expose the underlying :class:`PyKinectRuntime` instance."""

        if self._runtime is None:
            raise RuntimeError("Kinect runtime has already been closed.")
        return self._runtime

    def poll_frames(self) -> FrameSet:
        """Retrieve the latest frames from all enabled streams.

        This method is non-blocking; it returns ``None`` for streams
        that have not produced a new frame since the last poll.
        """

        color = self._extract_color_frame() if self._enable_color else None
        infrared = self._extract_infrared_frame() if self._enable_infrared else None
        depth = self._extract_depth_frame() if self._enable_depth else None
        body_index = (
            self._extract_body_index_frame() if self._enable_body_index else None
        )
        body_frame = self._extract_body_frame() if self._enable_body else None

        return FrameSet(
            color=color,
            infrared=infrared,
            depth=depth,
            body_index=body_index,
            body_frame=body_frame,
        )

    def frames(self) -> Iterator[FrameSet]:
        """Continuous generator yielding the latest :class:`FrameSet`.

        This method is convenient when you want to stream the frames in
        a ``for`` loop::

            for frames in sensor.frames():
                # process frames here

        The loop yields as fast as new frames are available.
        """

        while True:
            yield self.poll_frames()

    def close(self) -> None:
        """Release the Kinect sensor resources."""

        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def infrared_to_uint8(frame: np.ndarray) -> np.ndarray:
        """Scale a raw infrared frame to 8-bit for visualisation."""

        if frame.dtype != np.uint16:
            raise ValueError("Infrared frames must be uint16 arrays.")

        normalised = (frame.astype(np.float32) / np.iinfo(np.uint16).max) * 255.0
        return normalised.astype(np.uint8)

    @staticmethod
    def depth_to_colormap(
        frame: np.ndarray,
        *,
        min_distance_mm: int = 500,
        max_distance_mm: int = 4500,
    ) -> np.ndarray:
        """Convert a raw depth map to a coloured visualisation."""

        if frame.dtype != np.uint16:
            raise ValueError("Depth frames must be uint16 arrays.")

        clipped = np.clip(frame.astype(np.float32), min_distance_mm, max_distance_mm)
        span = max_distance_mm - min_distance_mm
        span = 1 if span <= 0 else span
        scaled = (clipped - min_distance_mm) / span
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled_uint8 = (scaled * 255).astype(np.uint8)
        return cv2.applyColorMap(scaled_uint8, cv2.COLORMAP_JET)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_color_frame(self) -> Optional[np.ndarray]:
        runtime = self.runtime
        if not runtime.has_new_color_frame():
            return None

        frame = runtime.get_last_color_frame()
        frame = frame.reshape((self.COLOR_HEIGHT, self.COLOR_WIDTH, 4)).astype(np.uint8)

        # Convert from RGBA to BGR for OpenCV compatibility
        frame = frame[:, :, :3][:, :, ::-1]
        return frame

    def _extract_infrared_frame(self) -> Optional[np.ndarray]:
        runtime = self.runtime
        if not runtime.has_new_infrared_frame():
            return None

        frame = runtime.get_last_infrared_frame()
        frame = frame.reshape((self.INFRARED_HEIGHT, self.INFRARED_WIDTH)).astype(
            np.uint16
        )
        return frame

    def _extract_depth_frame(self) -> Optional[np.ndarray]:
        runtime = self.runtime
        if not runtime.has_new_depth_frame():
            return None

        frame = runtime.get_last_depth_frame()
        frame = frame.reshape((self.DEPTH_HEIGHT, self.DEPTH_WIDTH)).astype(np.uint16)
        return frame

    def _extract_body_index_frame(self) -> Optional[np.ndarray]:
        runtime = self.runtime
        if not runtime.has_new_body_index_frame():
            return None

        frame = runtime.get_last_body_index_frame()
        frame = frame.reshape((self.DEPTH_HEIGHT, self.DEPTH_WIDTH)).astype(np.uint8)
        return frame

    def _extract_body_frame(self) -> Optional[Any]:
        runtime = self.runtime
        if not runtime.has_new_body_frame():
            return None

        return runtime.get_last_body_frame()


__all__ = ["KinectSensor", "FrameSet"]


# ----------------------------------------------------------------------
# Optional live preview
# ----------------------------------------------------------------------

# Set this constant to ``True`` to open a simple OpenCV preview of the
# colour, infrared, depth and body index streams.  The preview opens a
# window per stream and can be dismissed by pressing the ``q`` key in any
# of the windows.  The preview is disabled by default so importing the
# module never launches any UI unintentionally.
RUN_SENSOR_PREVIEW = False


def _run_sensor_preview() -> None:
    """Launch a basic multi-window preview of the Kinect streams."""

    window_names = {
        "color": "Kinect Colour",
        "infrared": "Kinect Infrared",
        "depth": "Kinect Depth",
        "body_index": "Kinect Body Index",
    }

    sensor = KinectSensor(
        enable_color=True,
        enable_infrared=True,
        enable_depth=True,
        enable_body_index=True,
    )

    try:
        with sensor:
            while True:
                frames = sensor.poll_frames()

                if frames.color is not None:
                    cv2.imshow(window_names["color"], frames.color)

                if frames.infrared is not None:
                    infrared_vis = KinectSensor.infrared_to_uint8(frames.infrared)
                    cv2.imshow(window_names["infrared"], infrared_vis)

                if frames.depth is not None:
                    depth_vis = KinectSensor.depth_to_colormap(frames.depth)
                    cv2.imshow(window_names["depth"], depth_vis)

                if frames.body_index is not None:
                    cv2.imshow(window_names["body_index"], frames.body_index)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__" and RUN_SENSOR_PREVIEW:
    _run_sensor_preview()
