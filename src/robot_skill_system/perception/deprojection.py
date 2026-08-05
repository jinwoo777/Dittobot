"""Distortion-aware local RGB-D pixel deprojection.

Core and mock operation use the pinhole path without importing optional hardware
packages.  Recorded intrinsics that declare a real distortion model are handled
only by the matching RealSense SDK implementation; unavailable or unsupported
calibration therefore fails closed instead of silently producing biased metric
geometry.
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from robot_skill_system.capture.interfaces import CameraIntrinsics

FloatArray = NDArray[np.float64]

_NO_DISTORTION_MODELS = {"none"}
_REALSENSE_DISTORTION_MODELS = {
    "brown_conrady",
    "ftheta",
    "inverse_brown_conrady",
    "kannala_brandt4",
    "modified_brown_conrady",
}


def _normalized_distortion_model(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    if normalized.startswith("rs2_distortion_"):
        normalized = normalized.removeprefix("rs2_distortion_")
    return normalized


def _load_pyrealsense2() -> ModuleType:
    try:
        import pyrealsense2
    except ImportError as exc:
        raise ValueError(
            "distortion-aware metric deprojection requires optional pyrealsense2"
        ) from exc
    return cast(ModuleType, pyrealsense2)


def _validated_inputs(
    intrinsics: CameraIntrinsics,
    pixels_xy_px: NDArray[np.float64],
    depths_m: NDArray[np.float64],
) -> None:
    if pixels_xy_px.ndim != 2 or pixels_xy_px.shape[1] != 2:
        raise ValueError("pixels_xy_px must have shape (N, 2)")
    if depths_m.ndim != 1 or len(depths_m) != len(pixels_xy_px):
        raise ValueError("depths_m must have shape (N,) matching pixels_xy_px")
    calibration_values = (
        intrinsics.fx_px,
        intrinsics.fy_px,
        intrinsics.cx_px,
        intrinsics.cy_px,
    )
    if not all(math.isfinite(value) for value in calibration_values):
        raise ValueError("camera intrinsics must be finite")
    if not np.all(np.isfinite(pixels_xy_px)):
        raise ValueError("pixel coordinates must be finite")
    if len(pixels_xy_px) and (
        np.any(pixels_xy_px[:, 0] < 0.0)
        or np.any(pixels_xy_px[:, 0] > intrinsics.width_px - 1)
        or np.any(pixels_xy_px[:, 1] < 0.0)
        or np.any(pixels_xy_px[:, 1] > intrinsics.height_px - 1)
    ):
        raise ValueError("pixel is outside colour image bounds")
    if not np.all(np.isfinite(depths_m)) or np.any(depths_m <= 0.0):
        raise ValueError("depths_m must contain finite positive values")


def _realsense_intrinsics(
    intrinsics: CameraIntrinsics, distortion_model: str
) -> tuple[ModuleType, Any]:
    if distortion_model not in _REALSENSE_DISTORTION_MODELS:
        raise ValueError(
            f"unsupported camera distortion model {intrinsics.distortion_model!r}"
        )
    coefficients = tuple(float(value) for value in intrinsics.distortion_coefficients)
    if len(coefficients) != 5 or not all(math.isfinite(value) for value in coefficients):
        raise ValueError(
            "RealSense distortion-aware deprojection requires five finite coefficients"
        )
    rs = _load_pyrealsense2()
    try:
        distortion = getattr(rs.distortion, distortion_model)
        sdk_intrinsics = rs.intrinsics()
        sdk_intrinsics.width = intrinsics.width_px
        sdk_intrinsics.height = intrinsics.height_px
        sdk_intrinsics.fx = intrinsics.fx_px
        sdk_intrinsics.fy = intrinsics.fy_px
        sdk_intrinsics.ppx = intrinsics.cx_px
        sdk_intrinsics.ppy = intrinsics.cy_px
        sdk_intrinsics.model = distortion
        sdk_intrinsics.coeffs = list(coefficients)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"pyrealsense2 cannot represent distortion model {distortion_model!r}"
        ) from exc
    return rs, sdk_intrinsics


def deproject_color_pixels(
    intrinsics: CameraIntrinsics,
    pixels_xy_px: NDArray[np.float64],
    depths_m: NDArray[np.float64],
) -> FloatArray:
    """Deproject colour pixels into camera-frame metres using audited intrinsics.

    The no-distortion path is vectorized and hardware independent.  A declared
    RealSense distortion model is never approximated as pinhole geometry.
    """

    pixels = np.asarray(pixels_xy_px, dtype=np.float64)
    depths = np.asarray(depths_m, dtype=np.float64)
    _validated_inputs(intrinsics, pixels, depths)
    if not len(pixels):
        return np.empty((0, 3), dtype=np.float64)

    distortion_model = _normalized_distortion_model(intrinsics.distortion_model)
    if distortion_model in _NO_DISTORTION_MODELS:
        return np.column_stack(
            (
                (pixels[:, 0] - intrinsics.cx_px) / intrinsics.fx_px * depths,
                (pixels[:, 1] - intrinsics.cy_px) / intrinsics.fy_px * depths,
                depths,
            )
        ).astype(np.float64, copy=False)

    rs, sdk_intrinsics = _realsense_intrinsics(intrinsics, distortion_model)
    try:
        points = np.asarray(
            [
                rs.rs2_deproject_pixel_to_point(
                    sdk_intrinsics,
                    [float(pixel[0]), float(pixel[1])],
                    float(depth_m),
                )
                for pixel, depth_m in zip(pixels, depths, strict=True)
            ],
            dtype=np.float64,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("RealSense distortion-aware metric deprojection failed") from exc
    if points.shape != (len(pixels), 3) or not np.all(np.isfinite(points)):
        raise ValueError("RealSense deprojection returned invalid camera-frame points")
    return points


def deproject_color_pixel(
    intrinsics: CameraIntrinsics,
    pixel_x_px: float,
    pixel_y_px: float,
    depth_m: float,
) -> FloatArray:
    """Deproject one colour pixel into a finite camera-frame point in metres."""

    points = deproject_color_pixels(
        intrinsics,
        np.asarray(((pixel_x_px, pixel_y_px),), dtype=np.float64),
        np.asarray((depth_m,), dtype=np.float64),
    )
    return np.asarray(points[0], dtype=np.float64)


__all__ = ["deproject_color_pixel", "deproject_color_pixels"]
